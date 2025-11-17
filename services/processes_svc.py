from schemas import processes_schemas
from typing import List, Dict, Tuple, Optional

from newspaper import Article
import asyncio
import httpx # httpx 임포트

from google import genai
from dotenv import load_dotenv
import os

# [추가] S-BERT 관련 임포트
from sentence_transformers import SentenceTransformer, util
import torch # S-BERT는 내부적으로 PyTorch 사용

load_dotenv()
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
CUSTOM_SEARCH_JSON_API_KEY=os.getenv("CUSTOM_SEARCH_JSON_API_KEY")
CUSTOM_SEARCH_ENGINE_API_KEY=os.getenv("CUSTOM_SEARCH_ENGINE_API_KEY")

# [수정] 상수 정의 (검색 50개, 선별 5개)
MAX_NEWS_SEARCH = 50 # 총 검색할 뉴스 개수
NEWS_PER_REQUEST = 10 # 한 번의 API 요청당 뉴스 개수 (Google Custom Search 최대값)
TOP_N_NEWS = 5 # S-BERT로 선별할 최종 뉴스 개수

# [추가] S-BERT 모델 로드 (전역 변수로 한 번만 로드)
# 다국어 모델 사용 (회의록이나 뉴스에 영어가 섞일 수 있으므로)
# (모델 로드에 시간이 걸릴 수 있으나, 서버 시작 시 1회만 수행됨)
print("S-BERT 모델 로드 중...")
try:
    sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("S-BERT 모델 로드 완료: paraphrase-multilingual-MiniLM-L12-v2")
except Exception as e:
    print(f"S-BERT 모델 로드 실패: {e}")
    sbert_model = None


async def summary_meeting_and_keyword_meeting_by_gemini(
    original_meeting: str
):
    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"회의록: {original_meeting}"

    response = await client.aio.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": processes_schemas.SummaryMeetingAndKeywordMeetingByLLM.model_json_schema(),
        },
    )

    summary_meeting_and_keyword_meeting = processes_schemas.SummaryMeetingAndKeywordMeetingByLLM.model_validate_json(response.text)

    summary_meeting = summary_meeting_and_keyword_meeting.summary_meeting
    keyword_meeting_list = summary_meeting_and_keyword_meeting.keyword_meeting_list

    return summary_meeting, keyword_meeting_list


# [수정] Google Custom Search API 병렬 요청 (총 50개)
async def news_url_list_by_custom_search_json_api(
    keyword_meeting_list: List[str]
) -> List[str]:
    
    if not CUSTOM_SEARCH_JSON_API_KEY or not CUSTOM_SEARCH_ENGINE_API_KEY:
        print("오류: .env 파일에 GOOGLE_SEARCH_CX 또는 GOOGLE_API_KEY가 없습니다.")
        return []

    query = " ".join(keyword_meeting_list) + " news -filetype:pdf"
    API_URL = "https://www.googleapis.com/customsearch/v1"
    
    # 1. 총 50개 검색 (10개씩 5번)
    num_requests = MAX_NEWS_SEARCH // NEWS_PER_REQUEST # 50 // 10 = 5
    start_indices = [1 + i * NEWS_PER_REQUEST for i in range(num_requests)] # [1, 11, 21, 31, 41]

    tasks = []
    
    # [수정] httpx.AsyncClient에 timeout=30.0 (30초) 설정 추가
    # (connect=30, read=30, write=30, pool=30)
    timeout_config = httpx.Timeout(30.0) 
    
    async with httpx.AsyncClient(timeout=timeout_config) as client:
        for start_index in start_indices:
            params = {
                "key": CUSTOM_SEARCH_JSON_API_KEY,
                "cx": CUSTOM_SEARCH_ENGINE_API_KEY,
                "q": query,
                "num": NEWS_PER_REQUEST, # 10개씩 요청
                "start": start_index,
                "sort": "date" # 최신순 정렬
            }
            # 2. 비동기 요청 태스크 생성
            tasks.append(client.get(API_URL, params=params))

        print(f"DEBUG: Google Custom Search 병렬 요청 {len(tasks)}개 시작 (타임아웃 30초)")
        
        try:
            # 3. asyncio.gather로 모든 요청을 병렬 처리
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            print(f"DEBUG: asyncio.gather 실행 중 오류: {e}")
            return []

    all_news_urls = set() # 중복 URL 제거용 Set
    
    for idx, res in enumerate(responses):
        # 4. 예외 처리
        if isinstance(res, Exception):
            # [수정] 어떤 예외(Exception)인지 자세히 출력
            print(f"오류: Google Custom Search API 호출 실패 (Request {idx+1}): {type(res).__name__} - {res}")
            continue
            
        if res.status_code != 200:
            # [수정] HTTP 오류일 경우, 응답 본문(res.text)을 자세히 출력 (API 키 오류 등 확인)
            print(f"오류: Google Custom Search API 호출 실패 (Request {idx+1}, Status {res.status_code})")
            print(f"  -> 응답: {res.text}")
            continue
        
        try:
            # 5. 결과 파싱 및 Set에 추가 (중복 제거)
            results = res.json()
            urls = [item['link'] for item in results.get('items', [])]
            for url in urls:
                all_news_urls.add(url) 
                
        except Exception as e:
            print(f"오류: Google Custom Search 결과 파싱 실패 (Request {idx+1}): {e}")

    news_url_list = list(all_news_urls)
    print(f"DEBUG: 총 {len(news_url_list)}개의 고유한 URL 검색 완료.")

    return news_url_list


def _crawl_one_article(url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    단일 URL을 크롤링하여 (제목, 본문) 튜플을 반환합니다.
    (이 함수는 동기 함수이며, 별도 스DML 스레드에서 실행됩니다.)
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        # 제목과 본문이 모두 있어야 성공
        if article.title and article.text:
            return article.title, article.text
        else:
            return None, None
    except Exception as e:
        print(f"Newspaper3k 크롤링 오류 (URL: {url}): {e}")
        return None, None

async def original_news_list_by_newspaper3k(
    news_url_list: List[str]
) -> List[Dict[str, Optional[str]]]:
    """
    뉴스 URL 리스트를 병렬로 크롤링하여,
    'news_items' DB 컬럼에 저장할 딕셔너리 리스트를 반환합니다.
    (요청서에 따라 기존 로직 유지)
    """
    
    if not news_url_list:
        return []

    print(f"DEBUG: Newspaper3k 크롤링 시작 (총 {len(news_url_list)}개 URL)")

    # 1. 'asyncio.to_thread'를 사용하여 동기 함수인 _crawl_one_article을
    #    비동기 태스크로 만듭니다.
    tasks = []
    for url in news_url_list:
        tasks.append(asyncio.to_thread(_crawl_one_article, url))

    # 2. 'asyncio.gather'로 모든 크롤링 작업을 병렬로 실행합니다.
    results = await asyncio.gather(*tasks)

    # 3. 원본 URL 리스트와 크롤링 결과 리스트를 'zip'으로 묶습니다.
    zipped_results = zip(news_url_list, results)

    # 4. 'news_items' DB 구조에 맞는 딕셔너리 리스트를 생성합니다.
    news_items_list = []
    idx = 1
    
    for url, (title, text) in zipped_results:
        # 크롤링에 성공한 (title과 text가 모두 있는) 경우에만
        if title and text:
            # DB에 저장할 객체(딕셔너리) 생성
            news_item = {
                "url": url,
                "title": title,      # 제목도 함께 저장
                "original": text,    # 본문
                "summary": None      # "summary는 아직 없어" 요청 반영
            }
            news_items_list.append(news_item)
            
            # 디버그 출력 (본문이 길 수 있으므로 200자만 출력)
            # print(f"  [크롤링 {idx}] URL: {url}, TITLE: {title}, TEXT: {text[:200]}...") 
            idx += 1
        else:
            print(f"DEBUG: 크롤링 실패 또는 내용 없음 (URL: {url})")

    print(f"DEBUG: 크롤링 성공 (총 {len(news_items_list)}개 / {len(news_url_list)}개 시도)")
    
    # 5. DB에 저장할 딕셔너리의 리스트를 반환합니다.
    return news_items_list


# [신규] S-BERT 관련도 분석 함수 (CPU-Bound)
def _run_sbert_similarity(
    summary_meeting: str, 
    news_items_list: List[Dict[str, Optional[str]]]
) -> List[Dict[str, Optional[str]]]:
    """
    [동기 함수] S-BERT를 사용하여 회의록 요약본과 뉴스 본문 간의 유사도를 계산.
    (asyncio.to_thread로 실행되어야 함)
    """
    
    if not sbert_model:
        print("  [S-BERT] 오류: S-BERT 모델이 로드되지 않았습니다. 상위 5개 뉴스를 그대로 반환합니다.")
        return news_items_list[:TOP_N_NEWS]

    if not news_items_list:
        return []
        
    print(f"  [S-BERT] S-BERT 인코딩 시작 (회의록 1개, 뉴스 {len(news_items_list)}개)")
    
    try:
        # 1. 회의록 요약본 임베딩
        meeting_embedding = sbert_model.encode(
            summary_meeting, 
            convert_to_tensor=True
        )
        
        # 2. 뉴스 본문(original) 리스트 임베딩
        corpus_texts = [
            item['original'] for item in news_items_list if item.get('original')
        ]
        # 'original'이 있는 아이템만 필터링 (중요)
        valid_news_items = [
            item for item in news_items_list if item.get('original')
        ]
        
        if not valid_news_items:
            print("  [S-BERT] 오류: 유효한 뉴스 본문이 없습니다.")
            return []

        corpus_embeddings = sbert_model.encode(
            corpus_texts, 
            convert_to_tensor=True
        )
        
        # 3. 코사인 유사도 계산
        cosine_scores = util.cos_sim(meeting_embedding, corpus_embeddings)[0] # [N]
        
        # 4. 상위 Top-N (TOP_N_NEWS) 선별
        scores_with_indices = list(zip(cosine_scores, range(len(valid_news_items))))
        
        # 점수(score) 기준으로 내림차순 정렬
        sorted_scores = sorted(scores_with_indices, key=lambda x: x[0], reverse=True)
        
        # 상위 N개 인덱스 추출
        top_n_indices = [idx for score, idx in sorted_scores[:TOP_N_NEWS]]
        
        # 상위 N개 뉴스 아이템 딕셔너리 반환
        top_n_news_items = [valid_news_items[i] for i in top_n_indices]
        
        print(f"  [S-BERT] 인코딩 및 유사도 계산 완료. Top {len(top_n_news_items)}개 선별.")
        
        # (디버그용) 상위 점수 출력
        for i, (score, idx) in enumerate(sorted_scores[:TOP_N_NEWS]):
             print(f"    [Top {i+1}] Score: {score.item():.4f}, URL: {valid_news_items[idx]['url']}")

        return top_n_news_items
        
    except Exception as e:
        print(f"  [S-BERT] 오류: S-BERT 처리 중 예외 발생: {e}")
        # 오류 발생 시, S-BERT 선별 없이 그냥 상위 N개 반환 (Fallback)
        return news_items_list[:TOP_N_NEWS]


async def cosine_similarity_by_sbert(
    summary_meeting: str,
    news_items_list: List[Dict[str, Optional[str]]]
) -> List[Dict[str, Optional[str]]]:
    """
    [비동기 래퍼] S-BERT (CPU-Bound) 작업을 별도 스레드에서 실행합니다.
    """
    print(f"DEBUG: S-BERT 유사도 분석 (asyncio.to_thread) 실행...")
    
    # 동기 함수인 _run_sbert_similarity를 별도 스레드에서 실행
    top_n_news = await asyncio.to_thread(
        _run_sbert_similarity,
        summary_meeting,
        news_items_list
    )
    
    return top_n_news


async def summary_news_list_by_gemini(
    news_items_list_selected: List[Dict[str, Optional[str]]]
) -> List[Dict[str, Optional[str]]]:
    
    client = genai.Client(api_key=GEMINI_API_KEY)

    print(f"DEBUG: Gemini 뉴스 요약 시작 (순차 실행, 총 {len(news_items_list_selected)}개)")

    for i in range(len(news_items_list_selected)):

        # (방어 코드)
        if not news_items_list_selected[i].get('original'):
            print(f"  [Gemini] 요약 건너뛰기 (원본 없음): {news_items_list_selected[i]['url']}")
            news_items_list_selected[i]["summary"] = "뉴스 원본을 가져올 수 없어 요약에 실패했습니다."
            continue

        print(f"  [Gemini] 요약 중 ({i+1}/{len(news_items_list_selected)}): {news_items_list_selected[i]['url']}")
        prompt = f"다음 뉴스를 한국어로 3~4문장으로 요약해줘: {news_items_list_selected[i]['original']}" # 프롬프트 개선

        try:
            response = await client.aio.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": processes_schemas.SummaryNewsByLLM.model_json_schema(),
                },
            )

            summary_news = processes_schemas.SummaryNewsByLLM.model_validate_json(response.text)
            summary = summary_news.summary
            news_items_list_selected[i]["summary"] = summary
        
        except Exception as e:
            print(f"  [GemM N] 오류: Gemini API 호출 실패 (URL: {news_items_list_selected[i]['url']}): {e}")
            news_items_list_selected[i]["summary"] = "Gemini API 호출 중 오류가 발생하여 요약에 실패했습니다."

    print(f"DEBUG: Gemini 뉴스 요약 완료.")
    return news_items_list_selected