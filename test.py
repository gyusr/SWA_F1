import httpx
async def test():
    async with httpx.AsyncClient(timeout=5) as c:
        r = await c.get("https://www.google.com")
        print(r.status_code)

test()