import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        errors = []
        page.on("pageerror", lambda err: errors.append(err.message))
        page.on("console", lambda msg: errors.append(f"Console: {msg.text}") if msg.type == "error" else None)
        
        try:
            await page.goto('https://prodromospapa.github.io/dollo-network-explorer/index.html', wait_until='networkidle')
            print("Page loaded")
            
            # set threshold to 0.05 and topn to 200
            await page.fill('#thresh', '0.05')
            await page.fill('#topn', '200')
            await page.evaluate("document.getElementById('thresh').dispatchEvent(new Event('input'))")
            await page.evaluate("document.getElementById('topn').dispatchEvent(new Event('input'))")
            
            # test search SCAPER
            await page.fill('#search', 'SCAPER')
            await page.wait_for_timeout(500)
            await page.keyboard.press('Enter')
            await page.wait_for_timeout(3000)
            
            gene_info = await page.evaluate("document.getElementById('gene-info').innerText")
            print("Gene Info:", gene_info)
            print("Errors:", errors)
        finally:
            await browser.close()

asyncio.run(main())
