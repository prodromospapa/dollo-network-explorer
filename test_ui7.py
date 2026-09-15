import asyncio
from playwright.async_api import async_playwright
import subprocess, time

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        errors = []
        page.on("pageerror", lambda err: errors.append(err.message))
        page.on("console", lambda msg: errors.append(f"Console: {msg.text}") if msg.type == "error" else None)
        
        server = subprocess.Popen(['python3', '-m', 'http.server', '8080'])
        time.sleep(1)
        
        try:
            await page.goto('http://localhost:8080/index.html')
            await page.wait_for_timeout(1000)
            
            # test search
            await page.fill('#search', 'SCAPER')
            await page.wait_for_timeout(500)
            await page.keyboard.press('Enter')
            await page.wait_for_timeout(2000)
            
            gene_info = await page.evaluate("document.getElementById('gene-info').innerText")
            print("Gene Info:", gene_info)
            
            print("Errors:", errors)
        finally:
            server.terminate()
            await browser.close()

asyncio.run(main())
