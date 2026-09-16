import asyncio
import subprocess
import time
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=['--use-gl=swiftshader', '--enable-webgl', '--no-sandbox', '--disable-setuid-sandbox']
        )
        page = await browser.new_page()
        
        errors = []
        page.on("pageerror", lambda err: errors.append(f"PageError: {err.message}"))
        page.on("console", lambda msg: errors.append(f"ConsoleError: {msg.text}") if msg.type == "error" else None)
        
        try:
            print("1. Loading GitHub Pages index.html...")
            await page.goto('https://prodromospapa.github.io/dollo-network-explorer/index.html', wait_until='networkidle')
            await page.wait_for_timeout(2000)
            
            # Check WebGL Whole Network mode
            status_text = await page.evaluate("document.getElementById('graph-status').innerText")
            print("Status text:", status_text)
            
            sigma_nodes = await page.evaluate("typeof sigmaGraph !== 'undefined' && sigmaGraph ? sigmaGraph.order : 0")
            sigma_edges = await page.evaluate("typeof sigmaGraph !== 'undefined' && sigmaGraph ? sigmaGraph.size : 0")
            print(f"WebGL Sigma graph: {sigma_nodes} nodes, {sigma_edges} edges")
            assert sigma_nodes == 11236, f"Expected 11236 nodes in Sigma, got {sigma_nodes}"
            assert sigma_edges > 70000, f"Expected >70000 edges in Sigma, got {sigma_edges}"
            
            # Test Search SCAPER in Whole Network mode
            print("2. Searching SCAPER...")
            await page.fill('#search', 'SCAPER')
            await page.wait_for_timeout(500)
            await page.keyboard.press('Enter')
            await page.wait_for_timeout(1000)
            
            gene_info = await page.evaluate("document.getElementById('gene-info').innerText")
            print("SCAPER Gene Info:\n", gene_info)
            assert "SCAPER" in gene_info, "Expected SCAPER in gene info"
            
            # Check partner count
            partner_count = await page.evaluate("document.querySelectorAll('#partner-list .partner').length")
            print(f"Rendered {partner_count} partners in sidebar")
            assert partner_count > 0, "Expected partners rendered in sidebar"
            
            # Switch to Cytoscape Gene Focus mode
            print("3. Switching to Gene Focus mode (Cytoscape)...")
            await page.click('#btn-view-cy')
            await page.wait_for_timeout(1000)
            
            cy_nodes = await page.evaluate("typeof cy !== 'undefined' && cy ? cy.nodes().length : 0")
            cy_edges = await page.evaluate("typeof cy !== 'undefined' && cy ? cy.edges().length : 0")
            print(f"Cytoscape Ego graph: {cy_nodes} nodes, {cy_edges} edges")
            assert cy_nodes >= 25, f"Expected >=25 nodes in Cytoscape, got {cy_nodes}"
            assert cy_edges >= 25, f"Expected >=25 edges in Cytoscape, got {cy_edges}"
            
            # Test View Cluster button
            print("4. Testing View Cluster C6...")
            cluster_btn = await page.query_selector("button:has-text('View Cluster →')")
            if cluster_btn:
                await cluster_btn.click()
                await page.wait_for_timeout(1000)
                cluster_info = await page.evaluate("document.getElementById('gene-info').innerText")
                print("Cluster Info:\n", cluster_info)
                assert "C6" in cluster_info or "cilium" in cluster_info
                
                cl_nodes = await page.evaluate("cy.nodes().length")
                cl_edges = await page.evaluate("cy.edges().length")
                print(f"Cytoscape Cluster C6 graph: {cl_nodes} nodes, {cl_edges} edges")
                assert cl_nodes > 0, "Expected cluster nodes in Cytoscape"
                
            # Test Switch back to Whole Network
            print("5. Testing Switch back to Whole Network...")
            await page.click('#btn-view-whole')
            await page.wait_for_timeout(500)
            current_mode = await page.evaluate("currentMode")
            assert current_mode == 'whole', f"Expected mode whole, got {current_mode}"
            
            print("\n==========================================")
            print("ALL DUAL-ENGINE VERIFICATIONS PASSED 100%!")
            print("Total console/page errors:", len(errors))
            if errors:
                print("Errors:", errors)
            print("==========================================")
                
        finally:
            await browser.close()

asyncio.run(main())
