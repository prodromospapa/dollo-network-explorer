import asyncio
from playwright.async_api import async_playwright

async def verify():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1440, "height": 950})
        
        errors = []
        page.on("pageerror", lambda err: errors.append(str(err)))
        page.on("console", lambda msg: errors.append(f"Console error: {msg.text}") if msg.type == "error" else None)

        print("1. Loading https://prodromospapa.github.io/dollo-network-explorer/index.html ...")
        await page.goto("https://prodromospapa.github.io/dollo-network-explorer/index.html", wait_until="domcontentloaded")
        await page.wait_for_selector("#btn-view-tree")
        
        print("2. Opening Tree View modal...")
        await page.click("#btn-view-tree")
        await page.wait_for_selector("#tree-modal", state="visible")
        
        # Wait until tree layout is loaded
        print("  Waiting for TREE_LAYOUT to load...")
        await page.wait_for_function("() => typeof TREE_LAYOUT !== 'undefined' && TREE_LAYOUT !== null")
        await page.wait_for_timeout(600)
        
        # Verify default tax level is Kingdom
        tax_val = await page.evaluate("TREE_TAX_LEVEL")
        print(f"  Current TREE_TAX_LEVEL: {tax_val}")
        assert tax_val == "kingdom", f"Expected default taxonomy 'kingdom', got '{tax_val}'"
        
        # Verify rendered sectors
        sectors_info = await page.evaluate("""() => {
            const cladeBlocks = (TREE_LAYOUT.clade_levels && TREE_LAYOUT.clade_levels[TREE_TAX_LEVEL]) || [];
            return {
                num_blocks: cladeBlocks.length,
                clades: cladeBlocks.map(b => ({ clade: b.clade, count: b.species.length, start: b.start_idx, end: b.end_idx }))
            };
        }""")
        print(f"  Kingdom sectors count: {sectors_info['num_blocks']}")
        
        opistho_clades = [c for c in sectors_info['clades'] if c['start'] >= 58 and c['end'] <= 103]
        print("  Opisthokonta region sectors in Kingdom mode:")
        for c in opistho_clades:
            print(f"    {c['clade']:20s} len={c['count']:2d} [{c['start']}..{c['end']}]")
            
        clade_names = [c['clade'] for c in opistho_clades]
        assert "Opisthokonta" not in clade_names, "ERROR: Opisthokonta must NOT appear inside Kingdom level!"
        assert clade_names == ["Metazoa", "Choanoflagellata", "Ichthyosporea", "Fungi", "Cristidiscoidea"], f"Unexpected Kingdom clades: {clade_names}"
        print("  ✓ Kingdom clades are 100% clean and non-nested!")

        # Screenshot Kingdom Dark
        await page.screenshot(path="/home/prodromosp/.gemini/antigravity/brain/f336eccd-e2d4-4d35-b6ed-ca14e9e38462/screenshot_clean_kingdom_dark.png")
        print("  Captured screenshot_clean_kingdom_dark.png")

        # 3. Switch to Phylum
        print("3. Switching to Phylum...")
        await page.select_option("#tree-tax-select", "phylum")
        await page.wait_for_timeout(800)
        await page.screenshot(path="/home/prodromosp/.gemini/antigravity/brain/f336eccd-e2d4-4d35-b6ed-ca14e9e38462/screenshot_clean_phylum.png")
        print("  Captured screenshot_clean_phylum.png")

        # 4. Switch to Supergroup
        print("4. Switching to Supergroup...")
        await page.select_option("#tree-tax-select", "supergroup")
        await page.wait_for_timeout(800)
        
        sg_opistho = await page.evaluate("""() => {
            const b = TREE_LAYOUT.clade_levels['supergroup'].find(x => x.clade === 'Opisthokonta');
            return b ? { clade: b.clade, count: b.species.length, start: b.start_idx, end: b.end_idx } : null;
        }""")
        print(f"  Supergroup Opisthokonta block: {sg_opistho}")
        assert sg_opistho and sg_opistho['count'] == 46, f"Expected Opisthokonta to have 46 leaves in 1 block, got {sg_opistho}"
        print("  ✓ Opisthokonta in Supergroup is 100% unbroken 46-leaf block!")
        await page.screenshot(path="/home/prodromosp/.gemini/antigravity/brain/f336eccd-e2d4-4d35-b6ed-ca14e9e38462/screenshot_clean_supergroup.png")
        print("  Captured screenshot_clean_supergroup.png")

        # 5. Switch to Light mode
        print("5. Testing Tree Light mode...")
        await page.select_option("#tree-theme-select", "light")
        await page.select_option("#tree-tax-select", "kingdom")
        await page.wait_for_timeout(800)
        await page.screenshot(path="/home/prodromosp/.gemini/antigravity/brain/f336eccd-e2d4-4d35-b6ed-ca14e9e38462/screenshot_clean_kingdom_light.png")
        print("  Captured screenshot_clean_kingdom_light.png")

        print("Errors encountered:", len(errors))
        if errors:
            for e in errors: print(" ", e)
        assert len(errors) == 0, "Errors were found during test!"
        print("ALL AUTOMATED VERIFICATION CHECKS PASSED!")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(verify())
