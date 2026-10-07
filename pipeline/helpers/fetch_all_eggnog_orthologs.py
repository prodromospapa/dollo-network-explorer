import csv
import json
import sys
import time
import pickle
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path("/home/prodromosp/dollo-network-explorer")
OUT = ROOT / "data" / "eggnog_all_genes"
CACHE = OUT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

UA = "Mozilla/5.0 (compatible; research-script/1.0; contact:prodromospapa@gmail.com)"
SLEEP = 0.3  # politeness delay between live HTTP calls

def http_get_json(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def cached_get(url, cache_key, timeout=20):
    cache_file = CACHE / f"{cache_key}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except Exception:
            pass
    try:
        data = http_get_json(url, timeout=timeout)
        cache_file.write_text(json.dumps(data))
        time.sleep(SLEEP)
        return data
    except urllib.error.HTTPError as e:
        err = {"_error": f"HTTP {e.code}", "url": url}
        cache_file.write_text(json.dumps(err))
        time.sleep(SLEEP)
        return err
    except Exception as e:
        err = {"_error": str(e), "url": url}
        cache_file.write_text(json.dumps(err))
        time.sleep(SLEEP)
        return err

def load_all_genes():
    # Load the 19,758 gene symbols from the orthogroup matrix index
    pkl_path = "/home/prodromosp/scaper_new/orthogroups.pkl"
    with open(pkl_path, "rb") as f:
        df = pickle.load(f)
    return list(df.index)

def batch_get_ensembl_proteins(genes):
    import mygene
    mg = mygene.MyGeneInfo()
    print(f"Querying MyGene for {len(genes)} genes in batch...")
    results = mg.querymany(genes, scopes='symbol', species='human', fields='ensembl.protein', as_dataframe=False)
    
    gene_to_ensp = {}
    for r in results:
        query = r['query']
        ens = r.get('ensembl', {})
        if isinstance(ens, list):
            prot = []
            for e in ens:
                p = e.get('protein', [])
                if isinstance(p, str):
                    p = [p]
                prot.extend(p)
            gene_to_ensp[query] = prot
        else:
            prot = ens.get('protein', [])
            if isinstance(prot, str):
                prot = [prot]
            if prot:
                gene_to_ensp[query] = prot
    return gene_to_ensp

def resolve_eggnog_protid(symbol, ensembl_protein_ids):
    for ensp in ensembl_protein_ids:
        protid = f"9606.{ensp}"
        url = f"https://eggnogdb.org/e7api/protein/{protid}/"
        data = cached_get(url, f"eggnog_protein_{protid}")
        if "_error" in data:
            continue
        focus = data.get("focus_prot", {})
        bestname = (focus.get("bestname") or "").upper()
        names = [n.strip().upper() for n in bestname.replace(",", " ").split()]
        if symbol.upper() in names or symbol.upper() == bestname.upper():
            return protid, focus
    return None, None

def fetch_orthologs(protid):
    url = f"https://eggnogdb.org/e7api/orthologs/{protid}/"
    data = cached_get(url, f"eggnog_orthologs_{protid}")
    if "_error" in data:
        return None, data["_error"]
    return data.get("orthologs", {}), None

def main():
    genes = load_all_genes()
    print(f"Loaded {len(genes)} genes", file=sys.stderr)
    
    # Load cached ensp mapping if available to save mygene quota
    ensp_cache = OUT / "mygene_ensp_map.json"
    if ensp_cache.exists():
        gene_to_ensp = json.loads(ensp_cache.read_text())
    else:
        gene_to_ensp = batch_get_ensembl_proteins(genes)
        ensp_cache.write_text(json.dumps(gene_to_ensp))

    resolution_log = []
    
    # Check progress
    existing_orthos = list(CACHE.glob("eggnog_orthologs_*.json"))
    print(f"Found {len(existing_orthos)} already cached ortholog responses.", file=sys.stderr)

    for i, symbol in enumerate(genes, 1):
        if i % 100 == 0:
            print(f"[{i}/{len(genes)}] {symbol}", file=sys.stderr)
            
        prot_ids = gene_to_ensp.get(symbol, [])
        if not prot_ids:
            resolution_log.append((symbol, "FAIL_MYGENE", "no protein IDs"))
            continue

        protid, focus = resolve_eggnog_protid(symbol, prot_ids)
        if not protid:
            resolution_log.append((symbol, "FAIL_EGGNOG_MATCH", f"{len(prot_ids)} candidates, none matched"))
            continue

        orthologs, err = fetch_orthologs(protid)
        if err:
            resolution_log.append((symbol, "FAIL_ORTHOLOGS", err))
            continue

        resolution_log.append((symbol, "OK", protid))

    with open(OUT / "resolution_log.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["symbol", "status", "detail"])
        w.writerows(resolution_log)

    n_ok = sum(1 for r in resolution_log if r[1] == "OK")
    print(f"\nDone. {n_ok}/{len(genes)} genes resolved to eggNOG orthologs.", file=sys.stderr)

if __name__ == "__main__":
    main()
