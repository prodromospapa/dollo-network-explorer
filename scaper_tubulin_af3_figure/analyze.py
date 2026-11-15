import json, sys, numpy as np
from Bio.PDB import MMCIFParser, Superimposer
D=sys.argv[1]; P=f"{D}/fold_scaper_n_terminal_and_one_tubulin_q9by12"
par=MMCIFParser(QUIET=True)
models=[par.get_structure(str(i),f"{P}_model_{i}.cif")[0] for i in range(5)]
ca=lambda m,c:[r['CA'] for r in m[c]]
ref=models[0]
for i,m in enumerate(models):
    s=Superimposer(); s.set_atoms(ca(ref,'B'),ca(m,'B')); s.apply(list(m.get_atoms()))
    a=np.array([x.coord for x in ca(m,'A')]); a0=np.array([x.coord for x in ca(ref,'A')])
    b=np.array([x.coord for x in ca(m,'B')])
    d=np.linalg.norm(a[:,None]-b[None],axis=2)
    ncont=(d<8).any(1).sum()
    print(i,"tubRMSD %.2f"%s.rms,"ScaperRMSD_vs0(tub-aligned) %.1f"%np.sqrt(((a-a0)**2).sum(1).mean()),
          "centroid shift %.1f"%np.linalg.norm(a.mean(0)-a0.mean(0)),"A res within 8A of tub:",ncont, "min dist %.1f"%d.min())
    pl=np.array([r['CA'].bfactor for r in m['A']]); print("  scaper pLDDT mean %.1f, frac>70 %.2f"%(pl.mean(),(pl>70).mean()))
    fd=json.load(open(f"{P}_full_data_{i}.json")); pae=np.array(fd['pae']); cp=np.array(fd['contact_probs'])
    ch=np.array(fd['token_chain_ids']); A=ch=='A'; B=ch=='B'
    print("  inter PAE mean %.1f; max contact prob A-B %.2f"%(pae[np.ix_(A,B)].mean(),cp[np.ix_(A,B)].max()))
