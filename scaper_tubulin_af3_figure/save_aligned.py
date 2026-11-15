import sys
from Bio.PDB import MMCIFParser, Superimposer, PDBIO
D=sys.argv[1]; P=f"{D}/fold_scaper_n_terminal_and_one_tubulin_q9by12"
par=MMCIFParser(QUIET=True); ms=[par.get_structure(str(i),f"{P}_model_{i}.cif") for i in range(5)]
ref=[r['CA'] for r in ms[0][0]['B']]
for i,s in enumerate(ms):
    sup=Superimposer(); sup.set_atoms(ref,[r['CA'] for r in s[0]['B']]); sup.apply(list(s.get_atoms()))
    io=PDBIO(); io.set_structure(s); io.save(f"m{i}.pdb")
