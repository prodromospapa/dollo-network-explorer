import gzip, json, sys, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.lines
from matplotlib.patches import Rectangle
from PIL import Image, ImageChops
S, REPO = sys.argv[1], sys.argv[2]
FO = f'{S}/folds'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
INK, INK2, MUTED = '#0f172a', '#475569', '#94a3b8'
HUM, TRY = '#d63838', '#6b6b78'

# ---- alignment block (human 612-643) from the TCS orthogroup SCAPER alignment
a = json.load(gzip.open(f'{REPO}/alignments_orthogroup/SCAPER.json.gz'))
hum = a['rows'][0]['s']; cols = []; hi = 0
for i, ch in enumerate(hum):
    if ch != '-': hi += 1
    if ch != '-' and 612 <= hi <= 643: cols.append((i, hi))
rows = {r['sp']: ''.join(r['s'][i] for i, _ in cols) for r in a['rows']}
present = [('Homo_sapiens', 'Homo sapiens'), ('Danio_rerio', 'Danio rerio'), ('Drosophila_melanogaster', 'Drosophila melanogaster'),
           ('Naegleria_gruberi', 'Naegleria gruberi'), ('Euglena_gracilis', 'Euglena gracilis')]
absent = [('Leishmania_major', 'Leishmania major'),
          ('Trypanosoma_brucei', 'Trypanosoma brucei'), ('Trypanosoma_cruzi', 'Trypanosoma cruzi')]
contacts = {622, 624, 625, 628, 629, 632, 633, 636, 639, 640}   # <4 Å to CNOT11 in 5/5 human AF3 models
# Interface residues mutated in Hopfler et al. 2023 (Mol Cell, doi:10.1016/j.molcel.2023.05.020); every mutant abolished autoregulation
MUT = {618: 'K', 625: 'K', 628: 'S', 629: 'K', 632: 'S', 633: 'K', 640: 'K'}
hot = set(MUT)
FOLDX = {636}   # not mutated in the paper, but FoldX ddG_bind ~1.6 kcal/mol to CNOT11
def clustal(c):
    if c in 'AILMFWVC': return '#80a0f0' if c != 'C' else '#f08080'
    if c in 'KR': return '#f01505'
    if c in 'DE': return '#c048c0'
    if c in 'NQST': return '#15c015'
    if c == 'G': return '#f09048'
    if c == 'P': return '#c0c000'
    if c in 'HY': return '#15a4a4'
    return None

fig = plt.figure(figsize=(16, 9), dpi=200, facecolor='white')
axA = fig.add_axes([0.20, 0.545, 0.68, 0.35])
n = len(cols); rows['MUT'] = ''.join(MUT.get(h, ' ') for _, h in cols)
order = present + [None] + absent
y = 0; ylab = []
for item in order:
    if item is None: y += 0.6; continue
    sp, lab = item; s = rows[sp]
    if sp == 'MUT':
        axA.add_patch(Rectangle((0, -y - 1), n, 1, fc='#f8fafc', ec='none'))
    for x, c in enumerate(s):
        col = clustal(c) if c != ' ' else None
        if sp == 'MUT' and col: col = '#fde68a'
        if col: axA.add_patch(Rectangle((x, -y - 1), 1, 1, fc=col, ec='white', lw=0.8, alpha=0.9))
        axA.text(x + 0.5, -y - 0.5, c, ha='center', va='center', fontsize=10.5, family='DejaVu Sans Mono',
                 color='#111' if col else MUTED, fontweight='bold' if sp == 'Homo_sapiens' else 'normal')
    ylab.append((-y - 0.5, lab, sp)); y += 1
for x, (_, h) in enumerate(cols):
    if h in hot:
        axA.add_patch(Rectangle((x, -y), 1, y, fill=False, ec='#000000', lw=3.6, zorder=5))
    if h in FOLDX:
        axA.add_patch(Rectangle((x, -y), 1, y, fill=False, ec='#000000', lw=3.0, ls=(0, (2.5, 1.5)), zorder=5))
    if h % 5 == 0 or h in hot | FOLDX:
        axA.text(x + 0.5, 1.05, str(h), ha='center', va='bottom', fontsize=9, color=INK2 if h not in hot | FOLDX else INK,
                 fontweight='bold' if h in hot | FOLDX else 'normal')
for yy, lab, sp in ylab:
    axA.text(-0.4, yy, lab, ha='right', va='center', fontsize=11, style='normal' if sp == 'MUT' else 'italic',
             color=INK, fontweight='bold' if sp in ('Homo_sapiens', 'Euglena_gracilis', 'Trypanosoma_brucei') else 'normal')
# group brackets
def bracket(y0, y1, txt, col):
    xb = n + 0.5
    axA.plot([xb, xb], [y0, y1], color=col, lw=3, clip_on=False, solid_capstyle='butt')
    axA.text(xb + 0.5, (y0 + y1) / 2, txt, ha='left', va='center', fontsize=11, color=col, fontweight='bold')
bracket(-0.1, -len(present) + 0.1, 'CNOT11\npresent', '#2563eb')
bracket(-len(present) - 0.7, -len(present) - len(absent) - 0.5, 'CNOT11\nabsent', '#9a3412')
# mutant combinations tested in Hopfler et al. 2023: one bracket per mutant, ticks at each mutated residue
xi = {h: x for x, (_, h) in enumerate(cols)}
COMBOS = [((618, 625), 0), ((628, 632), 0), ((633, 640), 0), ((628, 629, 632), 1)]
for res, lev in COMBOS:
    xs_ = [xi[h] + 0.5 for h in res]; yb = -y - 0.3 - 1.05 * lev
    ytop = -y - 0.05 if lev == 0 else yb + 0.22   # lower-level brackets get short ticks so they don't cross labels
    axA.plot([xs_[0], xs_[-1]], [yb, yb], color='#000', lw=2.0, solid_capstyle='butt', zorder=6)
    for xx in xs_:
        axA.plot([xx, xx], [ytop, yb], color='#000', lw=2.0, solid_capstyle='butt', zorder=6)
    axA.text((xs_[0] + xs_[-1]) / 2, yb - 0.1, ' + '.join(f'{rows["Homo_sapiens"][xi[h]]}{h}{MUT[h]}' for h in res),
             ha='center', va='top', fontsize=9.5, color=INK, fontweight='bold')
axA.set_xlim(0, n); axA.set_ylim(-y - 2.0, 1.9); axA.axis('off')
fig.text(0.015, 0.955, 'A', fontsize=20, fontweight='bold', color=INK)
fig.text(0.04, 0.955, 'SCAPER helix that contacts CNOT11 (human 612–643), TCS orthogroup alignment',
         fontsize=14, color=INK, va='baseline')


from matplotlib.patches import Rectangle as R2
fig.patches.append(R2((0.045, 0.902), 0.011, 0.02, transform=fig.transFigure, fill=False, ec='#000000', lw=3.6))
fig.text(0.06, 0.912, 'SCAPER residues mutated at the CNOT11 interface (Höpfler et al., 2023)', fontsize=10.5, color=INK2, va='center')
fig.patches.append(R2((0.42, 0.902), 0.011, 0.02, transform=fig.transFigure, fill=False, ec='#000000', lw=3.0, ls=(0, (2.5, 1.5))))
fig.text(0.435, 0.912, 'FoldX hotspot, not tested experimentally (ΔΔG_bind ≈ 1.6 kcal/mol)', fontsize=10.5, color=INK2, va='center')
# ---- structures (crop all renders to the same box so the scale matches)
EUG = '#1b9e77'
ims = [Image.open(f'{S}/fig/{f}.png').convert('RGB') for f in ('human', 'euglena', 'tbrucei')]
bbs = [ImageChops.difference(im, Image.new('RGB', im.size, (255, 255, 255))).getbbox() for im in ims]
box = (min(b[0] for b in bbs) - 20, min(b[1] for b in bbs) - 20, max(b[2] for b in bbs) + 20, max(b[3] for b in bbs) + 20)
ims = [im.crop(box) for im in ims]
def conf(path):
    cs = [json.load(open(path.format(i))) for i in range(5)]
    return [c['iptm'] for c in cs], [c['chain_pair_pae_min'][0][1] for c in cs]
hu_ip, hu_pae = conf(f'{FO}/scaper_region_cnot11/fold_scaper_region_cnot11_summary_confidences_{{}}.json')
eu_ip, eu_pae = conf(f'{FO}/euhu/fold_euglena_cnot11_human_summary_confidences_{{}}.json')
ce_ip, ce_pae = conf(f'{FO}/cehu/fold_c_elegancs_cnot11_human_summary_confidences_{{}}.json')
CEL = '#b5540a'
tb_ip, tb_pae = conf(f'{FO}/trypanosoma_region_cnot11/fold_trypanosoma_region_cnot11_summary_confidences_{{}}.json')
panels = [('B', ims[0], 'Human', 'SCAPER 593–672', '+ human CNOT11', HUM, hu_ip, hu_pae, 0.03),
          ('C', ims[1], 'E. gracilis', 'SCAPER 319–398', '+ human CNOT11', EUG, eu_ip, eu_pae, 0.225),
          ('D', ims[2], 'T. brucei', 'SCAPER 214–293', '+ human CNOT11', TRY, tb_ip, tb_pae, 0.42)]
for letter, im, sp, title, sub, col, ip, pae, x0 in panels:
    ax = fig.add_axes([x0, 0.04, 0.18, 0.35]); ax.imshow(im); ax.axis('off')
    fig.text(x0 - 0.015, 0.475, letter, fontsize=20, fontweight='bold', color=INK)
    fig.text(x0 + 0.01, 0.475, sp, fontsize=14, color=col, fontweight='bold', style='normal' if sp == 'Human' else 'italic')
    fig.text(x0 + 0.01, 0.448, title, fontsize=9.5, color=INK2)
    fig.text(x0 + 0.01, 0.428, sub, fontsize=9.5, color=INK2)
    fig.text(x0 + 0.01, 0.395, f'ipTM {np.mean(ip):.2f} · PAE {np.mean(pae):.1f} Å', fontsize=11, color=INK, fontweight='bold')
fig.text(0.03, 0.015, 'All SCAPER fragments folded with the same human CNOT11 (C-terminal domain shown). CNOT11 presence/absence from the TCS orthogroups. E. gracilis SCAPER with its own CNOT11: ipTM 0.75.', fontsize=9.5, color=INK2)

# ---- ipTM per model
# FoldX (foldx_scaper/results/region615_640): Ala scan of SCAPER 615-640 vs CNOT11 and beta-tubulin, plus the
# Hopfler E633K/D640K double mutant (results/variants/{cnot11,tubulin}/variants_raw.tsv, chain B partner)
import csv
FXD = '/mnt/home2/prodromosp/foldx_scaper/results'
scan = list(csv.DictReader(open(f'{FXD}/region615_640/region_compare.tsv'), delimiter='\t'))
def variant(part, name):
    r = next(r for r in csv.DictReader(open(f'{FXD}/variants/{part}/variants_raw.tsv'), delimiter='\t') if r['name'] == name)
    return float(r['ddG_B']), float(r['sd_B'])
HOP = [('EE-KK (Hopfler)', 'E618K/E625K'), ('FL-SS (Hopfler)', 'F628S/L632S'),
       ('FIL-SKS (Hopfler)', 'F628S/I629K/L632S'), ('ED-KK (Hopfler)', 'E633K/D640K')]
hop = [(*variant('cnot11', k), *variant('tubulin', k)) for k, _ in HOP]
CN, TU = '#2a78b5', '#c4572a'
axD = fig.add_axes([0.655, 0.165, 0.32, 0.24])
labs = [r['mutation'][:-1] for r in scan]
groups = [(float(r['ddG_CNOT11']), float(r['ddG_CNOT11_sd']), float(r['ddG_tub_b2']), float(r['ddG_tub_b2_sd'])) for r in scan]
xs = list(range(len(scan))); xh = [len(scan) + 0.9 + 1.0 * i for i in range(len(HOP))]
for x, (c, cs, tu, ts) in zip(xs + xh, groups + hop):
    for dx, y, sd, col in ((-0.2, c, cs, CN), (0.2, tu, ts, TU)):
        al = 1 if y >= 1 else 0.3
        axD.bar(x + dx, y, 0.4, color=matplotlib.colors.to_rgba(col, al), lw=0)
        axD.errorbar(x + dx, y, sd, fmt='none', ecolor=(0, 0, 0, al), capsize=1.5, lw=0.8)
axD.axvline(len(scan) + 0.1, color=MUTED, lw=0.8, ls=':')
axD.axhline(1, ls='--', lw=0.8, color='grey'); axD.axhline(0, color='k', lw=0.8)
axD.set_xticks(xs + xh, labs + [l for _, l in HOP])
for i, tl in enumerate(axD.get_xticklabels()):
    if i < len(labs):
        bold = int(labs[i][1:]) in hot | FOLDX
        tl.set_fontweight('bold' if bold else 'normal'); tl.set_color(INK if bold else INK2)
        tl.set_rotation(90); tl.set_fontsize(7.4)
    else:
        tl.set_fontweight('bold'); tl.set_color(INK); tl.set_fontsize(7.4); tl.set_rotation(90)
axD.set_xlim(-0.7, xh[-1] + 0.7); axD.set_ylim(-1, 5.6); axD.set_yticks([-1, 0, 1, 2, 3, 4, 5])
axD.tick_params(axis='y', labelsize=8.5); axD.tick_params(axis='x', length=2, pad=2)
axD.set_ylabel('ΔΔG_bind (kcal/mol)', fontsize=9.5)
for s in ('top', 'right'): axD.spines[s].set_visible(False)
axD.text(len(scan) / 2 - 0.5, 5.55, 'alanine scan (per residue)', ha='center', va='bottom', fontsize=8.5, color=INK2)
axD.text((xh[0] + xh[-1]) / 2, 5.55, 'Höpfler et al.\nmutants', linespacing=0.95, ha='center', va='bottom', fontsize=8.5, color=INK2)
axD.legend(handles=[Rectangle((0, 0), 1, 1, fc=CN), Rectangle((0, 0), 1, 1, fc=TU)], labels=['CNOT11', 'β-tubulin'],
           fontsize=8.5, frameon=False, loc='upper left', bbox_to_anchor=(0.0, 1.0), handlelength=1.2)
fig.text(0.61, 0.475, 'E', fontsize=20, fontweight='bold', color=INK)
fig.text(0.635, 0.475, 'Per-residue ΔΔG_bind of human SCAPER', fontsize=13, color=INK)
fig.text(0.635, 0.448, 'CNOT11 vs β-tubulin interfaces', fontsize=10, color=INK2)
for ext in ('png', 'svg', 'pdf'):
    fig.savefig(f'{S}/fig/SCAPER_CNOT11_trypanosoma.{ext}', facecolor='white')
