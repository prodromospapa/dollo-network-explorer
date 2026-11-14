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
absent = [('Caenorhabditis_elegans', 'Caenorhabditis elegans'), ('Leishmania_major', 'Leishmania major'),
          ('Trypanosoma_brucei', 'Trypanosoma brucei'), ('Trypanosoma_cruzi', 'Trypanosoma cruzi')]
contacts = {622, 624, 625, 628, 629, 632, 633, 636, 639, 640}   # <4 Å to CNOT11 in 5/5 human AF3 models
# Interface residues mutated in Hopfler et al. 2023 (Mol Cell, doi:10.1016/j.molcel.2023.05.020); every mutant abolished autoregulation
MUT = {618: 'K', 625: 'K', 628: 'S', 629: 'K', 632: 'S', 633: 'K', 640: 'K'}
hot = set(MUT)
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
        axA.add_patch(Rectangle((x, -y), 1, y, fill=False, ec='#e0a800', lw=2.2, zorder=5))
    if h == 620:
        axA.text(x + 0.5, 0.45, '†', ha='center', va='center', fontsize=13, color=INK, fontweight='bold')
    if h in contacts:
        axA.plot(x + 0.5, 0.45, 'o', ms=6, color=HUM, clip_on=False)
    if h % 5 == 0 or h in hot:
        axA.text(x + 0.5, 1.05, str(h), ha='center', va='bottom', fontsize=9, color=INK2 if h not in hot else INK,
                 fontweight='bold' if h in hot else 'normal')
for yy, lab, sp in ylab:
    axA.text(-0.4, yy, lab, ha='right', va='center', fontsize=11, style='normal' if sp == 'MUT' else 'italic',
             color=INK, fontweight='bold' if sp in ('Homo_sapiens', 'Trypanosoma_brucei') else 'normal')
# group brackets
def bracket(y0, y1, txt, col):
    xb = n + 0.5
    axA.plot([xb, xb], [y0, y1], color=col, lw=3, clip_on=False, solid_capstyle='butt')
    axA.text(xb + 0.5, (y0 + y1) / 2, txt, ha='left', va='center', fontsize=11, color=col, fontweight='bold')
bracket(-0.1, -len(present) + 0.1, 'CNOT11\npresent', '#2563eb')
bracket(-len(present) - 0.7, -len(present) - len(absent) - 0.5, 'CNOT11\nabsent', '#9a3412')
axA.set_xlim(0, n); axA.set_ylim(-y - 0.1, 1.9); axA.axis('off')
fig.text(0.015, 0.955, 'A', fontsize=20, fontweight='bold', color=INK)
fig.text(0.04, 0.955, 'SCAPER helix that contacts CNOT11 (human 612–643), TCS orthogroup alignment',
         fontsize=14, color=INK, va='baseline')


from matplotlib.patches import Rectangle as R2
fig.patches.append(R2((0.045, 0.902), 0.011, 0.02, transform=fig.transFigure, fill=False, ec='#e0a800', lw=2.2))
fig.text(0.06, 0.912, 'CNOT11-interface residues validated by mutagenesis (Höpfler et al., 2023)', fontsize=10.5, color=INK2, va='center')
fig.lines.append(matplotlib.lines.Line2D([0.43], [0.912], marker='o', ms=7, color=HUM, transform=fig.transFigure))
fig.text(0.438, 0.912, 'CNOT11 contact in 5/5 human AF3 models', fontsize=10.5, color=INK2, va='center')
fig.text(0.64, 0.912, '†', fontsize=13, color=INK, va='center', fontweight='bold')
fig.text(0.65, 0.912, 'ΔE620 patient mutation', fontsize=10.5, color=INK2, va='center')
# ---- structures (crop both renders to the same box so the scale matches)
ims = [Image.open(f'{S}/fig/{f}.png').convert('RGB') for f in ('human', 'tbrucei')]
bbs = [ImageChops.difference(im, Image.new('RGB', im.size, (255, 255, 255))).getbbox() for im in ims]
box = (min(b[0] for b in bbs) - 20, min(b[1] for b in bbs) - 20, max(b[2] for b in bbs) + 20, max(b[3] for b in bbs) + 20)
ims = [im.crop(box) for im in ims]
iptm = {}
for job in ('scaper', 'trypanosoma'):
    iptm[job] = [json.load(open(f'{FO}/{job}_region_cnot11/fold_{job}_region_cnot11_summary_confidences_{i}.json'))['iptm'] for i in range(5)]
    iptm[job + '_pae'] = [json.load(open(f'{FO}/{job}_region_cnot11/fold_{job}_region_cnot11_summary_confidences_{i}.json'))['chain_pair_pae_min'][0][1] for i in range(5)]
panels = [('B', ims[0], 'Human SCAPER 593–672', HUM, iptm['scaper'], iptm['scaper_pae'], 0.03),
          ('C', ims[1], 'T. brucei SCAPER 214–293', TRY, iptm['trypanosoma'], iptm['trypanosoma_pae'], 0.355)]
for letter, im, title, col, ip, pae, x0 in panels:
    ax = fig.add_axes([x0, 0.01, 0.31, 0.385]); ax.imshow(im); ax.axis('off')
    fig.text(x0 - 0.015, 0.475, letter, fontsize=20, fontweight='bold', color=INK)
    fig.text(x0 + 0.01, 0.475, title, fontsize=13, color=col, fontweight='bold',
             style='italic' if 'brucei' in title else 'normal')
    fig.text(x0 + 0.01, 0.448, f'+ human CNOT11 (C-terminal domain shown)', fontsize=10, color=INK2)
    fig.text(x0 + 0.01, 0.41, f'ipTM {np.mean(ip):.2f}   ·   min. interchain PAE {np.mean(pae):.1f} Å',
             fontsize=12, color=INK, fontweight='bold')

# ---- ipTM per model
axD = fig.add_axes([0.73, 0.12, 0.24, 0.31])
for k, (job, lab, col) in enumerate([('scaper', 'Human', HUM), ('trypanosoma', 'T. brucei', TRY)]):
    v = iptm[job]; xs = k + np.linspace(-0.12, 0.12, len(v))
    axD.scatter(xs, v, s=70, color=col, edgecolor='white', linewidth=2, zorder=3)
    axD.text(k, max(v) + 0.07, f'{np.mean(v):.2f}', ha='center', fontsize=12, color=INK, fontweight='bold')
axD.axhline(0.6, color=MUTED, lw=1, ls='--'); axD.text(1.45, 0.61, 'ipTM > 0.6\nlikely interface', fontsize=8.5, color=INK2, va='bottom', ha='right')
axD.set_xticks([0, 1]); axD.set_xticklabels(['Human', 'T. brucei'], fontsize=11)
axD.get_xticklabels()[1].set_style('italic')
axD.set_xlim(-0.5, 1.5); axD.set_ylim(0, 1); axD.set_ylabel('ipTM (5 AF3 models)', color=INK2)
for s in ('top', 'right'): axD.spines[s].set_visible(False)
for s in ('left', 'bottom'): axD.spines[s].set_color(MUTED)
axD.tick_params(colors=INK2); axD.grid(axis='y', color='#e2e8f0', lw=0.8); axD.set_axisbelow(True)
fig.text(0.70, 0.475, 'D', fontsize=20, fontweight='bold', color=INK)
fig.text(0.725, 0.475, 'Interface confidence', fontsize=13, color=INK)
fig.text(0.725, 0.448, 'both fragments folded with human CNOT11', fontsize=10, color=INK2)
for ext in ('png', 'svg', 'pdf'):
    fig.savefig(f'{S}/fig/SCAPER_CNOT11_trypanosoma.{ext}', facecolor='white')
