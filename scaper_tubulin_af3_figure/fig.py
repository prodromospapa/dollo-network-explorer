import json, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
D=sys.argv[1]; P=f"{D}/fold_scaper_n_terminal_and_one_tubulin_q9by12"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,
  "axes.edgecolor":"#52514e","axes.labelcolor":"#0b0b0b","xtick.color":"#52514e","ytick.color":"#52514e"})
C5=["#2a78d6","#eb6834","#1baf7a","#eda100","#6250d6"]
S=[json.load(open(f"{P}_summary_confidences_{i}.json")) for i in range(5)]
F=[json.load(open(f"{P}_full_data_{i}.json")) for i in range(5)]
ch=np.array(F[0]['token_chain_ids']); nA=(ch=='A').sum(); N=len(ch)
# per-residue pLDDT for Scaper (token-level: mean of atoms per residue)
def res_plddt(fd):
    import collections
    # atom-level plddt grouped via CA bfactor from cif instead
    return None
from Bio.PDB import MMCIFParser
par=MMCIFParser(QUIET=True)
pl=[np.array([r['CA'].bfactor for r in par.get_structure("x",f"{P}_model_{i}.cif")[0]['A']]) for i in range(5)]
cent=[0.0,13.3,25.9,34.6,35.9]

fig=plt.figure(figsize=(11,8.2),dpi=300)
gs=fig.add_gridspec(2,3,height_ratios=[1.15,1],width_ratios=[1.15,1.15,1],hspace=0.45,wspace=0.30,
                    left=0.05,right=0.95,top=0.88,bottom=0.08)
fig.suptitle("AlphaFold 3 cannot place the SCAPER N-terminus on β-tubulin",fontsize=13,fontweight="bold",x=0.04,ha="left")
fig.text(0.04,0.925,"SCAPER N-terminal fragment (94 aa, chain A) + β-tubulin (chain B, 445 aa) · AlphaFold Server, 5 samples · seed 1531038233",
         fontsize=9,color="#52514e",ha="left")
def lab(*a,**k): pass
def _unused(ax,t,x=-0.02,y=1.02): ax.text(x,y,t,transform=ax.transAxes,fontsize=13,fontweight="bold",va="bottom",ha="right" if x<0 else "left")

# A overlay
ax=fig.add_subplot(gs[0,0]); ax.imshow(plt.imread("overlay.png")); ax.axis("off"); lab(ax,"A",0.0,1.0)
ax.set_title(r"$\bf{A}$   5 samples superposed on tubulin:"+"\n"+"SCAPER lands somewhere different each time",fontsize=9.5,loc="left")
ax.legend(handles=[Line2D([0],[0],color=c,lw=3,label=f"model {i}  (Δcentroid {cent[i]:.0f} Å)") for i,c in enumerate(C5)]
          +[Patch(fc="#e5e5e5",ec="#999",label="β-tubulin (surface)")],
          loc="upper left",fontsize=7,frameon=False,ncol=2,bbox_to_anchor=(0,0.04))
# B pLDDT render
ax=fig.add_subplot(gs[0,1]); ax.imshow(plt.imread("plddt.png")); ax.axis("off"); lab(ax,"B",0.0,1.0)
ax.set_title(r"$\bf{B}$   Top-ranked model coloured by pLDDT:"+"\n"+"helical hairpin folds, but contact is not supported",fontsize=9.5,loc="left")
ax.legend(handles=[Patch(fc=c,label=l) for c,l in zip(["#0d57d3","#6acbf1","#fed936","#fd7d4d"],
          ["very high (>90)","confident (70–90)","low (50–70)","very low (<50)"])],
          loc="upper left",fontsize=7,frameon=False,ncol=2,title="pLDDT (SCAPER)",title_fontsize=7.5,bbox_to_anchor=(0,0.04))
# C PAE
ax=fig.add_subplot(gs[0,2]); pae=np.array(F[0]['pae'])
im=ax.imshow(pae,cmap="Greens_r",vmin=0,vmax=31.75,interpolation="nearest")
for v in [nA-0.5]: ax.axhline(v,color="#0b0b0b",lw=0.8); ax.axvline(v,color="#0b0b0b",lw=0.8)
ax.set_xticks([nA/2,nA+(N-nA)/2]); ax.set_xticklabels(["SCAPER","β-tubulin"]); ax.set_yticks([nA/2,nA+(N-nA)/2]); ax.set_yticklabels(["SCAPER","β-tubulin"],rotation=90,va="center")
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
ab=pae[:nA,nA:].mean()
ax.text(nA+(N-nA)/2,nA/2,f"inter-chain\nmean PAE\n{ab:.0f} Å",ha="center",va="center",fontsize=8,color="#0b0b0b")
cb=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.03); cb.set_label("Expected position error (Å)"); cb.outline.set_visible(False)
ax.set_title(r"$\bf{C}$   PAE (model 0): relative placement"+"\n"+"of the two chains is unknown",fontsize=9.5,loc="left"); lab(ax,"C",-0.05)

# D confidence metrics
ax=fig.add_subplot(gs[1,0:2]); x=np.arange(5); w=0.26
ip=[s['iptm'] for s in S]; pa=[s['chain_ptm'][0] for s in S]; pb=[s['chain_ptm'][1] for s in S]
for off,vals,c,l in [(-w,ip,"#e34948","ipTM (interface)"),(0,pa,"#2a78d6","pTM SCAPER"),(w,pb,"#8a8987","pTM β-tubulin")]:
    b=ax.bar(x+off,vals,w*0.92,color=c,label=l)
    for xi,v in zip(x+off,vals): ax.text(xi,v+0.015,f"{v:.2f}",ha="center",fontsize=7,color="#52514e")
ax.axhline(0.8,color="#52514e",ls="--",lw=0.8); ax.axhline(0.6,color="#52514e",ls=":",lw=0.8)
ax.text(4.55,0.815,"ipTM ≥ 0.8\nconfident complex",fontsize=7,ha="left",va="bottom",color="#52514e")
ax.text(4.55,0.615,"ipTM < 0.6\nlikely failed",fontsize=7,ha="left",va="bottom",color="#52514e")
ax.set_xticks(x); ax.set_xticklabels([f"model {i}\nrank score {s['ranking_score']:.2f}" for i,s in enumerate(S)])
ax.set_ylim(0,1.05); ax.set_ylabel("score"); ax.set_xlim(-0.5,5.25)
ax.legend(frameon=False,ncol=3,loc="upper left",bbox_to_anchor=(0,1.12),fontsize=8)
ax.set_title(r"$\bf{D}$   Both chains are individually predicted, the interface is not (ipTM 0.12–0.15 in every sample)",fontsize=9.5,loc="left",pad=22)
lab(ax,"D",-0.03,1.08)

# E per-residue pLDDT + contact prob (stacked, one axis each)
sg=gs[1,2].subgridspec(2,1,height_ratios=[3,1],hspace=0.12)
ax=fig.add_subplot(sg[0]); r=np.arange(1,nA+1)
ax.axhspan(0,50,color="#fd7d4d",alpha=0.08,lw=0); ax.axhspan(50,70,color="#fed936",alpha=0.10,lw=0)
ax.axhspan(70,90,color="#6acbf1",alpha=0.10,lw=0)
for i,p in enumerate(pl): ax.plot(r,p,color=C5[i],lw=1.2,alpha=0.9)
ax.set_ylim(0,100); ax.set_xlim(1,nA); ax.set_ylabel("pLDDT"); ax.tick_params(labelbottom=False)
ax.set_title(r"$\bf{E}$   SCAPER per-residue confidence"+"\n(5 models; colours as in A)",fontsize=9.5,loc="left")
cp=np.array([np.array(f['contact_probs'])[:nA,nA:].max(1) for f in F]).max(0)
ax2=fig.add_subplot(sg[1],sharex=ax)
ax2.bar(r,cp*100,width=0.9,color="#52514e")
ax2.set_ylim(0,100); ax2.set_yticks([0,50,100]); ax2.set_ylabel("P(contact)\n%",fontsize=8)
ax2.set_xlabel("SCAPER fragment residue")
ax2.text(nA-1,92,f"max contact probability with tubulin: {cp.max()*100:.0f}%",ha="right",va="top",fontsize=7,color="#52514e")
fig.savefig("scaper_tubulin_af3.png",dpi=300); fig.savefig("scaper_tubulin_af3.pdf")
