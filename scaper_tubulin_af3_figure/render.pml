for i in range(5): cmd.load(f"m{i}.pdb", f"m{i}")
hide everything
bg_color white
set ray_opaque_background, 0
set cartoon_transparency, 0
set ray_shadows, 0
set antialias, 2
set cartoon_fancy_helices, 1
# panel 1: overlay
create tub, m0 and chain B
show surface, tub
color grey90, tub
set surface_quality, 1
set transparency, 0.0, tub
python
cols=["0x2a78d6","0xeb6834","0x1baf7a","0xeda100","0x6250d6"]
for i,c in enumerate(cols):
    cmd.show("cartoon", f"m{i} and chain A")
    cmd.color(c, f"m{i} and chain A")
python end
orient tub or (m* and chain A)
turn x, -90
python
# choose view showing spread: orient on union of scaper copies
python end
orient (m* and chain A) or tub
png overlay.png, width=1600, height=1400, dpi=300, ray=1
# panel 2: model 0 colored by pLDDT
hide everything
show surface, tub
set transparency, 0.55, tub
show cartoon, m0 and chain A
set_color vh, [0.051,0.341,0.827]
set_color hi, [0.416,0.796,0.945]
set_color lo, [0.996,0.851,0.212]
set_color vl, [0.992,0.490,0.302]
color vl, m0 and chain A
color lo, m0 and chain A and b>50
color hi, m0 and chain A and b>70
color vh, m0 and chain A and b>90
color grey80, tub
png plddt.png, width=1600, height=1400, dpi=300, ray=1
