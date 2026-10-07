import json
from ete3 import Tree
from pathlib import Path

CACHE_DIR = Path("cache_eggnog_filtered")
DATA_DIR = Path("data/eggnog_dataset_filtered")

tree = Tree(str(CACHE_DIR / "tree.newick"), format=1)
all_nodes = [n for n in tree.traverse() if n is not tree]
tree.add_feature("_x", 0.0)
for node in tree.traverse():
    if node.up is not None:
        node.add_feature("_x", node.up._x + node.dist)

leaves_in_order = tree.get_leaves()
for i, leaf in enumerate(leaves_in_order):
    leaf.add_feature("_y", i)

for node in tree.traverse("postorder"):
    if not node.is_leaf():
        ys = [c._y for c in node.children]
        node.add_feature("_y", sum(ys) / len(ys))

nodes_out = []
for j, node in enumerate(all_nodes):
    child_ys = [c._y for c in node.children] if not node.is_leaf() else None
    nodes_out.append({
        "idx": j,
        "x": round(node._x, 5),
        "y": round(node._y, 3),
        "parent_x": round(node.up._x, 5),
        "is_leaf": node.is_leaf(),
        "species": str(node.name) if node.is_leaf() else None,
        "clade": "EggNOG",
        "n_panel_losses": 0,
        "children_y_min": round(min(child_ys), 3) if child_ys else None,
        "children_y_max": round(max(child_ys), 3) if child_ys else None,
    })

root_child_ys = [c._y for c in tree.children]
max_x = max(n["x"] for n in nodes_out)
root_out = {"x": 0.0, "y": tree._y, "children_y_min": round(min(root_child_ys), 3), "children_y_max": round(max(root_child_ys), 3)}

with open(DATA_DIR / "tree_layout.json", "w") as fh:
    json.dump({"nodes": nodes_out, "root": root_out, "n_leaves": len(leaves_in_order), "max_x": max_x}, fh)
