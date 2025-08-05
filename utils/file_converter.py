from xgboost import to_graphviz
import numpy as np
import os


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def convert_tree_to_pdf(model):
    dot_data = to_graphviz(model, tree_idx=199).source

    lines = dot_data.splitlines()
    new_lines = []
    for line in lines:
        if "label=" in line and "leaf=" in line:
            raw_val = float(line.split("leaf=")[-1].split('"')[0])
            prob = sigmoid(raw_val)
            label = "new" if prob >= 0.5 else "used"
            line = line.replace(f"leaf={raw_val}", f"leaf={label}")
        new_lines.append(line)

    with open("docs/Tree_rules.dot", "w") as f:
        f.write("\n".join(new_lines))

    os.system("dot -Tpdf docs/Tree_rules.dot -o docs/Tree_rules.pdf")
    os.remove("docs/Tree_rules.dot")
