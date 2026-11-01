"""Export the trained pipeline (scikit-learn preprocessing + CatBoost) into one small JSON file.

Why: CatBoost + scikit-learn + pandas are ~500 MB, too big for Vercel. The exported JSON
(model/lite_model.json) is run by backend/lite.py using only NumPy, with identical results.

Run after retraining:   python tools/export_lite_model.py
(needs the full requirements-full.txt environment; the website itself does not)
"""
import json
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
pipe = joblib.load(ROOT / "model" / "appendicitis_model.joblib")
info = json.loads((ROOT / "model" / "model_info.json").read_text(encoding="utf-8"))
prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]

# ------------------------------------------------------------------ preprocessing parameters
num_cols = list(prep.transformers_[0][2])
cat_cols = list(prep.transformers_[1][2])
num_pipe = prep.named_transformers_["num"]
imp, sc = num_pipe.named_steps["imp"], num_pipe.named_steps["sc"]
onehot = prep.named_transformers_["cat"].named_steps["oh"]
preprocessing = {
    "numeric_columns": num_cols,
    "medians": imp.statistics_.tolist(),
    "indicator_columns": [int(i) for i in imp.indicator_.features_],   # numeric columns that get a missing flag
    "scaler_mean": sc.mean_.tolist(),
    "scaler_scale": sc.scale_.tolist(),
    "categorical_columns": cat_cols,
    "categories": [[str(c) for c in cats] for cats in onehot.categories_],
    "fill_value": "Missing",
    "output_names": prep.get_feature_names_out().tolist(),
}

# ------------------------------------------------------------------ CatBoost trees
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "cb.json"
    clf.save_model(str(path), format="json")
    cb = json.loads(path.read_text())

flat_index = {f["feature_index"]: f["flat_feature_index"] for f in cb["features_info"]["float_features"]}
n_classes = len(clf.classes_)
trees = []
for t in cb["oblivious_trees"]:
    depth = len(t["splits"])
    feats = [flat_index[s["float_feature_index"]] for s in t["splits"]]
    borders = [s["border"] for s in t["splits"]]
    values = np.array(t["leaf_values"]).reshape(2 ** depth, n_classes)
    weights = np.array(t["leaf_weights"], dtype=float)
    # Path-dependent cover ratios for TreeSHAP. Split j decides bit j of the leaf index;
    # the root is the LAST split (highest bit), so a node at level k is identified by the top k bits.
    leaves = np.arange(2 ** depth)
    ratio = np.zeros((2 ** depth, depth))
    for j in range(depth):                              # split j sits at level (depth-1-j)
        child_key = leaves >> j                          # bits j..depth-1 → node after split j
        parent_key = leaves >> (j + 1)                   # bits j+1..depth-1 → node before split j
        child_w = np.array([weights[child_key == k].sum() for k in child_key])
        parent_w = np.array([weights[parent_key == k].sum() for k in parent_key])
        ratio[:, j] = np.divide(child_w, parent_w, out=np.zeros_like(child_w), where=parent_w > 0)
    trees.append({"features": feats, "borders": borders, "values": values.round(10).tolist(),
                  "cover_ratio": ratio.round(10).tolist()})

lite = {
    "format": "appendixai-lite-1",
    "classes": info["classes"],
    "preprocessing": preprocessing,
    "scale_and_bias": cb["scale_and_bias"],
    "trees": trees,
}
out = ROOT / "model" / "lite_model.json"
out.write_text(json.dumps(lite, separators=(",", ":")), encoding="utf-8")
print(f"Wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {len(trees)} trees)")
sys.exit(0)
