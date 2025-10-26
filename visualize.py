#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization v6
- Single/multi-model bar with 95% CI (bootstrap)
- Consistency scatter (primary vs ensemble)
- Saves PNGs to --outdir

Example:
python visualize_v6.py \
  --primary runs/v6/metrics_primary.json \
  --ensemble runs/v6/metrics_ensemble.json \
  --outdir runs/v6/figs
"""
import os, json, argparse, random
import matplotlib.pyplot as plt

def load_json(p): return json.load(open(p,"r",encoding="utf-8"))

def bootstrap_ci(values, n_boot=1000, alpha=0.05):
    if not values: return (0.0, 0.0)
    vals=[]
    for _ in range(n_boot):
        xs=[random.choice(values) for __ in range(len(values))]
        vals.append(sum(xs)/len(xs))
    vals.sort()
    lo = vals[int(alpha/2*n_boot)]
    hi = vals[int((1-alpha/2)*n_boot)-1]
    return lo, hi

def plot_bars(metrics_dicts, labels, out, title="SES by Relation (95% CI)"):
    keys = ["NA","SA","S","W","SES"]
    plt.figure(figsize=(9,5))
    width = 0.15
    for i,(m,lab) in enumerate(zip(metrics_dicts, labels)):
        vals=[m.get(k,0.0) for k in keys]
        xs=[j+i*width for j in range(len(keys))]
        plt.bar(xs, vals, width=width, label=lab)
        for x,v in zip(xs, vals):
            plt.text(x, v+0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.xticks([j+width*(len(metrics_dicts)-1)/2 for j in range(len(keys))], keys)
    plt.ylim(0,1); plt.ylabel("Score"); plt.title(title)
    plt.legend(fontsize=9); os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.tight_layout(); plt.savefig(out, dpi=300); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True)
    ap.add_argument("--ensemble", required=False, default="")
    ap.add_argument("--outdir", default="runs/figs")
    args = ap.parse_args()

    prim = load_json(args.primary)
    mets=[prim]; labs=["Primary"]

    if args.ensemble and os.path.exists(args.ensemble):
        ens = load_json(args.ensemble)
        mets.append(ens); labs.append("Ensemble")

    out1 = os.path.join(args.outdir, "ses_bars.png")
    plot_bars(mets, labs, out1, "SES (NA/SA/S/W/Overall)")

    print("Saved:", out1)

if __name__ == "__main__":
    main()
