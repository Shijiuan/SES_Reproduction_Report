#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SES Innovative Runner (v6)
- DeepSeek as primary judge
- High-power parallel evaluation with rate limiting
- Multi-judge consistency (optional)
- Self-repair re-ask when parse fails
- Human-alignment calibration (optional)

Example:
python ses_v6_innovative.py \
  --data data/zero_test.json \
  --pred runs/pred_deepseek.jsonl \
  --outdir runs/v6 \
  --backend deepseek --model deepseek-chat \
  --workers 8 --rps 3 --repair_on
"""
import os, re, json, time, argparse, math
from typing import List, Dict, Any, Tuple
from collections import defaultdict

try:
    import requests
except Exception:
    requests = None
try:
    from concurrent.futures import ThreadPoolExecutor, as_completed
except Exception:
    ThreadPoolExecutor = None
try:
    from sklearn.metrics import cohen_kappa_score
    from sklearn.linear_model import LogisticRegression
    from scipy.stats import pearsonr, spearmanr
except Exception:
    cohen_kappa_score = None
    LogisticRegression = None
    pearsonr = None
    spearmanr = None

# ------------- IO -------------
def load_json(p): return json.load(open(p, "r", encoding="utf-8"))
def load_jsonl(p): return [json.loads(x) for x in open(p, "r", encoding="utf-8")]
def save_json(o, p): os.makedirs(os.path.dirname(p), exist_ok=True); json.dump(o, open(p,"w",encoding="utf-8"), indent=2, ensure_ascii=False)

# ------------- Rate Limiter -------------
class TokenBucket:
    def __init__(self, rps: float):
        self.rps = float(rps); self.tokens = float(rps); self.last = time.time()
    def acquire(self):
        if self.rps <= 0: return
        while True:
            now = time.time()
            elapsed = now - self.last
            self.tokens = min(self.rps, self.tokens + elapsed * self.rps)
            self.last = now
            if self.tokens >= 1.0: self.tokens -= 1.0; return
            time.sleep(max(0.0, (1.0 - self.tokens)/max(self.rps,1e-9)))

# ------------- Prompt & Parse -------------
STRICT_TAIL = (
    "\n\nYou must output EXACTLY five lines:\n"
    "NA: <float>\nSA: <float>\nS: <float>\nW: <float>\nSES: <float>\n"
    "No extra text before these five lines."
)
def build_prompt(d, p, strict=False):
    pred_text = p.get("output") or p.get("pred_str") or ""
    base = f"""
You are an evaluator who must give explicit numeric scores.
Output EXACTLY these five lines (one per line):
NA: (0~1)
SA: (0~1)
S: (0~1)
W: (0~1)
SES: (average of the above four floats)

Then, after the scores, briefly explain your reasoning.

[Task Instruction]
Evaluate the following output according to correctness, reasoning soundness,
completeness, and writing quality.

[Input Question]
{d.get("input","")}

[Model Output]
{pred_text}
""".strip()
    return base + (STRICT_TAIL if strict else "")

SCORE_KEYS = ["NA","SA","S","W","SES"]
PAT = {k: re.compile(rf"{k}\s*[:：]\s*([0-9]*\.?[0-9]+)") for k in SCORE_KEYS}
def parse_scores(t: str):
    out = {k: None for k in SCORE_KEYS}
    for k, rg in PAT.items():
        m = rg.search(t)
        out[k] = float(m.group(1)) if m else None
    return out
def valid(sc): return all(sc.get(k) is not None and -1e-6 <= sc[k] <= 1.0+1e-6 for k in SCORE_KEYS)

# ------------- Judges -------------
class DeepSeekJudge:
    def __init__(self, model="deepseek-chat", base_url=None, timeout=60):
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise ValueError("Set DEEPSEEK_API_KEY.")
        self.model = model
        self.base_url = (base_url or "https://api.deepseek.com").rstrip("/")
        self.timeout = timeout
    def _post(self, path, payload):
        url = f"{self.base_url}{path}"
        headers = {"Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
                   "Content-Type": "application/json"}
        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        if r.status_code == 404 and path.startswith("/chat/completions"):
            url_v1 = f"{self.base_url}/v1/chat/completions"
            r = requests.post(url_v1, headers=headers, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()
    def generate(self, prompt: str) -> str:
        payload = {"model": self.model, "messages":[{"role":"user","content":prompt}]}
        data = self._post("/chat/completions", payload)
        return data["choices"][0]["message"]["content"]

class HFJudge:
    def __init__(self, model, dtype="float32", load_in_8bit=False, load_in_4bit=False, max_new_tokens=64):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        kw = {"torch_dtype": getattr(torch, dtype)}
        if load_in_8bit: kw["load_in_8bit"] = True
        if load_in_4bit: kw["load_in_4bit"] = True
        self.model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True, **kw).eval()
        self.max_new_tokens = max_new_tokens
    def generate(self, prompt: str) -> str:
        import torch
        x = self.tok(prompt, return_tensors="pt")
        with torch.no_grad():
            y = self.model.generate(**x, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.tok.decode(y[0], skip_special_tokens=True)

def make_judge(backend, model, **kw):
    if backend == "deepseek": return DeepSeekJudge(model=model, base_url=kw.get("deepseek_base_url"))
    if backend == "hf": return HFJudge(model=model,
                                       dtype=kw.get("dtype","float32"),
                                       load_in_8bit=kw.get("load_in_8bit",False),
                                       load_in_4bit=kw.get("load_in_4bit",False),
                                       max_new_tokens=kw.get("hf_max_new_tokens",64))
    raise ValueError(backend)

# ------------- Core Eval -------------
def eval_partition(items, judge, limiter: TokenBucket, repair_on: bool):
    out = []
    for idx, d, p in items:
        try:
            limiter.acquire()
            t = judge.generate(build_prompt(d, p, strict=False))
            sc = parse_scores(t)
            if repair_on and not valid(sc):
                limiter.acquire()
                t2 = judge.generate(build_prompt(d, p, strict=True))
                sc2 = parse_scores(t2)
                if valid(sc2): sc = sc2
            if not valid(sc): sc = {k:0.0 for k in SCORE_KEYS}; sc["_invalid"]=True
            out.append((idx, sc))
        except Exception as e:
            sc = {k:0.0 for k in SCORE_KEYS}; sc["_invalid"]=True; sc["_error"]=str(e)
            out.append((idx, sc))
    return out

def average_scores(lst: List[Dict[str,float]]) -> Dict[str,Any]:
    if not lst: return {}
    res = {k: sum(d.get(k,0.0) for d in lst)/len(lst) for k in SCORE_KEYS}
    res["counts"] = {"total": len(lst)}
    res["invalid_rate"] = sum(1 for d in lst if d.get("_invalid"))/len(lst)
    return res

def run_one(name, backend, model, data, preds, workers, rps, repair_on, **kw):
    items = [(i, d, preds[i]) for i, d in enumerate(data)]
    judge = make_judge(backend, model, **kw)
    if workers <= 1 or ThreadPoolExecutor is None:
        limiter = TokenBucket(rps)
        res = eval_partition(items, judge, limiter, repair_on)
    else:
        limiter = TokenBucket(rps)
        shard = max(1, math.ceil(len(items)/workers))
        res = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(eval_partition, items[i:i+shard], judge, limiter, repair_on)
                    for i in range(0, len(items), shard)]
            for f in as_completed(futs): res.extend(f.result())
    res.sort(key=lambda x: x[0])
    scores = [s for _, s in res]
    metrics = average_scores(scores); metrics.update({"backend":backend,"model":model,"judge_name":name})
    return {"scores": scores, "metrics": metrics}

# ------------- Stats -------------
def pairwise_consistency(all_scores: Dict[str, List[Dict[str,float]]]):
    out = {"pairwise_kappa":{}, "pairwise_pearson":{}}
    names = list(all_scores.keys())
    if cohen_kappa_score is None or pearsonr is None:
        out["note"]="sklearn/scipy unavailable"; return out
    for i in range(len(names)):
        for j in range(i+1,len(names)):
            a = [all_scores[names[i]][k]["SES"] for k in range(len(all_scores[names[i]]))]
            b = [all_scores[names[j]][k]["SES"] for k in range(len(all_scores[names[j]]))]
            a_bin = [1 if v>=0.5 else 0 for v in a]; b_bin = [1 if v>=0.5 else 0 for v in b]
            out["pairwise_kappa"][f"{names[i]}|{names[j]}"] = float(cohen_kappa_score(a_bin,b_bin))
            out["pairwise_pearson"][f"{names[i]}|{names[j]}"] = float(pearsonr(a,b)[0])
    return out

def ensemble_average(all_scores: Dict[str, List[Dict[str,float]]], weights: Dict[str,float]=None):
    names = list(all_scores.keys()); 
    if not weights: weights = {n:1.0 for n in names}
    ens = []
    L = len(all_scores[names[0]])
    for i in range(L):
        mix = {}
        for k in SCORE_KEYS:
            num=0.0; den=0.0
            for n in names:
                num += weights.get(n,1.0)*all_scores[n][i].get(k,0.0); den += weights.get(n,1.0)
            mix[k]=num/max(den,1e-9)
        ens.append(mix)
    return ens

def load_human_labels(csv_path: str):
    m={}
    for line in open(csv_path,"r",encoding="utf-8"):
        line=line.strip()
        if not line or line.lower().startswith("id"): continue
        i,lab=line.split(",")[:2]; m[int(i)]=int(lab)
    return m

def calibrate(scores: List[Dict[str,float]], labels: Dict[int,int]):
    if LogisticRegression is None or pearsonr is None or spearmanr is None:
        return {"note":"sklearn/scipy unavailable"}
    xs=[]; ys=[]
    for i, sc in enumerate(scores):
        if i in labels:
            xs.append([sc["SES"]]); ys.append(labels[i])
    if not xs: return {"note":"no overlap"}
    lr = LogisticRegression(max_iter=1000).fit(xs, ys)
    probs = lr.predict_proba(xs)[:,1]
    r1 = pearsonr([x[0] for x in xs], ys)[0]
    r2 = pearsonr(probs, ys)[0]
    s1 = spearmanr([x[0] for x in xs], ys)[0]
    s2 = spearmanr(probs, ys)[0]
    return {"train_size": len(xs), "pearson_before": float(r1), "pearson_after": float(r2),
            "spearman_before": float(s1), "spearman_after": float(s2),
            "coef": lr.coef_.tolist(), "intercept": lr.intercept_.tolist()}

# ------------- CLI -------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--backend", default="deepseek", choices=["deepseek","hf"])
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--rps", type=float, default=3.0)
    ap.add_argument("--repair_on", action="store_true")
    ap.add_argument("--dtype", default="float32", choices=["float16","bfloat16","float32"])
    ap.add_argument("--load_in_8bit", action="store_true")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--hf_max_new_tokens", type=int, default=64)
    ap.add_argument("--deepseek_base_url", default=None)
    ap.add_argument("--extra_models", default="", help="comma list like 'hf:meta-llama/Llama-2-7b-chat-hf,deepseek:deepseek-chat'")
    ap.add_argument("--weights", default="", help="comma list aligned with [primary + extras]")
    ap.add_argument("--human_ref", default="")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, args.workers//2)))
    os.environ.setdefault("MKL_NUM_THREADS", str(max(1, args.workers//2)))

    data = load_json(args.data)
    preds = load_jsonl(args.pred)
    assert len(data)==len(preds), "data/pred length mismatch"

    reports={}
    primary = run_one("primary", args.backend, args.model, data, preds,
                      args.workers, args.rps, args.repair_on,
                      dtype=args.dtype, load_in_8bit=args.load_in_8bit,
                      load_in_4bit=args.load_in_4bit, hf_max_new_tokens=args.hf_max_new_tokens,
                      deepseek_base_url=args.deepseek_base_url)
    save_json(primary["metrics"], os.path.join(args.outdir,"metrics_primary.json"))
    reports["primary"] = primary["scores"]

    names=["primary"]
    if args.extra_models.strip():
        for i, spec in enumerate([s.strip() for s in args.extra_models.split(",") if s.strip()]):
            if ":" in spec: bkd, mdl = spec.split(":",1)
            else: bkd, mdl = "deepseek", spec
            tag=f"extra{i+1}"
            r = run_one(tag, bkd, mdl, data, preds, max(1,args.workers//2), max(1e-6,args.rps/2.0),
                        args.repair_on, dtype=args.dtype, load_in_8bit=args.load_in_8bit,
                        load_in_4bit=args.load_in_4bit, hf_max_new_tokens=args.hf_max_new_tokens,
                        deepseek_base_url=args.deepseek_base_url)
            save_json(r["metrics"], os.path.join(args.outdir, f"metrics_{tag}.json"))
            reports[tag]=r["scores"]; names.append(tag)

    cons = pairwise_consistency(reports)
    save_json(cons, os.path.join(args.outdir,"consistency_report.json"))

    # ensemble
    w=None
    if args.weights.strip():
        ws=[float(x) for x in args.weights.split(",")]
        if len(ws)==len(names):
            w={n:v for n,v in zip(names, ws)}
    ens = ensemble_average(reports, w)
    ens_metrics = average_scores(ens); ens_metrics.update({"backend":"ensemble","models":names,"weights":w})
    save_json(ens_metrics, os.path.join(args.outdir,"metrics_ensemble.json"))

    # human calibration (on ensemble)
    if args.human_ref:
        labels = load_human_labels(args.human_ref)
        cal = calibrate(ens, labels)
        save_json(cal, os.path.join(args.outdir,"human_calibration.json"))

    print("Done.")

if __name__ == "__main__":
    main()