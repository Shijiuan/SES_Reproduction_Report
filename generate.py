# -*- coding: utf-8 -*-
"""
generate_v2.py  ——  生成 Argument′（pred.jsonl）
- 后端：hf / deepseek / dashscope
- 输入：--data  (json，含每条样本的 prompt/input)
- 输出：--out   (jsonl，形如 {"id_string": "...", "pred_str": "..."} )

依赖：
  pip install transformers accelerate sentencepiece tqdm requests
  # 若用 dashscope： pip install dashscope
"""

import os, json, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# -------------------- 后端封装 --------------------
class HFWrapper:
    def __init__(self, model, dtype="float32", max_new_tokens=128):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.max_new_tokens = int(max_new_tokens)

        # dtype
        d = (dtype or "float32").lower()
        if d in ("float16", "fp16"):
            torch_dtype = torch.float16
        elif d in ("bfloat16", "bf16"):
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        self.tok = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
        kwargs = dict(device_map="auto", trust_remote_code=True)
        # 这里不默认 4/8bit，尽量简化；如需可在此加 load_in_4bit/8bit
        kwargs.update(torch_dtype=torch_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)

        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        # 设备
        try:
            self.device = next(self.model.parameters()).device
        except Exception:
            self.device = "cpu"

    def chat(self, text: str) -> str:
        inputs = self.tok(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self.tok.eos_token_id,
                pad_token_id=self.tok.pad_token_id,
            )
        txt = self.tok.decode(out[0], skip_special_tokens=True)
        return txt.strip()


class DeepSeekWrapper:
    def __init__(self, model: str = "deepseek-chat", base_url: str = "https://api.deepseek.com", api_key: str | None = None, timeout: float = 45.0, max_retries: int = 6):
        import requests
        self.requests = requests
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set.")
        self.timeout = timeout
        self.max_retries = max_retries

    def chat(self, text: str) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "messages": [{"role": "user", "content": text}], "max_tokens": 256}
        to = self.timeout
        for attempt in range(self.max_retries):
            try:
                r = self.requests.post(url, headers=headers, json=payload, timeout=to)
                if r.status_code == 200:
                    data = r.json()
                    msg = (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "")
                    return (msg or "").strip()
                if r.status_code in (429, 500, 502, 503, 504):
                    import time, random
                    time.sleep(min(2 ** attempt, 18) + random.random() * 0.3)
                    continue
                break
            except Exception:
                import time, random
                time.sleep(min(2 ** attempt, 18) + random.random() * 0.3)
                continue
        return ""


class DashScopeWrapper:
    """
    通义千问（DashScope）后端
    需要环境变量：DASHSCOPE_API_KEY
    可选模型：qwen-plus / qwen-max / qwen-turbo ...
    """
    def __init__(self, model: str = "qwen-plus", api_key: str | None = None):
        import dashscope, os
        dashscope.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        if not dashscope.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set.")
        from dashscope import Generation
        self.Generation = Generation
        self.model = model

    def chat(self, text: str) -> str:
        rsp = self.Generation.call(
            model=self.model,
            input={"messages": [{"role": "user", "content": text}]},
            parameters={"result_format": "message", "max_tokens": 256},
        )
        try:
            return rsp.output["choices"][0]["message"]["content"].strip()
        except Exception:
            # 失败时尽量返回错误信息，避免丢样本
            return f"[DashScopeError] code={getattr(rsp, 'status_code', '?')} msg={getattr(rsp, 'message', '')}".strip()


# -------------------- 数据/提示处理 --------------------
def load_list(data_path: str):
    with open(data_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # 兼容两种格式：纯数组 或 {"data": [...]}
    items = obj["data"] if isinstance(obj, dict) and "data" in obj else obj
    return items

def extract_prompt(item: dict) -> tuple[str, str]:
    """
    返回 (id_string, prompt_text)
    优先使用 item["input"]；否则 fallback 到 "prompt" / "question"
    """
    sid = str(item.get("id_string") or item.get("id") or item.get("qid") or item.get("sample_id") or "")
    text = item.get("input") or item.get("prompt") or item.get("question") or ""
    return sid, text


# -------------------- 主流程 --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--backend", required=True, choices=["hf", "deepseek", "dashscope"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--hf_max_new_tokens", type=int, default=128)
    ap.add_argument("--max_workers", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    items = load_list(args.data)
    print(f"[INFO] loaded {len(items)} samples from {args.data}")

    # init backend
    if args.backend == "hf":
        client = HFWrapper(args.model, args.dtype, args.hf_max_new_tokens)
    elif args.backend == "deepseek":
        client = DeepSeekWrapper(args.model)
    else:
        client = DashScopeWrapper(args.model)

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {}
        for it in items:
            sid, prompt = extract_prompt(it)
            if not sid or not prompt:
                continue
            futures[ex.submit(client.chat, prompt)] = sid

        for fut in tqdm(as_completed(futures), total=len(futures), desc="generating"):
            sid = futures[fut]
            try:
                txt = fut.result() or ""
            except Exception as e:
                txt = f"[Error] {type(e).__name__}: {e}"
            results.append({"id_string": sid, "pred_str": txt})

    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] saved {len(results)} lines to {args.out}")


if __name__ == "__main__":
    main()
