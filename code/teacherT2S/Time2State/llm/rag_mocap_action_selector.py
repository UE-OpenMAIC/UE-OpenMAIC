                       

   

from __future__ import annotations

import argparse

import json

import math

import os

import re

from collections import Counter, defaultdict

from pathlib import Path

from typing import Any

import pandas as pd

RAG_DIR_DEFAULT = Path(r"D:\code\teacherT2S\Time2State\llm\mocap_action_rag")

OUT_DIR_DEFAULT = Path(r"D:\code\teacherT2S\Time2State\llm\rag_mocap_action_selector")

def ensure_dir(p: Path) -> None:

    p.mkdir(parents=True, exist_ok=True)

def clean(x: Any) -> str:

    if x is None:

        return ""

    if isinstance(x, float) and math.isnan(x):

        return ""

    return " ".join(str(x).replace("\r", " ").replace("\n", " ").replace("\t", " ").split()).strip()

def safe_float(x: Any, default: float = 0.0) -> float:

    try:

        if pd.isna(x):

            return float(default)

        value = float(x)

        return value if math.isfinite(value) else float(default)

    except Exception:

        return float(default)

def safe_int(x: Any, default: int = 0) -> int:

    try:

        if pd.isna(x):

            return int(default)

        return int(float(x))

    except Exception:

        return int(default)

def read_jsonl(path: Path) -> list[dict]:

    items = []

    with path.open("r", encoding="utf-8") as f:

        for line in f:

            if line.strip():

                items.append(json.loads(line))

    return items

def write_jsonl(items: list[dict], path: Path) -> None:

    with path.open("w", encoding="utf-8") as f:

        for item in items:

            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def tokenize(text: str) -> list[str]:

    text = clean(text).lower()

    words = re.findall(r"[a-zA-Z0-9_]+", text)

    chars = [c for c in text if "\u4e00" <= c <= "\u9fff"]

    bigrams = [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]

    return words + chars + bigrams

class BM25:

    def __init__(self, docs: list[dict]):

        self.docs = docs

        self.doc_tokens = [tokenize(d.get("title", "") + " " + d.get("content", "")) for d in docs]

        self.N = len(docs)

        self.avgdl = sum(len(t) for t in self.doc_tokens) / max(1, self.N)

        self.df = Counter()

        for toks in self.doc_tokens:

            self.df.update(set(toks))

    def score(self, query: str, top_k: int = 12, doc_types: set[str] | None = None) -> list[dict]:

        q = tokenize(query)

        if not q:

            return []

        q_count = Counter(q)

        out = []

        k1, b = 1.5, 0.75

        for i, d in enumerate(self.docs):

            if doc_types and d.get("doc_type") not in doc_types:

                continue

            toks = self.doc_tokens[i]

            tf = Counter(toks)

            dl = len(toks)

            score = 0.0

            for term in q_count.keys():

                if term not in tf:

                    continue

                idf = math.log(1 + (self.N - self.df[term] + 0.5) / (self.df[term] + 0.5))

                denom = tf[term] + k1 * (1 - b + b * dl / max(1e-6, self.avgdl))

                score += idf * tf[term] * (k1 + 1) / denom

            if score > 0:

                out.append((score, d))

        out.sort(key=lambda x: x[0], reverse=True)

        return [{"rank": j + 1, "score": float(s), **d} for j, (s, d) in enumerate(out[:top_k])]

def split_sentences(text: str) -> list[str]:

    text = clean(text)

                                  

    parts = re.split(r"(?<=[。！？!?；;\.])\s*", text)

    out = []

    for p in parts:

        p = p.strip()

        if p:

            out.append(p)

    if len(out) <= 1:

        out = [p.strip() for p in re.split(r"[\n]+", text) if p.strip()]

    return out

def load_input_text(args) -> str:

    if args.text_file:

        return Path(args.text_file).read_text(encoding="utf-8", errors="ignore")

    return args.text or args.q

def load_action_dictionary(rag_dir: Path) -> dict[str, dict]:

    path = rag_dir / "mocap_action_dictionary.csv"

    if not path.exists():

        return {}

    df = pd.read_csv(path, dtype=str).fillna("")

    out = {}

    for _, r in df.iterrows():

        mc = clean(r.get("动捕编号", ""))

        if mc.startswith("MC"):

            out[mc] = {

                "mocap_id": mc,

                "mocap_action": clean(r.get("动捕动作名称", "")),

                "mocap_coarse": clean(r.get("所属粗类", "")),

                "description": clean(r.get("说明", "")),

                "representatives": clean(r.get("代表local与专家描述", "")),

            }

    return out

def heuristic_boost(sentence: str, mc_id: str, action_name: str, coarse: str, previous_mocap_id: str = "") -> float:

    s = clean(sentence).lower()

    name = f"{action_name} {coarse}"

    boost = 0.0

    board_words = ["板书", "黑板", "写", "书写", "公式", "推导", "关键词", "画", "标注"]

    screen_words = ["屏幕", "ppt", "课件", "看这里", "请看", "看一下", "标题", "这个词", "这个字", "图", "表"]

    explain_words = ["讲", "解释", "说明", "理解", "为什么", "思考", "想一想", "问题", "回答"]

    summary_words = ["总结", "最后", "回顾", "因此", "所以", "归纳", "反思", "升华"]

    if any(w in s for w in board_words):

        if mc_id in {"MC12", "MC13", "MC14", "MC15"} or "板书" in name:

            boost += 3.0

        if mc_id in {"MC01", "MC04", "MC05", "MC06"}:

            boost -= 0.6

    if any(w in s for w in screen_words):

        if mc_id in {"MC02", "MC07", "MC08", "MC10"} or "指向" in name or "PPT" in name or "屏幕" in name:

            boost += 2.6

    if any(w in s for w in explain_words):

        if mc_id in {"MC01", "MC03", "MC04", "MC05", "MC07", "MC09"} or "讲解" in name:

            boost += 1.4

    if any(w in s for w in summary_words):

        if mc_id in {"MC01", "MC03", "MC04"} or "讲解" in name:

            boost += 1.2

                             

    if previous_mocap_id and len(s) <= 14 and mc_id == previous_mocap_id:

        boost += 1.0

    return boost

def retrieve_context(bm25: BM25, sentence: str, lesson_context: str, top_k: int) -> list[dict]:

    query = sentence

    if lesson_context:

        query = sentence + " " + clean(lesson_context)[:800]

    return bm25.score(

        query,

        top_k=top_k,

        doc_types={"sentence_mocap", "mocap_summary", "mocap_transition", "mocap_dictionary"},

    )

def score_mocap_candidates(sentence: str, contexts: list[dict], action_dict: dict[str, dict], previous_mocap_id: str = "") -> list[dict]:

    scores = defaultdict(float)

    evidence = defaultdict(list)

    doc_type_weight = {

        "sentence_mocap": 1.30,

        "mocap_summary": 0.95,

        "mocap_dictionary": 0.55,

        "mocap_transition": 0.35,

    }

    for c in contexts:

        meta = c.get("metadata", {}) or {}

        mc = clean(meta.get("mocap_id", ""))

        if not mc.startswith("MC"):

                                                             

            continue

        w = doc_type_weight.get(clean(c.get("doc_type")), 0.5)

        scores[mc] += safe_float(c.get("score")) * w

        if len(evidence[mc]) < 4:

            evidence[mc].append({

                "doc_type": clean(c.get("doc_type")),

                "title": clean(c.get("title")),

                "score": safe_float(c.get("score")),

                "content_excerpt": clean(c.get("content", ""))[:360],

            })

                               

    for mc in action_dict.keys():

        scores.setdefault(mc, 0.001)

    for mc in list(scores.keys()):

        d = action_dict.get(mc, {})

        scores[mc] += heuristic_boost(sentence, mc, clean(d.get("mocap_action", "")), clean(d.get("mocap_coarse", "")), previous_mocap_id)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    out = []

    for mc, score in ranked[:8]:

        d = action_dict.get(mc, {})

        out.append({

            "mocap_id": mc,

            "mocap_action": clean(d.get("mocap_action", "")),

            "mocap_coarse": clean(d.get("mocap_coarse", "")),

            "score": float(score),

            "description": clean(d.get("description", "")),

            "evidence": evidence.get(mc, []),

        })

    return out

def deterministic_plan(sentences: list[str], bm25: BM25, action_dict: dict[str, dict], lesson_context: str, top_k: int) -> tuple[list[dict], list[dict]]:

    plan = []

    retrieval_rows = []

    previous = ""

    for i, sent in enumerate(sentences, start=1):

        contexts = retrieve_context(bm25, sent, lesson_context, top_k=top_k)

        candidates = score_mocap_candidates(sent, contexts, action_dict, previous_mocap_id=previous)

        best = candidates[0] if candidates else {}

        item = {

            "sentence_id": i,

            "sentence": sent,

            "recommended_mocap_id": clean(best.get("mocap_id", "")),

            "recommended_mocap_action": clean(best.get("mocap_action", "")),

            "recommended_mocap_coarse": clean(best.get("mocap_coarse", "")),

            "confidence_score": safe_float(best.get("score")),

            "why": "本地 BM25 检索真实教师文本-动捕样例，并结合板书/指屏/讲解/总结等关键词先验加权。",

            "top_candidates": candidates[:5],

        }

        plan.append(item)

        retrieval_rows.append({"sentence_id": i, "sentence": sent, "contexts": contexts})

        previous = item["recommended_mocap_id"] or previous

    return plan, retrieval_rows

def build_prompt_for_chat(sentences: list[str], deterministic: list[dict], retrieval_rows: list[dict], lesson_context: str) -> str:

    blocks = []

    for row, det in zip(retrieval_rows, deterministic):

        ctx_text = "\n".join([

            f"【{c.get('rank')}｜{c.get('doc_type')}｜score={safe_float(c.get('score')):.3f}】{clean(c.get('title'))}\n{clean(c.get('content'))[:1200]}"

            for c in row["contexts"][:8]

        ])

        cand_text = json.dumps(det.get("top_candidates", [])[:5], ensure_ascii=False, indent=2)

        blocks.append(

            f"### 句子 {row['sentence_id']}\n"

            f"{row['sentence']}\n\n"

            f"本地候选：\n{cand_text}\n\n"

            f"检索上下文：\n{ctx_text}"

        )

    joined = "\n\n".join(blocks)

    return f"""你是数字人教师动捕动作选择器。你必须从 MC01-MC15 中为每个句子选择一个可驱动的动捕动作。

整段授课文本上下文：
{clean(lesson_context)[:2500]}

候选句子、检索到的真实教师文本-动捕样例、本地候选如下：
{joined}

请输出严格 JSON 数组。每个元素格式如下：
{ 
  "sentence_id": 1,
  "sentence": "原句",
  "recommended_mocap_id": "MCxx",
  "recommended_mocap_action": "动捕动作名称",
  "motion_intent": "为什么这个文本需要这个动作",
  "duration_policy": "保持上一动作/切换动作/作为过渡，建议持续多久",
  "evidence": "引用检索到的真实教师样例或动作字典",
  "risk": "可能不自然的风险；没有则写无"
} 

硬约束：
1. 只能输出 MC01-MC15，不能输出 G/local，也不能输出 NO_MOCAP。
2. 不要选择识别失败、遮挡失败、0不适合作为动捕的动作。
3. 短句优先保持上一动捕动作；涉及 PPT/屏幕/题目/图表时优先指向或 PPT 操作；涉及板书/公式/关键词书写时优先板书动作。
4. 输出必须是 JSON，不要加 Markdown。"""

def deepseek_chat(prompt: str, model: str) -> str:

    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not api_key:

        raise RuntimeError("缺少 OPENAI_API_KEY；不用 LLM 时请不要加 --use-chat。")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com").strip()

    client = OpenAI(api_key=api_key, base_url=base_url)

    resp = client.chat.completions.create(

        model=model,

        messages=[

            {"role": "system", "content": "你是严谨的数字人教师动捕动作选择专家。"},

            {"role": "user", "content": prompt},

        ],

        temperature=0.15,

    )

    return resp.choices[0].message.content

def parse_args():

    ap = argparse.ArgumentParser()

    ap.add_argument("mode", nargs="?", choices=["select", "plan", "show-actions"], default="plan")

    ap.add_argument("--rag-dir", type=str, default=str(RAG_DIR_DEFAULT))

    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR_DEFAULT))

    ap.add_argument("--text", type=str, default="")

    ap.add_argument("--q", type=str, default="")

    ap.add_argument("--text-file", type=str, default="")

    ap.add_argument("--lesson-text", type=str, default="", help="可选，整段课稿上下文；不填时使用 --text/--text-file")

    ap.add_argument("--lesson-file", type=str, default="", help="可选，整段课稿文件；不填时使用 --text-file")

    ap.add_argument("--sentence-only", action="store_true", help="只用当前句子检索，不拼接整段课稿上下文；适合 demo 展示动作差异。")

    ap.add_argument("--top-k", type=int, default=12)

    ap.add_argument("--use-chat", action="store_true", help="调用 DeepSeek/OpenAI 兼容接口做最终裁决")

    ap.add_argument("--chat-model", type=str, default=os.getenv("CHAT_MODEL", "deepseek-chat"))

    return ap.parse_args()

def main():

    args = parse_args()

    rag_dir = Path(args.rag_dir)

    out_dir = Path(args.out_dir)

    ensure_dir(out_dir)

    docs_path = rag_dir / "mocap_action_rag_docs.jsonl"

    if not docs_path.exists():

        raise FileNotFoundError(f"找不到 RAG 文档：{docs_path}。请先运行 build_mocap_action_rag_from_marked_video.py")

    action_dict = load_action_dictionary(rag_dir)

    if not action_dict:

        raise RuntimeError(f"动作字典为空：{rag_dir / 'mocap_action_dictionary.csv'}")

    if args.mode == "show-actions":

        print(json.dumps(list(action_dict.values()), ensure_ascii=False, indent=2))

        return

    docs = read_jsonl(docs_path)

    bm25 = BM25(docs)

    text = load_input_text(args)

    if args.sentence_only:

        lesson_context = ""

    elif args.lesson_file:

        lesson_context = Path(args.lesson_file).read_text(encoding="utf-8", errors="ignore")

    elif args.lesson_text:

        lesson_context = args.lesson_text

    elif args.text_file:

        lesson_context = text

    else:

        lesson_context = text

    sentences = split_sentences(text)

    if args.mode == "select" and len(sentences) > 1:

                                        

        sentences = [clean(text)]

    if not sentences:

        raise ValueError("没有输入文本。请提供 --text 或 --text-file。")

    plan, retrieval_rows = deterministic_plan(sentences, bm25, action_dict, lesson_context, top_k=int(args.top_k))

    write_jsonl(retrieval_rows, out_dir / "last_mocap_action_retrieval.jsonl")

    (out_dir / "last_mocap_action_plan_deterministic.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] deterministic plan: {out_dir / 'last_mocap_action_plan_deterministic.json'}")

    print(f"[OK] retrieval: {out_dir / 'last_mocap_action_retrieval.jsonl'}")

    if not args.use_chat:

        print(json.dumps(plan, ensure_ascii=False, indent=2))

        return

    prompt = build_prompt_for_chat(sentences, plan, retrieval_rows, lesson_context)

    (out_dir / "last_mocap_action_selector_prompt.md").write_text(prompt, encoding="utf-8")

    ans = deepseek_chat(prompt, args.chat_model)

    (out_dir / "last_mocap_action_plan_chat.json").write_text(ans, encoding="utf-8")

    print(f"[OK] chat prompt: {out_dir / 'last_mocap_action_selector_prompt.md'}")

    print(f"[OK] chat answer: {out_dir / 'last_mocap_action_plan_chat.json'}")

    print(ans)

if __name__ == "__main__":

    main()
