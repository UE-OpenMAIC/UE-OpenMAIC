

                       

   

from __future__ import annotations

import argparse

import json

import math

import os

import re

from pathlib import Path

from collections import Counter, defaultdict

from typing import Any

import pandas as pd

INPUT_DIR_DEFAULT = Path(r"D:\code\teacherT2S\Time2State\llm\text_action_grammar_orientation8")

OUT_DIR_DEFAULT = Path(r"D:\code\teacherT2S\Time2State\llm\rag_action_selector_orientation8")

def ensure_dir(p: Path):

    p.mkdir(parents=True, exist_ok=True)

def clean(x: Any) -> str:

    if x is None:

        return ""

    return " ".join(str(x).replace("\r", " ").replace("\n", " ").replace("\t", " ").split()).strip()

def si(x, default=-1):

    try:

        if pd.isna(x):

            return int(default)

        return int(float(x))

    except Exception:

        return int(default)

def sf(x, default=0.0):

    try:

        if pd.isna(x):

            return float(default)

        return float(x)

    except Exception:

        return float(default)

def read_csv(path: Path) -> pd.DataFrame:

    if not path.exists():

        print(f"[WARN] 缺少文件：{path}")

        return pd.DataFrame()

    return pd.read_csv(path)

def tokenize(text: str) -> list[str]:

    text = clean(text).lower()

    words = re.findall(r"[a-zA-Z0-9_]+", text)

    chars = [c for c in text if "\u4e00" <= c <= "\u9fff"]

    bigrams = [chars[i] + chars[i+1] for i in range(len(chars) - 1)]

    return words + chars + bigrams

def add_doc(docs: list[dict], doc_type: str, title: str, content: str, metadata: dict):

    content = clean(content)

    if not content:

        return

    docs.append({

        "doc_id": f"{doc_type}_{len(docs):08d}",

        "doc_type": doc_type,

        "title": clean(title),

        "content": content[:6000],

        "metadata": metadata,

    })

def build_documents(input_dir: Path) -> list[dict]:

    docs = []

    sent = read_csv(input_dir / "sentence_text_action_alignment.csv")

    if not sent.empty:

        for _, r in sent.iterrows():

            gs = si(r.get("global_state", -1))

            ls = si(r.get("local_meta_state", -1))

            title = f"句子动作样本 | {clean(r.get('stage',''))} | G{gs} L{ls}"

            content = (

                f"阶段：{clean(r.get('stage',''))}\n"

                f"教师文本：{clean(r.get('text',''))}\n"

                f"global_state：G{gs}\n"

                f"local_state：L{ls}\n"

                f"prototype：{clean(r.get('prototype_id',''))}\n"

                f"时间：{sf(r.get('text_start_sec')):.2f}-{sf(r.get('text_end_sec')):.2f}s\n"

                f"重叠状态：{clean(r.get('all_overlapped_states',''))}"

            )

            add_doc(docs, "sentence_action", title, content, {

                "stage": clean(r.get("stage","")),

                "video_id": clean(r.get("video_id","")),

                "global_state": gs,

                "local_meta_state": ls,

            })

    local = read_csv(input_dir / "local_state_text_examples.csv")

    if not local.empty:

        for _, r in local.iterrows():

            gs = si(r.get("global_state", -1))

            ls = si(r.get("local_meta_state", -1))

            title = f"动作小类摘要 | G{gs} L{ls}"

            content = (

                f"global_state：G{gs}\n"

                f"local_state：{clean(r.get('local_class_key',''))}\n"

                f"主要阶段：{clean(r.get('top_stages',''))}\n"

                f"对应文本例子：{clean(r.get('example_texts',''))}"

            )

            add_doc(docs, "local_state_summary", title, content, {

                "global_state": gs,

                "local_meta_state": ls,

                "stage": clean(r.get("top_stages","")),

            })

    global_df = read_csv(input_dir / "global_to_local_state_text_summary.csv")

    if not global_df.empty:

        for _, r in global_df.iterrows():

            gs = si(r.get("global_state", -1))

            title = f"跨视频动作大类摘要 | G{gs}"

            content = (

                f"global_state：G{gs}\n"

                f"支持视频数：{si(r.get('support_videos',0),0)}\n"

                f"包含 local 小类：{clean(r.get('local_classes',''))}\n"

                f"文本例子：{clean(r.get('example_texts',''))}"

            )

            add_doc(docs, "global_state_summary", title, content, {"global_state": gs})

    for fname, doc_type in [

        ("global_transition_grammar.csv", "transition_global"),

        ("local_transition_grammar.csv", "transition_local"),

        ("stage_conditioned_transition_grammar.csv", "transition_stage"),

    ]:

        df = read_csv(input_dir / fname)

        if df.empty:

            continue

        for _, r in df.iterrows():

            key = clean(r.get("transition_key", ""))

            title = f"动作转移 | {key}"

            content = (

                f"转移：{key}\n"

                f"阶段：{clean(r.get('stage','未限定'))}\n"

                f"from_global：G{si(r.get('from_global_state',-1))}\n"

                f"to_global：G{si(r.get('to_global_state',-1))}\n"

                f"出现次数：{si(r.get('count',0),0)}\n"

                f"支持视频数：{si(r.get('support_videos',0),0)}\n"

                f"附近文本：{clean(r.get('example_texts',''))}"

            )

            add_doc(docs, doc_type, title, content, {

                "stage": clean(r.get("stage","")),

                "transition_key": key,

                "from_global_state": si(r.get("from_global_state",-1)),

                "to_global_state": si(r.get("to_global_state",-1)),

            })

    return docs

class BM25:

    def __init__(self, docs: list[dict]):

        self.docs = docs

        self.doc_tokens = [tokenize(d["title"] + " " + d["content"]) for d in docs]

        self.N = len(docs)

        self.avgdl = sum(len(t) for t in self.doc_tokens) / max(1, self.N)

        self.df = Counter()

        for toks in self.doc_tokens:

            self.df.update(set(toks))

    def score(self, query: str, doc_types: set[str] | None = None, top_k: int = 8):

        q = tokenize(query)

        if not q:

            return []

        q_count = Counter(q)

        out = []

        k1, b = 1.5, 0.75

        for i, d in enumerate(self.docs):

            if doc_types and d["doc_type"] not in doc_types:

                continue

            toks = self.doc_tokens[i]

            dl = len(toks)

            tf = Counter(toks)

            score = 0.0

            for term, qtf in q_count.items():

                if term not in tf:

                    continue

                idf = math.log(1 + (self.N - self.df[term] + 0.5) / (self.df[term] + 0.5))

                denom = tf[term] + k1 * (1 - b + b * dl / max(1e-6, self.avgdl))

                score += idf * tf[term] * (k1 + 1) / denom

            out.append((score, d))

        out.sort(key=lambda x: x[0], reverse=True)

        return [{"rank": j+1, "score": s, **d} for j, (s, d) in enumerate(out[:top_k]) if s > 0]

def save_jsonl(items: list[dict], path: Path):

    with path.open("w", encoding="utf-8") as f:

        for x in items:

            f.write(json.dumps(x, ensure_ascii=False) + "\n")

def load_jsonl(path: Path) -> list[dict]:

    items = []

    with path.open("r", encoding="utf-8") as f:

        for line in f:

            if line.strip():

                items.append(json.loads(line))

    return items

def build_index(input_dir: Path, out_dir: Path, rebuild: bool):

    ensure_dir(out_dir)

    docs_path = out_dir / "action_selector_docs.jsonl"

    if docs_path.exists() and not rebuild:

        print(f"[SKIP] 已存在：{docs_path}")

        return

    docs = build_documents(input_dir)

    if not docs:

        raise RuntimeError("没有构建出动作 RAG 文档")

    save_jsonl(docs, docs_path)

    pd.DataFrame([{k:v for k,v in d.items() if k != "content"} for d in docs]).to_csv(out_dir / "action_selector_doc_index.csv", index=False, encoding="utf-8-sig")

    print(f"[OK] docs={len(docs)} -> {docs_path}")

def split_sentences(text: str) -> list[str]:

    text = clean(text)

    parts = re.split(r"[。！？!?；;]\s*", text)

    return [p.strip() for p in parts if p.strip()]

def load_input_text(args) -> str:

    if args.text_file:

        return Path(args.text_file).read_text(encoding="utf-8", errors="ignore")

    return args.text or args.q

def retrieve_context(docs: list[dict], sentence: str, top_k: int) -> list[dict]:

    bm25 = BM25(docs)

                       

    return bm25.score(

        sentence,

        doc_types={"sentence_action", "local_state_summary", "global_state_summary", "transition_global", "transition_local", "transition_stage"},

        top_k=top_k,

    )

def build_prompt(sentences: list[str], contexts: dict[int, list[dict]], stage_hint: str = "") -> str:

    blocks = []

    for i, s in enumerate(sentences, start=1):

        ctx = contexts.get(i, [])

        ctx_text = "\n".join([

            f"【{c['rank']}｜{c['doc_type']}｜score={c['score']:.3f}】{c['title']}\n{c['content']}"

            for c in ctx

        ])

        blocks.append(f"### 句子 {i}\n{s}\n\n检索上下文：\n{ctx_text}")

    joined = "\n\n".join(blocks)

    return f"""你是数字人教师动作组织规划器。你需要为每个语句安排动作，不是只做四分类，而是要考虑动作元和 A->B 动作序列。

课程阶段提示：{stage_hint or "未指定，请根据语句判断"}

待规划语句与真实教师检索上下文如下：
{joined}

请输出严格 JSON，格式如下：
[
  { 
    "sentence_id": 1,
    "sentence": "...",
    "stage": "导入/新课/反思/结束/其他",
    "teaching_function": "引题/提问/文本细读/解释概念/强调重点/等待回答/情感升华/总结收束等",
    "action_category": "正向站立讲授/静止讲授/侧身讲授/板书书写/过渡",
    "action_primitive": "具体动作元名称，如果无法确定写候选",
    "recommended_global_state": "G?",
    "recommended_local_state_or_example": "L? 或真实样例",
    "previous_current_next": "前一动作 -> 当前动作 -> 后一动作",
    "duration_suggestion": "建议持续时间或是否保持上一动作",
    "transition_reason": "为什么这样接，引用 A->B 规律或真实教师样例",
    "risk": "可能不合理之处"
  } 
]

硬约束：
1. 不要使用遮挡、识别失败、不确定状态作为可驱动动作。
2. 短句优先保持上一动作，长句才考虑切换。
3. 板书书写只在文本细读、公式/关键词书写、屏幕/黑板强调等情况下使用。
4. 必须尽量利用检索到的真实教师动作转移规律。
"""

def deepseek_chat(prompt: str, model: str):

    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not api_key:

        raise RuntimeError("缺少 OPENAI_API_KEY")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com").strip()

    client = OpenAI(api_key=api_key, base_url=base_url)

    resp = client.chat.completions.create(

        model=model,

        messages=[

            {"role": "system", "content": "你是严谨的数字人教师动作组织专家。"},

            {"role": "user", "content": prompt},

        ],

        temperature=0.2,

    )

    return resp.choices[0].message.content

def parse_args():

    ap = argparse.ArgumentParser()

    ap.add_argument("mode", choices=["build", "plan"])

    ap.add_argument("--input-dir", type=str, default=str(INPUT_DIR_DEFAULT))

    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR_DEFAULT))

    ap.add_argument("--rebuild", action="store_true")

    ap.add_argument("--text", type=str, default="")

    ap.add_argument("--q", type=str, default="")

    ap.add_argument("--text-file", type=str, default="")

    ap.add_argument("--stage", type=str, default="")

    ap.add_argument("--top-k", type=int, default=8)

    ap.add_argument("--no-chat", action="store_true")

    ap.add_argument("--chat-model", type=str, default=os.getenv("CHAT_MODEL", "deepseek-chat"))

    return ap.parse_args()

def main():

    args = parse_args()

    input_dir = Path(args.input_dir)

    out_dir = Path(args.out_dir)

    ensure_dir(out_dir)

    if args.mode == "build":

        build_index(input_dir, out_dir, args.rebuild)

        return

    docs_path = out_dir / "action_selector_docs.jsonl"

    if not docs_path.exists():

        build_index(input_dir, out_dir, rebuild=True)

    docs = load_jsonl(docs_path)

    text = load_input_text(args)

    sentences = split_sentences(text)

    if not sentences:

        raise ValueError("没有可规划的句子。请提供 --text 或 --text-file")

    contexts = {}

    for i, sent in enumerate(sentences, start=1):

        contexts[i] = retrieve_context(docs, sent, args.top_k)

    prompt = build_prompt(sentences, contexts, args.stage)

    (out_dir / "last_action_selector_prompt.md").write_text(prompt, encoding="utf-8")

    save_jsonl([{"sentence_id": i, "sentence": s, "contexts": contexts[i]} for i, s in enumerate(sentences, 1)], out_dir / "last_action_selector_retrieval.jsonl")

    print(f"[OK] prompt: {out_dir / 'last_action_selector_prompt.md'}")

    print(f"[OK] retrieval: {out_dir / 'last_action_selector_retrieval.jsonl'}")

    if args.no_chat:

        print("未调用 DeepSeek。")

        return

    ans = deepseek_chat(prompt, args.chat_model)

    (out_dir / "last_action_plan.json").write_text(ans, encoding="utf-8")

    print(f"[OK] answer: {out_dir / 'last_action_plan.json'}")

    print(ans)

if __name__ == "__main__":

    main()
