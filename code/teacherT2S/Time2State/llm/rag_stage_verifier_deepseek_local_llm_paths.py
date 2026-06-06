

                       

   

from __future__ import annotations

import argparse

import json

import math

import os

import re

from pathlib import Path

from collections import Counter

from typing import Any

import pandas as pd

INPUT_DIR_DEFAULT = Path(r"D:\code\teacherT2S\Time2State\llm\course_stage_grammar_orientation8")

OUT_DIR_DEFAULT = Path(r"D:\code\teacherT2S\Time2State\llm\rag_stage_verifier_orientation8")

def ensure_dir(p: Path):

    p.mkdir(parents=True, exist_ok=True)

def clean(x: Any) -> str:

    if x is None:

        return ""

    return " ".join(str(x).replace("\r", " ").replace("\n", " ").replace("\t", " ").split()).strip()

def si(x, default=0):

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

        "content": content[:7000],

        "metadata": metadata,

    })

def build_documents(input_dir: Path) -> list[dict]:

    docs = []

    block = read_csv(input_dir / "stage_block_corpus.csv")

    if not block.empty:

        for _, r in block.iterrows():

            stage = clean(r.get("stage",""))

            title = f"阶段块样例 | {stage} | {clean(r.get('video_id',''))}"

            content = (

                f"阶段：{stage}\n"

                f"阶段功能：{clean(r.get('stage_role',''))}\n"

                f"阶段时长：{sf(r.get('stage_duration_sec')):.2f}s\n"

                f"句子数量：{si(r.get('n_sentences'),0)}\n"

                f"开头文本：{clean(r.get('opening_texts',''))}\n"

                f"结尾文本：{clean(r.get('closing_texts',''))}\n"

                f"完整阶段文本：{clean(r.get('full_stage_text',''))}"

            )

            add_doc(docs, "stage_block", title, content, {

                "stage": stage,

                "video_id": clean(r.get("video_id","")),

                "stage_duration_sec": sf(r.get("stage_duration_sec")),

                "n_sentences": si(r.get("n_sentences"),0),

            })

    seq = read_csv(input_dir / "stage_sequence_summary.csv")

    if not seq.empty:

        for _, r in seq.iterrows():

            title = f"阶段顺序样例 | {clean(r.get('video_id',''))}"

            content = (

                f"视频：{clean(r.get('video_id',''))}\n"

                f"阶段名称序列：{clean(r.get('stage_names',''))}\n"

                f"阶段顺序与比例：{clean(r.get('stage_sequence',''))}\n"

                f"总时长：{sf(r.get('total_duration_sec')):.2f}s"

            )

            add_doc(docs, "stage_sequence", title, content, {

                "video_id": clean(r.get("video_id","")),

                "stage_names": clean(r.get("stage_names","")),

                "n_stages": si(r.get("n_stages"),0),

            })

    trans = read_csv(input_dir / "stage_transition_summary.csv")

    if not trans.empty:

        for _, r in trans.iterrows():

            title = f"阶段转移样例 | {clean(r.get('transition_key',''))}"

            content = (

                f"阶段转移：{clean(r.get('transition_key',''))}\n"

                f"出现次数：{si(r.get('count'),0)}\n"

                f"支持视频数：{si(r.get('support_videos'),0)}\n"

                f"转移附近文本：{clean(r.get('example_texts',''))}"

            )

            add_doc(docs, "stage_transition", title, content, {

                "from_stage": clean(r.get("from_stage","")),

                "to_stage": clean(r.get("to_stage","")),

                "transition_key": clean(r.get("transition_key","")),

            })

    sent = read_csv(input_dir / "stage_sentence_corpus.csv")

    if not sent.empty:

                       

        for stage, g in sent.groupby("stage"):

            texts = " ".join(g["text"].astype(str).tolist()[:120])

            title = f"阶段语料总览 | {stage}"

            content = (

                f"阶段：{stage}\n"

                f"样本句子数：{len(g)}\n"

                f"真实教师话语样本：{texts[:6000]}"

            )

            add_doc(docs, "stage_sentence_pool", title, content, {"stage": clean(stage), "n_sentences": int(len(g))})

    return docs

class BM25:

    def __init__(self, docs):

        self.docs = docs

        self.doc_tokens = [tokenize(d["title"] + " " + d["content"]) for d in docs]

        self.N = len(docs)

        self.avgdl = sum(len(t) for t in self.doc_tokens) / max(1, self.N)

        self.df = Counter()

        for toks in self.doc_tokens:

            self.df.update(set(toks))

    def score(self, query: str, top_k: int = 10, doc_types: set[str] | None = None):

        q = tokenize(query)

        q_count = Counter(q)

        out = []

        k1, b = 1.5, 0.75

        for i, d in enumerate(self.docs):

            if doc_types and d["doc_type"] not in doc_types:

                continue

            toks = self.doc_tokens[i]

            tf = Counter(toks)

            dl = len(toks)

            score = 0.0

            for term, _ in q_count.items():

                if term not in tf:

                    continue

                idf = math.log(1 + (self.N - self.df[term] + 0.5) / (self.df[term] + 0.5))

                denom = tf[term] + k1 * (1 - b + b * dl / max(1e-6, self.avgdl))

                score += idf * tf[term] * (k1 + 1) / denom

            out.append((score, d))

        out.sort(key=lambda x: x[0], reverse=True)

        return [{"rank": j+1, "score": s, **d} for j, (s, d) in enumerate(out[:top_k]) if s > 0]

def save_jsonl(items, path: Path):

    with path.open("w", encoding="utf-8") as f:

        for x in items:

            f.write(json.dumps(x, ensure_ascii=False) + "\n")

def load_jsonl(path: Path):

    items = []

    with path.open("r", encoding="utf-8") as f:

        for line in f:

            if line.strip():

                items.append(json.loads(line))

    return items

def build_index(input_dir: Path, out_dir: Path, rebuild: bool):

    ensure_dir(out_dir)

    docs_path = out_dir / "stage_verifier_docs.jsonl"

    if docs_path.exists() and not rebuild:

        print(f"[SKIP] 已存在：{docs_path}")

        return

    docs = build_documents(input_dir)

    if not docs:

        raise RuntimeError("没有构建出阶段 RAG 文档")

    save_jsonl(docs, docs_path)

    pd.DataFrame([{k:v for k,v in d.items() if k != "content"} for d in docs]).to_csv(out_dir / "stage_verifier_doc_index.csv", index=False, encoding="utf-8-sig")

    print(f"[OK] docs={len(docs)} -> {docs_path}")

def load_input_text(args) -> str:

    if args.text_file:

        return Path(args.text_file).read_text(encoding="utf-8", errors="ignore")

    return args.text or args.q

def retrieve_context(docs: list[dict], text: str, top_k: int):

    bm25 = BM25(docs)

    return bm25.score(text, top_k=top_k, doc_types={"stage_block", "stage_sequence", "stage_transition", "stage_sentence_pool"})

def build_prompt(course_text: str, contexts: list[dict]) -> str:

    ctx = "\n".join([

        f"【{c['rank']}｜{c['doc_type']}｜score={c['score']:.3f}】{c['title']}\n{c['content']}"

        for c in contexts

    ])

    return f"""你是课程阶段后验评估专家。你需要判断 AI 生成的课程内容是否符合真实教师的课程阶段安排。

真实教师阶段样例与检索上下文：
{ctx}

待评估课稿：
{course_text}

请输出严格 JSON：
{ 
  "overall_score": 0-100,
  "stage_sequence_judgement": "阶段顺序是否合理",
  "stage_ratio_judgement": "阶段比例是否合理",
  "stage_function_check": [
    { 
      "stage": "导入/新课/反思/结束/其他",
      "is_reasonable": true/false,
      "evidence": "根据课稿和真实样例说明",
      "problem": "存在的问题",
      "revision": "修改建议"
    } 
  ],
  "missing_stages": [],
  "redundant_or_overlong_stages": [],
  "导入是否合理": "必须单独判断导入是否完成引题、激活旧知、提出问题",
  "新课是否合理": "必须单独判断新课是否承担核心讲解和问题推进",
  "反思是否合理": "必须单独判断反思是否有总结/迁移/升华",
  "结束是否合理": "必须单独判断结束是否自然收束",
  "final_revision_plan": "如何重写或调整阶段结构"
} 

注意：
1. 你不是评价动作，而是评价课程阶段安排。
2. 必须以真实教师阶段样例作为参照。
3. 如果导入只有寒暄没有问题引入，要指出。
4. 如果新课过短或反思过长，要指出。
5. 如果阶段顺序缺失或混乱，要指出。
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

            {"role": "system", "content": "你是严谨的课程阶段后验评估专家。"},

            {"role": "user", "content": prompt},

        ],

        temperature=0.2,

    )

    return resp.choices[0].message.content

def parse_args():

    ap = argparse.ArgumentParser()

    ap.add_argument("mode", choices=["build", "verify"])

    ap.add_argument("--input-dir", type=str, default=str(INPUT_DIR_DEFAULT))

    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR_DEFAULT))

    ap.add_argument("--rebuild", action="store_true")

    ap.add_argument("--text", type=str, default="")

    ap.add_argument("--q", type=str, default="")

    ap.add_argument("--text-file", type=str, default="")

    ap.add_argument("--top-k", type=int, default=10)

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

    docs_path = out_dir / "stage_verifier_docs.jsonl"

    if not docs_path.exists():

        build_index(input_dir, out_dir, rebuild=True)

    docs = load_jsonl(docs_path)

    course_text = load_input_text(args)

    if not course_text.strip():

        raise ValueError("没有待评估课稿。请提供 --text 或 --text-file")

    contexts = retrieve_context(docs, course_text, args.top_k)

    save_jsonl(contexts, out_dir / "last_stage_verifier_retrieval.jsonl")

    prompt = build_prompt(course_text, contexts)

    (out_dir / "last_stage_verifier_prompt.md").write_text(prompt, encoding="utf-8")

    print(f"[OK] prompt: {out_dir / 'last_stage_verifier_prompt.md'}")

    print(f"[OK] retrieval: {out_dir / 'last_stage_verifier_retrieval.jsonl'}")

    if args.no_chat:

        print("未调用 DeepSeek。")

        return

    ans = deepseek_chat(prompt, args.chat_model)

    (out_dir / "last_stage_verification.json").write_text(ans, encoding="utf-8")

    print(f"[OK] answer: {out_dir / 'last_stage_verification.json'}")

    print(ans)

if __name__ == "__main__":

    main()
