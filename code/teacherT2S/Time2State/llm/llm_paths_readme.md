# 新版补充：红/蓝标记过滤 + 文本到动捕动作 RAG

你这次要的核心功能不再只是输出 G/local，而是输出可驱动数字人教师的 MC01-MC15 动捕动作。新增说明见：

```text
mocap_rag_readme.md
```

新增脚本：

```text
build_mocap_action_rag_from_marked_video.py
rag_mocap_action_selector.py
run_build_mocap_action_rag.cmd
run_select_mocap_action_demo.cmd
```

推荐先运行：

```powershell
python .\build_mocap_action_rag_from_marked_video.py --rebuild --save-marker-masks
python .\rag_mocap_action_selector.py plan --text-file .\demo_lesson.txt
```

---

# llm 路径版使用说明

把这三个脚本复制到：

```text
D:\code\teacherT2S\Time2State\llm
```

三个脚本分别是：

```text
build_text_action_and_stage_grammar_orientation8_llm_paths.py
rag_action_selector_deepseek_local_llm_paths.py
rag_stage_verifier_deepseek_local_llm_paths.py
```

## 输出路径

### 1. build 脚本输出两个库

```text
动作组织库：
D:\code\teacherT2S\Time2State\llm\text_action_grammar_orientation8

课程阶段库：
D:\code\teacherT2S\Time2State\llm\course_stage_grammar_orientation8
```

### 2. 动作选择 RAG 输出

```text
D:\code\teacherT2S\Time2State\llm\rag_action_selector_orientation8
```

主要输出：

```text
action_selector_docs.jsonl
action_selector_doc_index.csv
last_action_selector_prompt.md
last_action_selector_retrieval.jsonl
last_action_plan.json
```

### 3. 阶段后验 RAG 输出

```text
D:\code\teacherT2S\Time2State\llm\rag_stage_verifier_orientation8
```

主要输出：

```text
stage_verifier_docs.jsonl
stage_verifier_doc_index.csv
last_stage_verifier_prompt.md
last_stage_verifier_retrieval.jsonl
last_stage_verification.json
```

## 推荐运行顺序

```powershell
cd D:\code\teacherT2S\Time2State\llm
conda activate t2s-llm

python .\build_text_action_and_stage_grammar_orientation8_llm_paths.py

python .\rag_action_selector_deepseek_local_llm_paths.py build
python .\rag_stage_verifier_deepseek_local_llm_paths.py build
```

## DeepSeek 环境变量

```powershell
$env:OPENAI_API_KEY="YOUR_API_KEY"
$env:OPENAI_BASE_URL="https://api.deepseek.com"
$env:CHAT_MODEL="deepseek-chat"
```

## 给每句话安排动作

```powershell
python .\rag_action_selector_deepseek_local_llm_paths.py plan `
  --text-file .\new_lesson.txt `
  --chat-model deepseek-chat
```

## 检查课稿阶段安排

```powershell
python .\rag_stage_verifier_deepseek_local_llm_paths.py verify `
  --text-file .\new_lesson.txt `
  --chat-model deepseek-chat
```
