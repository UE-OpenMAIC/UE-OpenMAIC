# 红/蓝标记过滤 + 文本到动捕动作 RAG 使用说明

把本压缩包里的脚本复制到：

```text
D:\code\teacherT2S\Time2State\llm
```

## 1. 新增脚本

```text
build_mocap_action_rag_from_marked_video.py
rag_mocap_action_selector.py
run_build_mocap_action_rag.cmd
run_select_mocap_action_demo.cmd
```

原来的阶段后验脚本没有删，可以继续用；新的两个脚本负责“文本 -> 动捕动作”。

## 2. build 脚本做什么

`build_mocap_action_rag_from_marked_video.py` 会：

1. 遍历 `D:\code\teacherT2S\yolo\input` 下所有视频；
2. 对每个视频按 12 fps 扫描左上角 20×20 色块；
3. 默认删除红色、蓝色标记对应的 ASR 文本；
4. 将剩余文本和 layer2/final_meta 的动作时间段对齐；
5. 读取 `D:\code\teacherT2S\doc\digitalAction` 下的 local→动捕动作映射表；
6. 把每条教师文本落到 MC01-MC15；
7. 输出 RAG 库。

主要输出路径：

```text
D:\code\teacherT2S\Time2State\llm\mocap_action_rag
```

主要文件：

```text
teacher_text_mocap_alignment_filtered.csv
teacher_text_dropped_by_red_blue_marker.csv
mocap_action_text_examples.csv
mocap_transition_grammar.csv
mocap_action_rag_docs.jsonl
mocap_action_rag_doc_index.csv
```

## 3. 构建 RAG 库

```powershell
cd D:\code\teacherT2S\Time2State\llm
conda activate t2s-llm
python .\build_mocap_action_rag_from_marked_video.py --rebuild --save-marker-masks
```

或双击：

```text
run_build_mocap_action_rag.cmd
```

如果你的视频左上角除了红/蓝还要跳过绿，可以运行：

```powershell
python .\build_mocap_action_rag_from_marked_video.py --skip-marker-colors red,blue,green --rebuild
```

## 4. 输入文本，返回动捕动作

单句：

```powershell
python .\rag_mocap_action_selector.py select --text "请同学们看屏幕上的这个关键词"
```

整篇课稿逐句规划：

```powershell
python .\rag_mocap_action_selector.py plan --text-file .\demo_lesson.txt
```

输出路径：

```text
D:\code\teacherT2S\Time2State\llm\rag_mocap_action_selector\last_mocap_action_plan_deterministic.json
D:\code\teacherT2S\Time2State\llm\rag_mocap_action_selector\last_mocap_action_retrieval.jsonl
```

## 5. 使用 DeepSeek 二次裁决

默认不调用 API，只用本地 RAG 检索直接返回 MC 动作。若要调用 DeepSeek：

```powershell
$env:OPENAI_API_KEY="YOUR_API_KEY"
$env:OPENAI_BASE_URL="https://api.deepseek.com"
$env:CHAT_MODEL="deepseek-chat"
python .\rag_mocap_action_selector.py plan --text-file .\demo_lesson.txt --use-chat
```

输出：

```text
last_mocap_action_plan_chat.json
last_mocap_action_selector_prompt.md
```

## 6. 关键参数

```powershell
--marker-sample-fps 12
--skip-marker-colors red,blue
--drop-marker-overlap-threshold 0.15
--min-text-action-overlap-ratio 0.10
```

含义：

- `marker-sample-fps`：每秒抽多少帧扫描左上角色块；默认 12，与姿态 CSV 常用帧率一致。
- `skip-marker-colors`：哪些颜色视为非教师文本；默认红/蓝。
- `drop-marker-overlap-threshold`：一句 ASR 文本中红/蓝采样点占比超过多少就删除。
- `min-text-action-overlap-ratio`：文本与动作段重叠比例低于该值时不强行对齐。

## 7. 输出 JSON 字段

`rag_mocap_action_selector.py` 返回的核心字段：

```json
{
  "sentence_id": 1,
  "sentence": "请同学们看屏幕上的这个关键词",
  "recommended_mocap_id": "MC07",
  "recommended_mocap_action": "侧身/斜侧指向屏幕",
  "recommended_mocap_coarse": "侧向讲授",
  "confidence_score": 12.34,
  "top_candidates": []
}
```

其中 `recommended_mocap_id` 就是后续驱动数字人教师时要接的动捕动作编号。
