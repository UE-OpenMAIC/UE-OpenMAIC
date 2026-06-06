# UE-OpenMAIC / ER-MSSF

This repository is built upon and extends the original OpenMAIC project:

**Original OpenMAIC Repository:**  
https://github.com/THU-MAIC/OpenMAIC

OpenMAIC, short for **Open Multi-Agent Interactive Classroom**, is an open-source AI platform for generating rich and interactive classroom experiences with AI teachers, AI classmates, slides, quizzes, simulations, whiteboard interaction, and real-time discussions.

Based on this foundation, this repository extends OpenMAIC toward **state-driven virtual AI teacher generation** by integrating unsupervised time-series state detection, teacher posture state modeling, cross-video prototype alignment, and UE-based virtual teacher motion organization.

This repository provides the code and ethically releasable materials for the paper:

**Unsupervised State Detection with Entropy Regularization for Time Series Data with Application to Virtual AI Teacher**

The repository contains implementation and reproduction materials for:

1. **ER-MSSF**: Entropy-Regularized Multi-branch State Sequence Fusion for unsupervised time-series state detection.
2. **Public benchmark experiments** on six time-series state detection datasets.
3. **CLaP transfer experiments** for evaluating the cross-framework applicability of the reliability-based branch selection mechanism.
4. **Virtual AI teacher application**, including teacher posture state detection, cross-video prototype alignment, expert-label evaluation, and UE/OpenMAIC-based virtual teacher materials.

---

## Relation to OpenMAIC

This project inherits the OpenMAIC concept of an AI-driven interactive classroom and further extends it in the following aspects:

- introducing unsupervised teaching posture state detection with ER-MSSF;
- detecting and aligning teacher posture states from classroom video time series;
- mapping detected global teaching states to virtual teacher motion organization;
- integrating state-driven behavior control into a UE/OpenMAIC-based virtual AI teacher system.

The original OpenMAIC project focuses on multi-agent interactive classroom generation, while this repository focuses on **state detection, teaching behavior modeling, and virtual AI teacher motion organization** built on top of the OpenMAIC-style classroom framework.

---

## Repository Structure

```text
UE-OpenMAIC/
│
├── OpenMAIC/
│   └── UE/OpenMAIC-related virtual AI teacher system materials.
│
├── baseline/
│   └── Baseline reproduction scripts and running instructions.
│
├── benchmark/
│   └── Expert annotations, expert-evaluation results, final validation tables,
│       benchmark records, and paper-related result documents.
│
├── code/
│   └── teacherT2S/
│       ├── Time2State/
│       │   ├── Time2State/
│       │   │   └── Adapted Time2State backbone.
│       │   ├── TSpy/
│       │   │   └── Time-series utility package.
│       │   ├── data/
│       │   │   └── Public benchmark data used by the Time2State-side scripts.
│       │   ├── scripts/
│       │   │   └── Supporting scripts.
│       │   ├── run_teacher_state_detection_orientation8.py
│       │   ├── align_cross_video_prototypes_x1_orientation8.py
│       │   ├── evaluate_expert_labels_teacher_mask.py
│       │   └── make_final_expert_validation_table.py
│       │
│       ├── our/
│       │   ├── ER-MSSF benchmark implementation.
│       │   └── mocap_topk_sensitivity/
│       │       └── MoCap Top-K sensitivity experiment.
│       │
│       └── ourClap/
│           └── CLaP transfer experiment scripts.
│
├── dataset/
│   └── Ethically releasable processed data, demo data, and data preparation notes.
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## Method Overview

ER-MSSF addresses unsupervised state detection under realistic unlabeled settings, where ground-truth labels are unavailable for selecting the best candidate branch or configuration.

The framework constructs multiple candidate Time2State branches from different temporal scales and parameter configurations, evaluates branch reliability without using ground-truth labels, and fuses reliable branches through meta-state alignment and reliability-weighted voting.

The branch reliability score combines three aspects:

- **state-distribution health**,
- **inter-branch consistency**,
- **prediction stability**.

For the virtual AI teacher application, ER-MSSF is applied to teacher posture time series extracted from classroom videos. The detected local states are further aligned across videos through X1-only prototype aggregation, producing global teaching posture states that are used for state-driven virtual teacher motion organization.

---

## Paper Reproduction Map

| Paper component | Repository location |
|---|---|
| ER-MSSF benchmark experiments | `code/teacherT2S/our/` |
| Public benchmark baselines | `baseline/` |
| CLaP transfer experiment | `code/teacherT2S/ourClap/` |
| MoCap Top-K sensitivity analysis | `code/teacherT2S/our/mocap_topk_sensitivity/` |
| Adapted Time2State backbone | `code/teacherT2S/Time2State/Time2State/` |
| Teacher posture state detection | `code/teacherT2S/Time2State/run_teacher_state_detection_orientation8.py` |
| Cross-video prototype alignment | `code/teacherT2S/Time2State/align_cross_video_prototypes_x1_orientation8.py` |
| Expert-label evaluation | `code/teacherT2S/Time2State/evaluate_expert_labels_teacher_mask.py` |
| Final expert validation table | `code/teacherT2S/Time2State/make_final_expert_validation_table.py` |
| Expert annotations and evaluation records | `benchmark/` |
| UE/OpenMAIC virtual teacher materials | `OpenMAIC/` |
| Dataset and privacy notes | `dataset/README.md` |

---

## Teacher Application Pipeline

The teacher-application scripts are located in:

```text
code/teacherT2S/Time2State/
```

The main scripts are:

```text
run_teacher_state_detection_orientation8.py
align_cross_video_prototypes_x1_orientation8.py
evaluate_expert_labels_teacher_mask.py
make_final_expert_validation_table.py
```

Their roles are:

| Script | Function |
|---|---|
| `run_teacher_state_detection_orientation8.py` | Runs multi-branch teacher posture state detection. |
| `align_cross_video_prototypes_x1_orientation8.py` | Aligns local meta-states across videos into global teaching posture states. |
| `evaluate_expert_labels_teacher_mask.py` | Evaluates detected global states against expert annotations under teacher-only masks. |
| `make_final_expert_validation_table.py` | Generates the final expert validation table used in the paper. |

Recommended execution order:

```bash
cd code/teacherT2S/Time2State

python run_teacher_state_detection_orientation8.py
python align_cross_video_prototypes_x1_orientation8.py
python evaluate_expert_labels_teacher_mask.py
python make_final_expert_validation_table.py
```

Please update the input and output paths according to your local data location before running the scripts.

---

## Benchmark and Expert Annotation Files

The `benchmark/` directory contains paper-related benchmark documents and expert annotation materials, including:

```text
benchmark/
│
├── expertA/
├── expertB/global/
├── expertC/global/
├── expertD/global/
├── _eval_BCD_global_3class_teacher_mask_grid/
├── _final_expert_validation_table/
├── datasetN/
└── digitalAction/
```

These files support the teacher-application evaluation part of the paper, including:

- expert annotation records,
- B/C/D expert global-label evaluation,
- consensus-label evaluation,
- conservative merged-label evaluation,
- final expert validation table.

Raw classroom videos and raw audio are not included for privacy reasons.

---

## Benchmark Experiments

The ER-MSSF benchmark implementation is located in:

```text
code/teacherT2S/our/
```

The baseline reproduction materials are located in:

```text
baseline/
```

The public benchmark experiments involve six datasets:

```text
Synthetic
MoCap
ActRecTut
PAMAP2
UCR-SEG
USC-HAD
```

Please refer to the subdirectory README files and configuration files for detailed running instructions.

---

## CLaP Transfer Experiment

The CLaP transfer experiment is located in:

```text
code/teacherT2S/ourClap/
```

This part evaluates whether the proposed reliability-based branch selection mechanism can be transferred to another state detection framework.

---

## MoCap Top-K Sensitivity Experiment

The MoCap Top-K sensitivity experiment is located in:

```text
code/teacherT2S/our/mocap_topk_sensitivity/
```

This experiment analyzes the influence of the number of selected branches in the fusion stage.

The sensitivity experiment is used only for post-hoc robustness analysis. In the main experiments, the default Top-K value is fixed before evaluation and kept consistent across datasets.

---

## Dataset Availability

Raw classroom videos and raw audio recordings are **not released** because they contain identifiable human information and were collected under informed consent for internal research use only.

This repository releases:

- code for ER-MSSF and related experiments,
- processed and ethically releasable materials where applicable,
- expert annotation records and evaluation scripts where appropriate,
- public benchmark preparation instructions,
- selected anonymized examples or derived features when available,
- configuration files and reproduction scripts.

For public benchmark datasets, please obtain the datasets from their official sources or prepare them according to the instructions in:

```text
dataset/README.md
```

or the corresponding data folder under:

```text
code/teacherT2S/Time2State/data/
```

---

## Privacy and Ethics Notes

The classroom-video data were collected with informed consent. Raw videos, raw audio, and directly identifiable information are excluded from this repository.

When illustrative frames, pose features, or examples are provided, identifiable information is removed or anonymized where necessary.

The expert annotation files are released only where they do not contain raw identifiable video/audio content.

---

## Environment

The experiments were conducted on a workstation with:

```text
NVIDIA GeForce RTX 4090 GPU
Intel Core Ultra 9 265K CPU
```

A typical Python environment can be prepared with:

```bash
pip install -r requirements.txt
```

Some baseline methods may require additional dependencies. Please refer to the README files in the corresponding subdirectories.

---

## Notes

This repository is organized to support paper review and reproduction. Some large intermediate files, private classroom recordings, and personally identifiable materials are not included.

The released scripts and materials are intended to reproduce the main experimental pipeline, result tables, and application workflow described in the paper.

---

## Citation

If you use this repository, please cite the corresponding paper:

```bibtex
@misc{yan2026ermssf,
  title={Unsupervised State Detection with Entropy Regularization for Time Series Data with Application to Virtual AI Teacher},
  author={Yan, Chufei and Lv, Yiyan and Liu, Shaoyin and Meng, Yanli and Wang, Yulei},
  year={2026},
  note={Manuscript under review}
}
```

---

## Contact

For questions about the code or reproduction materials, please contact:

```text
Chufei Yan
yanchufei@nenu.edu.cn
```