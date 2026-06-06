# UE-OpenMAIC

This project is based on **OpenMAIC**.

UE-OpenMAIC repository: https://github.com/UE-OpenMAIC/UE-OpenMAIC

OpenMAIC repository: https://github.com/THU-MAIC/OpenMAIC

## Paper

This repository provides the code and ethically releasable materials for:

**Unsupervised State Detection with Entropy Regularization for Time Series Data with Application to Virtual AI Teacher**

The repository contains three main parts:

1. **ER-MSSF benchmark experiments** for unsupervised time-series state detection on six public datasets.
2. **CLaP transfer experiments** for testing the cross-framework applicability of the reliability-based branch selection mechanism.
3. **Virtual AI teacher application materials**, including teacher posture state detection, cross-video prototype alignment, expert-label evaluation, and OpenMAIC/UE-based system materials.

## Repository Map

- `code/teacherT2S/our/`: ER-MSSF implementation and benchmark scripts.
- `code/teacherT2S/ourClap/`: CLaP transfer experiment scripts.
- `code/teacherT2S/Time2State/`: adapted Time2State backbone and teacher-application scripts.
- `baseline/`: baseline reproduction scripts and reference outputs.
- `benchmark/`: result tables, evaluation records, and paper-related benchmark documents.
- `dataset/`: ethically releasable processed data and dataset preparation notes.
- `OpenMAIC/`: OpenMAIC/UE-related application source materials.

## Data Availability

Raw classroom videos and audio recordings are not released because they contain identifiable human information and were collected under informed consent for internal research use only. We release ethically shareable processed materials, scripts, configuration files, and selected anonymized examples where applicable. Public benchmark datasets should be obtained from their official sources or prepared following the instructions in this repository.

## Main Reproduction Entry

Please see:

- `baseline/README.md` for baseline reproduction.
- `code/teacherT2S/README.md` for ER-MSSF and teacher-application scripts.
- `dataset/README.md` for dataset preparation and privacy notes.

## Repository Structure

```text
UE-OpenMAIC/
+-- baseline/
+-- dataset/
+-- benchmark/
+-- code/
+-- OpenMAIC/
```
