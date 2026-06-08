# Figure Scripts

This folder contains auxiliary scripts for generating paper figures and visualization materials used in the ER-MSSF experiments.

## Files

| Script | Purpose |
|---|---|
| `build_final_result_figures.py` | Builds the main benchmark-result figures and rank-style comparison outputs for ER-MSSF and baseline methods. |
| `build_mocap_branch_case_visualization.py` | Builds the MoCap case visualization for selected candidate branches, the fused meta-state sequence, and the ground-truth sequence. |

## Suggested Location

Place this folder under:

```text
supplements/figure_scripts/
```

or:

```text
code/teacherT2S/visualization/
```

The first option is recommended because these scripts are mainly used for generating paper figures rather than running the core ER-MSSF pipeline.

## Usage Notes

These scripts were used to generate paper-level SVG/PDF/PNG visualizations from existing experiment outputs. They do not replace the main training or evaluation pipeline.

Before running the scripts, update the local path configuration near the top of each file, such as:

```text
D:\code\teacherT2S\...
```

to match your local repository and result directories.

## Dependencies

The scripts mainly depend on:

```text
numpy
pandas
matplotlib
scipy
```

The MoCap branch visualization script also requires the corresponding ER-MSSF runner, MoCap configuration file, and public MoCap data prepared in the expected directory structure.

## Privacy Note

These scripts operate on processed experimental results and public benchmark data. They should not include raw classroom videos, raw audio recordings, or directly identifiable personal information.
