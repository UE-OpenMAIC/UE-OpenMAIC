# Baseline Scripts Usage

This directory contains baseline implementations and precomputed outputs used by UE-OpenMAIC. Most baselines are organized with the same pattern:

```text
baseline/
  <method>/
    <dataset>/
      RUN_*.cmd
      run_*.py
      *_config.txt
      results_*/
    _shard/ or _shared/
```

## Quick Start

Run commands from the repository root:

```powershell
cd D:\code\OpenMAIC
```

For Windows users, the preferred entry point is usually the dataset-level `.cmd` script:

```powershell
.\baseline\t2s\uschad\RUN_T2S_USCHAD_NO_FLASH_LOGGER.cmd
```

When a Python runner is available, it can also be called directly:

```powershell
python .\baseline\t2s\uschad\run_t2s_uschad.py
```

Each dataset folder normally contains a `*_config.txt` file. Edit that file before running if you need to change data paths, output paths, window settings, labels, or method-specific parameters.

Outputs are written to the corresponding `results_*` directory under each dataset folder. Existing result files in this repository are kept as reference outputs.

## Common Workflow

1. Choose a baseline method, for example `t2s`, `ticc`, `Clap`, or `M2QD_CSPA`.
2. Choose a dataset folder, for example `uschad`, `mocap`, `pamap2`, `synthetic`, `ucrseg`, or `actrectut`.
3. Check the dataset configuration file, usually named like `uschad_config.txt` or `pamap2_zero_config.txt`.
4. Run the matching `RUN_*.cmd` script from the repository root.
5. Inspect the generated files in the local `results_*` folder.

## Baseline Methods

### AutoPlait

Path: `baseline/autoplait/`

AutoPlait provides dataset-level command scripts:

- `actrectut/RUN_AUTOPLAIT_ACTRECTUT.cmd`
- `mocap/RUN_AUTOPLAIT_MOCAP.cmd`
- `pamap2/RUN_AUTOPLAIT_PAMAP2_ZERO.cmd`
- `synthetic/RUN_AUTOPLAIT_SYNTHETIC.cmd`
- `ucrseg/RUN_AUTOPLAIT_UCRSEG.cmd`
- `uschad/RUN_AUTOPLAIT_USCHAD.cmd`

AutoPlait also includes `_shard/autoplait_wsl_bridge.cmd`, which indicates that some runs may depend on a WSL-side AutoPlait executable or bridge environment.

### CLaP

Path: `baseline/Clap/`

Use the dataset-level `RUN_CLAP_*.cmd` scripts, or run the matching Python runner directly:

- `actrectut/run_clap_actrectut.py`
- `mocap/run_clap_mocap.py`
- `pamap2/run_clap_pamap2_zero_all_full_clasp_suss.py`
- `synthetic/run_clap_synthetic.py`
- `ucrseg/run_clap_ucrseg.py`
- `uschad/run_clap_uschad.py`

Configuration files are placed in each dataset folder, such as `pamap2/pamap2_zero_all_full_clasp_suss_config.txt`.

### ClaSP

Path: `baseline/clasp/`

Run the dataset-level command scripts:

- `actrectut/RUN_CLASP_ACTRECTUT.cmd`
- `mocap/RUN_CLASP_MOCAP.cmd`
- `pamap2/RUN_CLASP_PAMAP2_ZERO.cmd`
- `synthetic/RUN_CLASP_SYNTHETIC.cmd`
- `ucrseg/RUN_CLASP_UCRSEG.cmd`
- `uschad/RUN_CLASP_USCHAD.cmd`

Each dataset folder includes a `*_clasp_config.txt` file.

### Classification Label Profile

Path: `baseline/classification-label-profile-main/`

This baseline keeps its original project layout. Start with:

- `baseline/classification-label-profile-main/README.md`
- `baseline/classification-label-profile-main/src/`
- `baseline/classification-label-profile-main/notebooks/`
- `baseline/classification-label-profile-main/requirements.txt`

Install its dependencies in a suitable Python environment before running notebooks or source scripts.

### E2USD

Path: `baseline/e2usd/`

Use the strict fixed live command scripts:

- `actrectut/RUN_E2USD_ACTRECTUT_STRICT_FIXED_LIVE.cmd`
- `mocap/RUN_E2USD_MOCAP_STRICT_FIXED_LIVE.cmd`
- `pamap2/RUN_E2USD_PAMAP2_ZERO_STRICT_FIXED_LIVE.cmd`
- `synthetic/RUN_E2USD_SYNTHETIC_STRICT_FIXED_LIVE.cmd`
- `ucrseg/RUN_E2USD_UCRSEG_STRICT_FIXED_LIVE.cmd`
- `uschad/RUN_E2USD_USCHAD_STRICT_FIXED_LIVE.cmd`

Python runners are also available as `run_e2usd_*.py` in each dataset folder.

### EC-TDWM

Path: `baseline/EC_TDWM/`

Use the dataset-level `RUN_EC_TDWM_*.cmd` scripts, or run the matching `run_ec_tdwm_*.py` file directly. Shared implementation code is under `baseline/EC_TDWM/_shared/`.

### HVGH

Path: `baseline/hvgh/`

Use the dataset-level command scripts:

- `actrectut/RUN_HVGH_ACTRECTUT.cmd`
- `mocap/RUN_HVGH_MOCAP.cmd`
- `pamap2/RUN_HVGH_PAMAP2_ZERO.cmd`
- `synthetic/RUN_HVGH_SYNTHETIC.cmd`
- `ucrseg/RUN_HVGH_UCRSEG.cmd`
- `uschad/RUN_HVGH_USCHAD.cmd`

The original HVGH notes are stored under `baseline/hvgh/_shard/original/`.

### M2QD-CSPA

Path: `baseline/M2QD_CSPA/`

Use the dataset-level `RUN_M2QD_CSPA_*.cmd` scripts, or call the matching `run_m2qd_cspa_*.py` file directly. Shared method code is under the method directory and dataset-level outputs are stored in `results_m2qd_cspa_*` folders.

### T2S

Path: `baseline/t2s/`

Use the dataset-level command scripts:

- `actrectut/RUN_T2S_ACTRECTUT_NO_FLASH_LOGGER.cmd`
- `mocap/RUN_T2S_MOCAP_NO_FLASH_LOGGER.cmd`
- `pamap2/RUN_ALL.cmd`
- `synthetic/RUN_T2S_SYNTHETIC_NO_FLASH_LOGGER.cmd`
- `ucrseg/RUN_T2S_UCRSEG_NO_FLASH_LOGGER.cmd`
- `uschad/RUN_T2S_USCHAD_NO_FLASH_LOGGER.cmd`

Python runners are available as `run_t2s_*.py` in each dataset folder. Shared code is under `baseline/t2s/_shard/`.

### TICC

Path: `baseline/ticc/`

Use the dataset-level command scripts:

- `actrectut/RUN_TICC_ACTRECTUT_DIRECT_PRINT_STAY.cmd`
- `mocap/RUN_TICC_MOCAP_DIRECT_PRINT_STAY.cmd`
- `pamap2/RUN_TICC_PAMAP2_ZERO_DIRECT_PRINT_STAY.cmd`
- `synthetic/RUN_TICC_SYNTHETIC_DIRECT_PRINT_STAY.cmd`
- `ucrseg/RUN_TICC_UCRSEG_DIRECT_PRINT_STAY.cmd`
- `uschad/RUN_TICC_USCHAD_DIRECT_PRINT_STAY.cmd`

Shared TICC implementation files are under `baseline/ticc/_shard/`.

## Notes

- Prefer running from `D:\code\OpenMAIC` so relative paths in `.cmd` and config files resolve consistently.
- Some baselines require method-specific dependencies. If a script fails because a package is missing, install the package into the active Python or Conda environment and rerun the same command.
- Some result folders are large. Regenerating all datasets for all methods can take a long time.
- Do not delete existing `results_*` folders unless you intentionally want to remove reference outputs.
