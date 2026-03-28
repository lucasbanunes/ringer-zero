# ringer-zero
Repo with code for developing the ringer model for L0 trigger in the ATLAS Experiment.

# Setup

## 1) Create and activate the environment

This repository expects Python 3.12 and PyROOT 6.36.06, which can be installed through Conda. Run the setup script:

```bash
bash setup.sh
```

## 3) Verify CLI is available

```bash
python cli.py --help
python cli.py vqat --help
python cli.py vqat run-training --help
```

# Training models from the CLI

Training is configured through a YAML file passed to the VQAT CLI command.

## 1) Prepare a training config file

Create a file (for example, `configs/vqat_training_job.yaml`) with the following structure:

```yaml
dataset_dir: /path/to/parquet_dataset_dir
data_table: data
rings_col: trig_L2_calo_rings
kfold_table: standard_binning_kfold
label_col: label
fold_col: fold
et_col: trig_L2_calo_et
et_bins:
  - 15000.0
  - 20000.0
  - 30000.0
  - 40000.0
  - 50000.0
  - .inf
eta_col: trig_L2_calo_eta
eta_bins:
  - 0.0
  - 0.8
  - 1.37
  - 1.54
  - 2.37
  - 2.5
tag: vqat
b0: 22
i0: 7
batch_size: 1024
inits: 5
dry_run: false
executor_config:
  cpus_per_task: 1
  executor_type: debug
  logs_dir: ./logs
  name: vqat_training
  slurm_array_parallelism: 1
  slurm_partition: null
  stderr_to_stdout: true
  timeout_min: 60
```

Notes:
- `dataset_dir` must contain the parquet tables used by `data_table`, `kfold_table`, and `ref`.
- `et_bins` and `eta_bins` define bin edges. Training runs for each adjacent bin interval.
- `executor_config.executor_type` controls execution backend (for local/debug usage, keep `debug`).
- `dry_run: true` submits only the first bin combination and is useful to validate configuration.

If you want to run a job but do not know all available configuration parameters, print the full configuration schema directly in the CLI help. This command shows the complete `VQATTrainingJob` schema (including nested fields such as `executor_config`) so you can build your YAML file with all supported options.

```bash
python cli.py vqat run-training --help
```

## 2) Run training

```bash
python cli.py vqat run-training --config configs/vqat_training_job.yaml
```

## Useful CLI extras

To inspect dataset schemas before configuring training:

```bash
python cli.py datasets print-schema --dataset-dir /path/to/parquet_dataset_dir
```
