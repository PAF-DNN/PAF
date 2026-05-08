# PAF

Probabilistic Activation Flow experiments, model utilities, and evaluation code.

## Project Layout

```text
src/        Python source packages
config/     YAML configuration files
scripts/    Setup and utility scripts
```

## Setup

Install the project dependencies into a local `.venv`:

```bash
./scripts/install_requirements.sh
```

The script expects Python 3.11 to be available as `python3.11`. If another Python executable should be used, override it:

```bash
PYTHON_BIN=python3 ./scripts/install_requirements.sh
```

On Windows, run the script from Git Bash, WSL, or another Bash-compatible shell.

## Running

After setup:

```bash
source .venv/bin/activate
MPLCONFIGDIR=.cache/matplotlib PYTHONPATH=src python -m Evaluation.paf_main
```

The Python `graphviz` package is installed by `requirements.txt`, but some visualizations also need the system Graphviz executable. On macOS:

```bash
brew install graphviz
```
