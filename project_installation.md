# Project Installation Guide

This project uses **`uv`** for fast, reproducible Python environment management, with optional **Conda** support for users who prefer it or are working on HPC systems.

The repository includes:
- `pyproject.toml` — primary dependency specification
- `requirements.txt` — compatibility / fallback dependency list

---

## Prerequisites

Before installing, ensure you have:

- **Python ≥ 3.12**
- **uv** from [here](https://docs.astral.sh/uv/getting-started/installation/)

## Installation using uv

### 1. Load the Required Compiler (HPC/ Module Systems)
If you are on an HPC cluster or system with environment modules:
```bash
module load gcc_12.2.0
```
Ensure gcc and g++ are correctly exported:
```bash
export CC=$(which gcc)
export CXX=$(which g++)
```
You can verify with:
```bash
gcc --version
g++ --version
```
### 2. Create and Sync the uv Env
From project root, after you have created the uv environment using:
```bash
uv venv
source .venv/bin/activate
```
Run this:
```bash
uv sync --all-extras
```
This will:
	- Create a local virtual environment
	- Install all dependencies from pyproject.toml
	- Include all optional extras
	- Respect the existing lockfile (if present)

Once complete, the environment is ready to use.

## Installation using Conda
### 1. Create conda environment
```bash
conda create -n <env-name> python=3.12 -y
conda activate <env-name>
```

Run the following commands:
```bash
module load gcc_12.2.0
export CC=$(which gcc)
export CXX=$(which g++)
```
Goal is for the C and Cpp compilers to be able to execute the library code.

Next, install via requirements.txt
```bash
pip install -r requirements.txt
```

Quick test to check aam has been installed, run:
```bash
python -m aam.cli --help
# Or
python -m aam.cli pretrain --help
```