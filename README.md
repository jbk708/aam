# Attention All Microbes (AAM)

Deep Learning Method for Microbial Sequencing Data using PyTorch

## Installation

### Prerequisites

- [Mamba](https://mamba.readthedocs.io/) or [Conda](https://docs.conda.io/) (Mamba is recommended for faster package resolution)

### Create Environment

Create the conda/mamba environment from the `environment.yml` file:

```bash
mamba env create -f environment.yml
```

Or with conda:

```bash
conda env create -f environment.yml
```

### Activate Environment

Activate the environment:

```bash
mamba activate aam
```

Or with conda:

```bash
conda activate aam
```

### Install Package

Install the package in editable mode:

```bash
pip install -e .
```

For development with all optional dependencies:

```bash
pip install -e ".[dev,docs,training]"
```

Or install specific optional dependencies:

```bash
# Development dependencies
pip install -e ".[dev]"

# Documentation dependencies
pip install -e ".[docs]"

# Training dependencies
pip install -e ".[training]"
```





