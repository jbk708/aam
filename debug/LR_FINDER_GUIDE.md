# Learning Rate Finder Guide

## Overview

The Learning Rate Finder utility helps identify the optimal initial learning rate for training by performing a learning rate range test. This is particularly useful when training stagnates at local minima (e.g., around epoch 34) and you need to find a better starting learning rate.

## How It Works

The LR finder implements a learning rate range test similar to fastai's approach:

1. **Exponential LR Increase**: Starts with a very low learning rate (default: 1e-7) and exponentially increases it to a high value (default: 10.0)
2. **Loss Tracking**: Trains the model for a specified number of iterations while tracking the loss at each learning rate
3. **Optimal LR Detection**: Finds the learning rate where the loss decreases fastest (steepest negative slope)
4. **Early Stopping**: Automatically stops if loss diverges (becomes too large)

## Basic Usage

### Import and Setup

```python
from aam.training.lr_finder import LearningRateFinder
from aam.training.trainer import create_optimizer
from aam.training.losses import MultiTaskLoss
import torch

# Setup your model, optimizer, and loss function
model = ...  # Your model
optimizer = create_optimizer(model, optimizer_type="adamw", lr=1e-4)
loss_fn = MultiTaskLoss(penalty=1.0, nuc_penalty=0.0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create LR finder
lr_finder = LearningRateFinder(model, optimizer, loss_fn, device)
```

### Run LR Range Test

```python
# Run the LR finder
lrs, losses, suggested_lr = lr_finder.find_lr(
    train_loader,  # Your DataLoader
    start_lr=1e-7,  # Starting LR (default: 1e-7)
    end_lr=10.0,    # Ending LR (default: 10.0)
    num_iter=100,   # Number of iterations (default: 100)
)

print(f"Suggested learning rate: {suggested_lr:.2e}")
```

### Visualize Results

```python
# Plot the results
from pathlib import Path

output_path = Path("lr_finder_plot.png")
fig = lr_finder.plot(output_path=output_path)

# The plot shows:
# - Learning rate (x-axis, log scale)
# - Loss (y-axis)
# - Red dashed line marking the suggested optimal LR
```

## Complete Example

```python
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from aam.models.sequence_encoder import SequenceEncoder
from aam.training.lr_finder import LearningRateFinder
from aam.training.trainer import create_optimizer
from aam.training.losses import MultiTaskLoss
from aam.data.dataset import ASVDataset, collate_fn

# Load your data
table_path = "path/to/table.biom"
unifrac_matrix_path = "path/to/unifrac_matrix.npy"
train_dataset = ASVDataset(table_path, unifrac_matrix_path, split="train")
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn,
)

# Create model
model = SequenceEncoder(
    vocab_size=6,
    embedding_dim=128,
    max_bp=100,
    token_limit=256,
    encoder_type="unifrac",
    predict_nucleotides=False,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Create optimizer and loss
optimizer = create_optimizer(model, optimizer_type="adamw", lr=1e-4)
loss_fn = MultiTaskLoss(penalty=1.0, nuc_penalty=0.0)

# Run LR finder
lr_finder = LearningRateFinder(model, optimizer, loss_fn, device)

print("Running LR range test...")
lrs, losses, suggested_lr = lr_finder.find_lr(
    train_loader,
    start_lr=1e-7,
    end_lr=10.0,
    num_iter=100,
)

print(f"\nSuggested learning rate: {suggested_lr:.2e}")
print(f"Tested {len(lrs)} learning rates")

# Plot results
output_path = Path("lr_finder_plot.png")
lr_finder.plot(output_path=output_path)
print(f"\nPlot saved to {output_path}")
```

## Parameters

### `find_lr()` Parameters

- **`train_loader`** (DataLoader): Training data loader
- **`start_lr`** (float, default: 1e-7): Starting learning rate for the range test
- **`end_lr`** (float, default: 10.0): Ending learning rate for the range test
- **`num_iter`** (int, default: 100): Number of iterations to run
- **`smooth_factor`** (float, default: 0.05): Smoothing factor for loss (0-1, higher = more smoothing)
- **`diverge_threshold`** (float, default: 5.0): Stop early if loss > `diverge_threshold * best_loss`

### `plot()` Parameters

- **`output_path`** (Path, optional): Path to save the plot
- **`skip_start`** (int, default: 10): Number of initial points to skip (noisy start)
- **`skip_end`** (int, default: 5): Number of final points to skip (noisy end)

## Interpreting Results

### Good LR Range Test

A good LR range test should show:
1. **Initial Phase**: Loss decreases slowly at very low LRs
2. **Optimal Phase**: Steep decrease in loss (this is where the suggested LR is)
3. **Plateau Phase**: Loss plateaus or decreases slowly
4. **Divergence Phase**: Loss increases rapidly (LR too high)

The **suggested LR** is typically at the beginning of the optimal phase, where the loss decreases fastest.

### Example Plot Interpretation

```
Loss
  |
  |     /\
  |    /  \
  |   /    \____
  |  /          \
  | /            \___
  |/________________\___  (divergence)
  |
  +----------------------> LR (log scale)
  1e-7             10.0
        ^
        |
    Suggested LR (steepest descent)
```

## Tips

1. **Use a subset of data**: For faster results, use a subset of your training data (e.g., first 1000 samples)

2. **Adjust iteration count**: More iterations = more precise but slower. Start with 50-100 iterations.

3. **Check the plot**: Always visualize the results to verify the suggested LR makes sense

4. **Multiple runs**: If results are inconsistent, run the LR finder multiple times and average the suggested LRs

5. **Model state**: The LR finder modifies model weights. If you want to preserve the original model state, save/load checkpoints before/after

6. **LR restoration**: The LR finder automatically restores the original optimizer learning rate after completion

## Troubleshooting

### Loss Diverges Immediately

- **Problem**: Loss becomes NaN or very large right away
- **Solution**: Lower `end_lr` (e.g., 1.0 instead of 10.0) or use a smaller `num_iter`

### No Clear Optimal LR

- **Problem**: Loss curve is flat or noisy
- **Solution**: 
  - Increase `num_iter` for more data points
  - Adjust `smooth_factor` for smoother curves
  - Check that your model and data are correct

### Suggested LR Too High/Low

- **Problem**: Suggested LR doesn't work well in actual training
- **Solution**: 
  - Use the plot to manually select an LR from the optimal phase
  - Try LRs around the suggested value (e.g., 0.5x, 2x the suggested LR)
  - The suggested LR is a starting point, not a guarantee

## Integration with Training

After finding the optimal LR, use it in your training:

```python
# Use the suggested LR
optimal_lr = suggested_lr

# Create optimizer with optimal LR
optimizer = create_optimizer(model, optimizer_type="adamw", lr=optimal_lr)

# Or update existing optimizer
for param_group in optimizer.param_groups:
    param_group["lr"] = optimal_lr
```

## References

- FastAI's LR finder: https://docs.fast.ai/callback.schedule.html#LRFinder
- Leslie Smith's paper: "Cyclical Learning Rates for Training Neural Networks" (2015)
