# Validation Prediction Plots

**Status:** ✅ Completed

## Objective
Create validation prediction plots showing predicted vs actual values with linear fit, R² metric, and 1:1 reference line. Save plots to both TensorBoard and disk files.

## Requirements

### Plot Features
- **Scatter plot**: Predicted vs actual values
- **Linear fit**: Regression line through data points
- **R² metric**: Displayed in title or legend
- **1:1 reference line**: Perfect prediction line (y=x) for comparison
- **Target predictions only**: Focus on main prediction task (not count/base)

### Output Formats
- **TensorBoard**: Log plot using `writer.add_figure()` 
- **Disk files**: Save as PNG files to `{output_dir}/plots/` directory
- **Naming**: `pred_vs_actual_epoch_{epoch}.png` or `pred_vs_actual_best.png`

### Classification Support
- **Confusion matrix**: For classification tasks, create confusion matrix instead of scatter plot
- **Metrics**: Display accuracy, precision, recall, F1 in plot

### Triggering
- **Plot when validation improves**: Only create plots when validation loss improves (new best model)
- **Configurable**: Add flag to enable/disable plot generation

### Library
- Use `matplotlib` or `seaborn` for plotting
- Add as dependency if not already present

## Implementation Requirements

### New Dependencies
- Add `matplotlib` to `pyproject.toml` dependencies (or `seaborn` if preferred)
- Add to `environment.yml` if using conda

### Changes to `trainer.py`

**New Method**: `_create_prediction_plot()`
- Create scatter plot with predicted vs actual
- Add linear fit line (use `numpy.polyfit` or `scipy.stats.linregress`)
- Calculate R² (already computed in metrics)
- Add 1:1 reference line
- Format plot with labels, title, legend

**New Method**: `_create_confusion_matrix_plot()` (for classification)
- Create confusion matrix heatmap
- Display accuracy, precision, recall, F1
- Format with labels and title

**Modify**: `train()` method
- Call plot creation when validation improves
- Save plot to disk
- Log plot to TensorBoard
- Only for target predictions (not count/base)

**New Parameter**: `save_plots: bool = True`
- Add to `train()` method signature
- Allow disabling plot generation

### Plot Directory
- Create `{checkpoint_dir}/../plots/` or `{output_dir}/plots/` directory
- Save plots with descriptive filenames

## Implementation Checklist

- [x] Add matplotlib/seaborn dependency
- [x] Create `_create_prediction_plot()` method for regression
- [x] Create `_create_confusion_matrix_plot()` method for classification
- [x] Integrate plot creation into `train()` method (when validation improves)
- [x] Save plots to disk as PNG files
- [x] Log plots to TensorBoard using `add_figure()`
- [x] Add `save_plots` parameter to `train()` method
- [x] Create plots directory automatically
- [x] Test with regression tasks
- [x] Test with classification tasks
- [x] Test plot saving and TensorBoard logging
- [x] Update CLI to support `--save-plots` flag (optional - implemented as `save_plots` parameter in `train()` method)

## Key Considerations

### Plot Quality
- **Figure size**: Use appropriate size (e.g., 8x6 inches)
- **DPI**: Use 100-150 DPI for good quality
- **Labels**: Clear axis labels ("Predicted", "Actual")
- **Title**: Include epoch number and R² value
- **Legend**: Show linear fit equation or R² value

### Performance
- **Plot only on improvement**: Reduces overhead
- **Close figures**: Use `plt.close()` after saving to free memory
- **Optional**: Make plot generation optional via flag

### Classification
- **Confusion matrix**: Use `sklearn.metrics.confusion_matrix`
- **Heatmap**: Use `seaborn.heatmap` or `matplotlib.imshow`
- **Metrics display**: Show accuracy, precision, recall, F1 in title or text box

### File Management
- **Naming**: `pred_vs_actual_epoch_{epoch}.png` for each improvement
- **Or**: `pred_vs_actual_best.png` (replace on improvement)
- **Directory**: `{output_dir}/plots/` or `{checkpoint_dir}/../plots/`

## Testing Requirements

- Test plot creation for regression tasks
- Test plot creation for classification tasks
- Test plot saving to disk
- Test TensorBoard logging of plots
- Test that plots are only created when validation improves
- Test with `save_plots=False` to disable plotting
- Verify plots include all required elements (scatter, fit, R², 1:1 line)

## Example Plot Structure

### Regression Plot
```
Title: "Predicted vs Actual (Epoch {epoch}, R² = {r2:.4f})"
X-axis: "Actual"
Y-axis: "Predicted"
- Scatter points (predicted vs actual)
- Linear fit line (blue, labeled "Linear Fit, R² = {r2:.4f}")
- 1:1 reference line (dashed gray, labeled "Perfect Prediction")
- Legend with R² value
```

### Classification Plot
```
Title: "Confusion Matrix (Epoch {epoch})"
- Heatmap of confusion matrix
- Text box with: Accuracy, Precision, Recall, F1 scores
- Class labels on axes
```

## Notes

- Plotting adds minimal overhead when only done on improvement
- Both TensorBoard and disk saves provide flexibility
- Matplotlib is standard and lightweight
- Seaborn provides nicer styling but adds dependency
- Consider using matplotlib with seaborn style: `plt.style.use('seaborn-v0_8')`
