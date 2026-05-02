import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 加载主程序保存的数据
# ============================================================
data = np.load('report3/monotonicity_data.npz')
ave_rooms_range = data['ave_rooms_range']
pred_baseline = data['pred_baseline']
pred_constrained = data['pred_constrained']
mvr_baseline = data['mvr_baseline']
mvr_constrained = data['mvr_constrained']
y_test = data['y_test']
y_pred_baseline = data['y_pred_baseline']
y_pred_constrained = data['y_pred_constrained']

# ============================================================
# 图1：单调性对比曲线 (核心图表)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ave_rooms_range, pred_baseline, 'r-', linewidth=1.5,
        label=f'Ordinary RF (MVR={mvr_baseline:.2%})', alpha=0.8)
ax.plot(ave_rooms_range, pred_constrained, 'b-', linewidth=2,
        label=f'Monotonicity-Constrained RF (MVR={mvr_constrained:.2%})')
ax.set_xlabel('AveRooms', fontsize=12)
ax.set_ylabel('Predicted MedHouseVal ($100k)', fontsize=12)
ax.set_title('Monotonicity Comparison: Predicted Price vs AveRooms', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('report3/fig1_monotonicity_comparison.png', dpi=150)
plt.close(fig)
print("fig1_monotonicity_comparison.png saved")

# ============================================================
# 图2：预测 vs 真实散点图
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, y_pred, title in zip(
    axes,
    [y_pred_baseline, y_pred_constrained],
    ['Ordinary RF: Predicted vs Actual', 'Monotonicity-Constrained RF: Predicted vs Actual']
):
    ax.scatter(y_test, y_pred, alpha=0.3, s=8, edgecolors='none')
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Actual MedHouseVal ($100k)', fontsize=11)
    ax.set_ylabel('Predicted MedHouseVal ($100k)', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig('report3/fig2_pred_vs_actual.png', dpi=150)
plt.close(fig)
print("fig2_pred_vs_actual.png saved")

# ============================================================
# 图3：残差分布对比
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

residuals_baseline = y_test - y_pred_baseline
residuals_constrained = y_test - y_pred_constrained

for ax, res, title in zip(
    axes,
    [residuals_baseline, residuals_constrained],
    ['Ordinary RF Residuals', 'Monotonicity-Constrained RF Residuals']
):
    ax.hist(res, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Residual ($100k)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig('report3/fig3_residual_distribution.png', dpi=150)
plt.close(fig)
print("fig3_residual_distribution.png saved")

# ============================================================
# 图4：单调性局部放大对比（AveRooms 2-6 区间）
# ============================================================
zoom_mask = (ave_rooms_range >= 2) & (ave_rooms_range <= 6)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ave_rooms_range[zoom_mask], pred_baseline[zoom_mask], 'r-o',
        linewidth=1.5, markersize=5, label=f'Ordinary RF (MVR={mvr_baseline:.2%})', alpha=0.8)
ax.plot(ave_rooms_range[zoom_mask], pred_constrained[zoom_mask], 'b-s',
        linewidth=2, markersize=5, label=f'Constrained RF (MVR={mvr_constrained:.2%})')
ax.set_xlabel('AveRooms', fontsize=12)
ax.set_ylabel('Predicted MedHouseVal ($100k)', fontsize=12)
ax.set_title('Zoomed View (AveRooms 2-6): Monotonicity Detail', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('report3/fig4_monotonicity_zoom.png', dpi=150)
plt.close(fig)
print("fig4_monotonicity_zoom.png saved")

print("\nAll figures saved to report3/")
