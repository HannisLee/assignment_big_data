import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ============================================================
# Load data from main.py
# ============================================================
data = np.load('outlier_data.npz', allow_pickle=True)
points = data['points']
point_ids = data['point_ids']
labels = data['labels']
eps = float(data['eps'])
minpts = int(data['minpts'])
num_clusters = int(data['num_clusters'])
cluster_sizes = data['cluster_sizes']
scores = data['scores']
nearest_other = data['nearest_other']
nearest_other_dist = data['nearest_other_dist']

n = len(points)

# ============================================================
# Color setup
# ============================================================
cluster_colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6', '#1abc9c', '#e74c3c']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# ============================================================
# Subplot 1: Scatter plot with clusters and eps circles
# ============================================================
for i in range(n):
    x, y = points[i]
    c_idx = labels[i]
    color = cluster_colors[c_idx % len(cluster_colors)]
    is_outlier = scores[i] > 2

    # Draw epsilon circle (Euclidean)
    circle = Circle((x, y), eps, fill=False, edgecolor='gray',
                     linestyle='--', alpha=0.25, linewidth=1)
    ax1.add_patch(circle)

    if is_outlier:
        ax1.scatter(x, y, c='red', s=220, marker='X', zorder=5,
                    edgecolors='darkred', linewidths=1.5)
    else:
        ax1.scatter(x, y, c=color, s=180, edgecolors='black',
                    linewidths=1.0, zorder=4)

    # Annotation
    ann = f'{point_ids[i]}'
    if is_outlier:
        ann += ' (outlier)'
    else:
        ann += f' (C{c_idx + 1})'
    ax1.annotate(ann, (x, y),
                 textcoords="offset points", xytext=(8, 8),
                 fontsize=9, fontweight='bold')

# Draw lines from outlier to its nearest neighbor in another cluster
for i in range(n):
    if scores[i] > 2:
        o_idx = i
        o_prime_name = nearest_other[i]
        o_prime_idx = list(point_ids).index(o_prime_name)
        ax1.plot([points[o_idx, 0], points[o_prime_idx, 0]],
                 [points[o_idx, 1], points[o_prime_idx, 1]],
                 'r--', alpha=0.6, linewidth=1.5, zorder=2)

ax1.set_xlabel('x', fontsize=13)
ax1.set_ylabel('y', fontsize=13)
ax1.set_title(f'Outlier Detection via DBSCAN (Euclidean, ε={eps}, MinPts={minpts})', fontsize=13)
ax1.set_xlim(-2, 10)
ax1.set_ylim(-2, 5)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(-2, 11))
ax1.set_yticks(range(-2, 6))

# ============================================================
# Subplot 2: Bar chart of outlier scores
# ============================================================
colors_bar = ['#e74c3c' if scores[i] > 2 else '#3498db' for i in range(n)]
bars = ax2.bar(range(n), scores, color=colors_bar, edgecolor='black', linewidth=0.8)

ax2.axhline(y=2.0, color='red', linestyle='--', linewidth=2, label='Threshold (score = 2)')

ax2.set_xticks(range(n))
ax2.set_xticklabels(point_ids, fontsize=11, fontweight='bold')
ax2.set_xlabel('Point', fontsize=13)
ax2.set_ylabel('Outlier Score', fontsize=13)
ax2.set_title('Outlier Scores for Each Object', fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, axis='y', alpha=0.3)
ax2.set_ylim(0, max(scores) + 0.5)

# Annotate score values on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax2.annotate(f'{height:.3f}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 4), textcoords="offset points",
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

fig.tight_layout()
fig.savefig('outlier_result.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("outlier_result.png saved")
