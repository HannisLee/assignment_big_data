import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

# ============================================================
# Load data from main.py
# ============================================================
data = np.load('report5/dbscan_data.npz', allow_pickle=True)
points = data['points']
point_ids = data['point_ids']
labels = data['labels']
eps = int(data['eps'])
minpts = int(data['minpts'])
neighborhoods = data['neighborhoods']

n = len(points)

# ============================================================
# Classify points
# ============================================================
cluster_color = '#2ecc71'  # green for cluster
noise_color = '#95a5a6'    # gray for noise

core_indices = []
border_indices = []
noise_indices = []

for i in range(n):
    if labels[i] < 0:
        noise_indices.append(i)
    elif len(neighborhoods[i]) >= minpts:
        core_indices.append(i)
    else:
        border_indices.append(i)

fig, ax = plt.subplots(figsize=(12, 6))

# Draw Manhattan epsilon diamonds (L1 ball) for each point
for i in range(n):
    x, y = points[i]
    diamond = MplPolygon(
        [[x, y + eps], [x + eps, y], [x, y - eps], [x - eps, y]],
        fill=False, edgecolor='gray', linestyle='--', alpha=0.3, linewidth=1
    )
    ax.add_patch(diamond)

# Plot cluster core points
if core_indices:
    core_pts = points[core_indices]
    ax.scatter(core_pts[:, 0], core_pts[:, 1],
               c=cluster_color, s=180, label='Core point',
               edgecolors='black', linewidths=1.2, zorder=3)

# Plot cluster border points
if border_indices:
    border_pts = points[border_indices]
    ax.scatter(border_pts[:, 0], border_pts[:, 1],
               c=cluster_color, s=180, label='Border point',
               edgecolors='#e67e22', linewidths=2.5, zorder=3)

# Plot noise points
if noise_indices:
    noise_pts = points[noise_indices]
    ax.scatter(noise_pts[:, 0], noise_pts[:, 1],
               c=noise_color, s=180, marker='X', label='Noise',
               edgecolors='black', linewidths=1.2, zorder=3)

# Annotate each point with its ID and type
for i in range(n):
    x, y = points[i]
    pid = point_ids[i]
    if labels[i] < 0:
        ann = f'{pid} (noise)'
    elif len(neighborhoods[i]) >= minpts:
        ann = f'{pid} (core)'
    else:
        ann = f'{pid} (border)'
    ax.annotate(ann, (x, y),
                textcoords="offset points", xytext=(8, 8),
                fontsize=10, fontweight='bold')

ax.set_xlabel('x', fontsize=13)
ax.set_ylabel('y', fontsize=13)
ax.set_title(f'DBSCAN Clustering Result (Manhattan, eps={eps}, MinPts={minpts})', fontsize=15)
ax.legend(fontsize=11, loc='upper left')
ax.set_xlim(-3, 11)
ax.set_ylim(-3, 5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_xticks(range(-2, 12))
ax.set_yticks(range(-3, 6))

fig.tight_layout()
fig.savefig('report5/dbscan_result.png', dpi=150)
plt.close(fig)
print("dbscan_result.png saved")
