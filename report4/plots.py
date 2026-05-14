import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# Load data from main.py
# ============================================================
data = np.load('report4/kmeans_data.npz', allow_pickle=True)
points = data['points']
point_ids = data['point_ids']
assignments = data['assignments']
centroids = data['centroids']
initial_seeds = data['initial_seeds']

colors = ['#e74c3c', '#2ecc71', '#3498db']
cluster_labels = ['Cluster 1', 'Cluster 2', 'Cluster 3']

fig, ax = plt.subplots(figsize=(10, 8))

# Plot points colored by cluster
for c_idx in range(3):
    mask = assignments == c_idx
    cluster_pts = points[mask]
    ids = point_ids[mask]
    ax.scatter(cluster_pts[:, 0], cluster_pts[:, 1],
               c=colors[c_idx], s=150, label=cluster_labels[c_idx],
               edgecolors='black', linewidths=1.2, zorder=3)
    for pt, pid in zip(cluster_pts, ids):
        ax.annotate(str(pid), (pt[0], pt[1]),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=12, fontweight='bold')

# Plot initial seeds (dashed outline)
ax.scatter(initial_seeds[:, 0], initial_seeds[:, 1],
           c='none', s=250, edgecolors='gray', linewidths=2,
           linestyles='dashed', label='Initial seeds', zorder=2)

# Plot final centroids (star marker)
for c_idx in range(3):
    ax.scatter(centroids[c_idx, 0], centroids[c_idx, 1],
               marker='*', s=400, c=colors[c_idx], edgecolors='black',
               linewidths=1.2, zorder=4)
    ax.annotate(f'centroid', (centroids[c_idx, 0], centroids[c_idx, 1]),
                textcoords="offset points", xytext=(8, -12),
                fontsize=9, fontstyle='italic', color=colors[c_idx])

ax.set_xlabel('x', fontsize=13)
ax.set_ylabel('y', fontsize=13)
ax.set_title('K-Means Clustering Result (k=3)', fontsize=15)
ax.legend(fontsize=11, loc='upper left')
ax.set_xlim(-0.5, 9.5)
ax.set_ylim(0.5, 11)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_xticks(range(0, 10))
ax.set_yticks(range(0, 12))

fig.tight_layout()
fig.savefig('report4/kmeans_result.png', dpi=150)
plt.close(fig)
print("kmeans_result.png saved")
