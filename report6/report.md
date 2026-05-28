# L6 Assignment

**Name:** LI HAN
**Student ID:** 33C26029
**GitHub Repository:** https://github.com/HannisLee/assignment_big_data

## Problem Statement

Consider the following eight objects. Our task is to detect outliers using clustering-based method.

**Data:**

| ID | (x, y) |
|----|--------|
| A  | (0, 0) |
| B  | (1, 0) |
| C  | (2, 1) |
| D  | (3, 0) |
| E  | (4, 2) |
| F  | (5, 0) |
| G  | (6, 0) |
| H  | (8, 0) |

**Conditions:**
- Distance measure: Euclidean distance (L2 norm)
- DBSCAN parameters: ε = 1.5, MinPts = 2
- Processing order: A → B → C → D → E → F → G → H
- Unclustered objects are treated as single-object clusters

**Outlier scoring:**

$$\text{score}(o) = \frac{\text{dist}(o,\, o')}{|c|}$$

- $o'$: nearest object to $o$ in **another cluster**
- $|c|$: number of objects in $o$'s cluster
- If score > 2 → outlier

## Algorithm

This problem involves two phases:

**Phase 1 — DBSCAN Clustering:**

1. For each unvisited point p (in alphabetical order):
   - Mark p as visited
   - Retrieve ε-neighborhood: N_eps(p) = {q | dist(p, q) ≤ ε}
   - If |N_eps(p)| < MinPts: mark as noise (may later become border point)
   - If |N_eps(p)| ≥ MinPts: p is a **core point** — create a new cluster and expand
2. Cluster expansion: assign reachable points to the cluster; if they are also core points, continue expanding recursively
3. Noise points are treated as single-object clusters for scoring

**Phase 2 — Outlier Score Computation:**

For each object o, find the nearest object o' in a **different** cluster, then compute score(o) = dist(o, o') / |c|. Objects with score > 2 are outliers.

## Step-by-Step Calculation

### 1. Euclidean Distance Matrix

Euclidean distance: $d(p, q) = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}$

|      |    A |    B |    C |    D |    E |    F |    G |    H |
|------|------|------|------|------|------|------|------|------|
| **A** | 0    | 1    | 2.236| 3    | 4.472| 5    | 6    | 8    |
| **B** | 1    | 0    | 1.414| 2    | 3.606| 4    | 5    | 7    |
| **C** | 2.236| 1.414| 0    | 1.414| 2.236| 3.162| 4.123| 6.083|
| **D** | 3    | 2    | 1.414| 0    | 2.236| 2    | 3    | 5    |
| **E** | 4.472| 3.606| 2.236| 2.236| 0    | 2.236| 2.828| 4.472|
| **F** | 5    | 4    | 3.162| 2    | 2.236| 0    | 1    | 3    |
| **G** | 6    | 5    | 4.123| 3    | 2.828| 1    | 0    | 2    |
| **H** | 8    | 7    | 6.083| 5    | 4.472| 3    | 2    | 0    |

### 2. Epsilon Neighborhoods (ε = 1.5)

| Point | N_eps(point)        | \|N_eps\| | Type     |
|:-----:|:--------------------|:--------:|:---------|
| **A** | {A, B}              | 2        | **Core** |
| **B** | {A, B, C}           | 3        | **Core** |
| **C** | {B, C, D}           | 3        | **Core** |
| **D** | {C, D}              | 2        | **Core** |
| **E** | {E}                 | 1        | Non-core |
| **F** | {F, G}              | 2        | **Core** |
| **G** | {F, G}              | 2        | **Core** |
| **H** | {H}                 | 1        | Non-core |

### 3. DBSCAN Execution

#### Processing A(0,0)
- N_eps(A) = {A, B}, |N_eps| = 2
- 2 ≥ MinPts = 2 → A is **CORE**
- Create **Cluster 1**, assign A to Cluster 1
- **Expand Cluster 1 from A:**
  - **B(1,0)**: assigned to Cluster 1
    - B is CORE (|N_eps| = 3), expanding...
  - **C(2,1)**: assigned to Cluster 1
    - C is CORE (|N_eps| = 3), expanding...
  - **D(3,0)**: assigned to Cluster 1
    - D is CORE (|N_eps| = 2), expanding...

#### Processing B(1,0)
- Already visited → Cluster 1

#### Processing C(2,1)
- Already visited → Cluster 1

#### Processing D(3,0)
- Already visited → Cluster 1

#### Processing E(4,2)
- N_eps(E) = {E}, |N_eps| = 1
- 1 < MinPts = 2 → E is marked as **NOISE**

#### Processing F(5,0)
- N_eps(F) = {F, G}, |N_eps| = 2
- 2 ≥ MinPts = 2 → F is **CORE**
- Create **Cluster 2**, assign F to Cluster 2
- **Expand Cluster 2 from F:**
  - **G(6,0)**: assigned to Cluster 2
    - G is CORE (|N_eps| = 2), expanding...

#### Processing G(6,0)
- Already visited → Cluster 2

#### Processing H(8,0)
- N_eps(H) = {H}, |N_eps| = 1
- 1 < MinPts = 2 → H is marked as **NOISE**

### 4. Treat Noise as Single-Object Clusters

| Original | Cluster |
|----------|---------|
| {A, B, C, D} | Cluster 1 (|c| = 4) |
| {F, G} | Cluster 2 (|c| = 2) |
| {E} (noise) | Cluster 3 (|c| = 1) |
| {H} (noise) | Cluster 4 (|c| = 1) |

### 5. Outlier Score Computation

For each object o, find the nearest object o' in **another** cluster, then compute score(o) = dist(o, o') / |c|:

**A (Cluster 1, |c| = 4):**
- Nearest in other clusters: E (C3) at √20 ≈ 4.472
- score(A) = 4.472 / 4 = **1.118**

**B (Cluster 1, |c| = 4):**
- Nearest in other clusters: E (C3) at √13 ≈ 3.606
- score(B) = 3.606 / 4 = **0.901**

**C (Cluster 1, |c| = 4):**
- Nearest in other clusters: E (C3) at √5 ≈ 2.236
- score(C) = 2.236 / 4 = **0.559**

**D (Cluster 1, |c| = 4):**
- Nearest in other clusters: F (C2) at 2.000
- score(D) = 2.000 / 4 = **0.500**

**E (Cluster 3, |c| = 1):**
- Nearest in other clusters: C (C1) at √5 ≈ 2.236
- score(E) = 2.236 / 1 = **2.236 > 2 → OUTLIER**

**F (Cluster 2, |c| = 2):**
- Nearest in other clusters: D (C1) at 2.000
- score(F) = 2.000 / 2 = **1.000**

**G (Cluster 2, |c| = 2):**
- Nearest in other clusters: H (C4) at 2.000
- score(G) = 2.000 / 2 = **1.000**

**H (Cluster 4, |c| = 1):**
- Nearest in other clusters: G (C2) at 2.000
- score(H) = 2.000 / 1 = **2.000** (not > 2, not an outlier)

## Final Result

| Point | Cluster | \|c\| | o'  | dist(o, o') | score | Outlier? |
|-------|---------|------|-----|-------------|-------|----------|
| A     | C1      | 4    | E   | 4.472       | 1.118 | No       |
| B     | C1      | 4    | E   | 3.606       | 0.901 | No       |
| C     | C1      | 4    | E   | 2.236       | 0.559 | No       |
| D     | C1      | 4    | F   | 2.000       | 0.500 | No       |
| E     | C3      | 1    | C   | 2.236       | 2.236 | **YES**  |
| F     | C2      | 2    | D   | 2.000       | 1.000 | No       |
| G     | C2      | 2    | H   | 2.000       | 1.000 | No       |
| H     | C4      | 1    | G   | 2.000       | 2.000 | No       |

**Detected outlier: E(4, 2)**

E is the only object whose score (2.236) exceeds the threshold of 2. Note that H has a score of exactly 2.0, which does **not** exceed the threshold (score must be strictly greater than 2).

![Outlier Detection Result](outlier_result.png)

## Code

main.py

```python
import sys
import io
import numpy as np
from collections import deque

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============================================================
# Data
# ============================================================
point_ids = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
points = np.array([
    [0, 0],  # A
    [1, 0],  # B
    [2, 1],  # C
    [3, 0],  # D
    [4, 2],  # E
    [5, 0],  # F
    [6, 0],  # G
    [8, 0],  # H
])

eps = 1.5
minpts = 2


def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


# ============================================================
# Print header
# ============================================================
print("=" * 70)
print("L6: Clustering-Based Outlier Detection")
print("=" * 70)
print(f"Points: {', '.join(f'{pid}({p[0]},{p[1]})' for pid, p in zip(point_ids, points))}")
print(f"Distance: Euclidean (L2)")
print(f"eps = {eps}, MinPts = {minpts}")
print(f"Processing order: {' → '.join(point_ids)}")
print()

# ============================================================
# Distance matrix
# ============================================================
n = len(points)
dist_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        dist_matrix[i, j] = euclidean(points[i], points[j])

print("=" * 70)
print("Euclidean Distance Matrix")
print("=" * 70)
header = f"{'':>8}"
for pid in point_ids:
    header += f"{pid:>8}"
print(header)
print("-" * (8 + 8 * n))
for i in range(n):
    row = f"{point_ids[i]:>8}"
    for j in range(n):
        row += f"{dist_matrix[i, j]:>8.3f}"
    print(row)
print()

# ============================================================
# Epsilon neighborhoods
# ============================================================
print("=" * 70)
print(f"Epsilon Neighborhoods (eps = {eps})")
print("=" * 70)
neighborhoods = []
for i in range(n):
    neighbors = [j for j in range(n) if dist_matrix[i, j] <= eps]
    neighborhoods.append(neighbors)
    neighbor_ids = [point_ids[j] for j in neighbors]
    is_core = "Core" if len(neighbors) >= minpts else "Non-core"
    print(f"  N_eps({point_ids[i]}) = {{{', '.join(neighbor_ids)}}}, |N_eps| = {len(neighbors)} → {is_core}")
print()

# ============================================================
# Core point summary
# ============================================================
print("=" * 70)
print("Core Point Identification (|N_eps| >= MinPts)")
print("=" * 70)
print(f"{'Point':<8} {'|N_eps|':<10} {'MinPts':<10} {'Type':<12}")
print("-" * 40)
for i in range(n):
    ptype = "Core" if len(neighborhoods[i]) >= minpts else "Non-core"
    print(f"{point_ids[i]:<8} {len(neighborhoods[i]):<10} {minpts:<10} {ptype:<12}")
print()

# ============================================================
# DBSCAN Algorithm
# ============================================================
print("=" * 70)
print("DBSCAN Algorithm Execution")
print("=" * 70)
print()

UNDEFINED = -1
NOISE = -2

labels = np.full(n, UNDEFINED, dtype=int)
visited = set()
cluster_id = -1


def expand_cluster(p_idx, cluster_id):
    queue = deque()
    for nb in neighborhoods[p_idx]:
        if labels[nb] == UNDEFINED or labels[nb] == NOISE:
            queue.append(nb)

    while queue:
        q_idx = queue.popleft()

        if labels[q_idx] == NOISE:
            labels[q_idx] = cluster_id
            print(f"      {point_ids[q_idx]}({points[q_idx][0]},{points[q_idx][1]}) was NOISE → now assigned to Cluster {cluster_id + 1} (border point)")

        if labels[q_idx] != UNDEFINED:
            continue

        labels[q_idx] = cluster_id
        if q_idx not in visited:
            visited.add(q_idx)
            print(f"      {point_ids[q_idx]}({points[q_idx][0]},{points[q_idx][1]}) → assigned to Cluster {cluster_id + 1} (visited)")

        q_neighbors = neighborhoods[q_idx]
        if len(q_neighbors) >= minpts:
            print(f"      {point_ids[q_idx]} is CORE (|N_eps| = {len(q_neighbors)}), expanding cluster...")
            for nb in q_neighbors:
                if labels[nb] == UNDEFINED or labels[nb] == NOISE:
                    queue.append(nb)


# Process each point in alphabetical order
for i in range(n):
    pid = point_ids[i]
    px, py = points[i]

    print(f"Processing point {pid}({px},{py})")

    if i in visited:
        current_label = labels[i]
        if current_label == NOISE:
            print(f"  {pid} already visited → NOISE")
        elif current_label >= 0:
            print(f"  {pid} already visited → Cluster {current_label + 1}")
        print()
        continue

    visited.add(i)
    neighbors = neighborhoods[i]
    neighbor_ids_str = ', '.join(point_ids[j] for j in neighbors)

    print(f"  N_eps({pid}) = {{{neighbor_ids_str}}}, |N_eps| = {len(neighbors)}")

    if len(neighbors) < minpts:
        labels[i] = NOISE
        print(f"  |N_eps| = {len(neighbors)} < MinPts = {minpts} → {pid} marked as NOISE")
        print()
        continue

    cluster_id += 1
    labels[i] = cluster_id
    print(f"  |N_eps| = {len(neighbors)} >= MinPts = {minpts} → {pid} is CORE point")
    print(f"  Create Cluster {cluster_id + 1}, assigning {pid} to Cluster {cluster_id + 1}")
    print(f"  Expanding Cluster {cluster_id + 1} from {pid}...")

    expand_cluster(i, cluster_id)
    print()

# ============================================================
# DBSCAN Result
# ============================================================
print()
print("=" * 70)
print("DBSCAN CLUSTERING RESULT")
print("=" * 70)
print()

num_clusters = cluster_id + 1
print(f"Number of clusters: {num_clusters}")
print()

for c in range(num_clusters):
    members = [point_ids[i] for i in range(n) if labels[i] == c]
    core_pts = [point_ids[i] for i in range(n) if labels[i] == c and len(neighborhoods[i]) >= minpts]
    border_pts = [point_ids[i] for i in range(n) if labels[i] == c and len(neighborhoods[i]) < minpts]
    print(f"  Cluster {c + 1}: {{{', '.join(members)}}}")
    print(f"    Core points: {{{', '.join(core_pts)}}}")
    print(f"    Border points: {{{', '.join(border_pts)}}}")

noise_pts = [point_ids[i] for i in range(n) if labels[i] == NOISE]
print(f"  Noise: {{{', '.join(noise_pts)}}}")
print()

# ============================================================
# Treat noise as single-object clusters
# ============================================================
print("=" * 70)
print("Treat Unclustered Objects as Single-Object Clusters")
print("=" * 70)
print()

final_labels = labels.copy()
final_cluster_id = cluster_id

for i in range(n):
    if final_labels[i] == NOISE:
        final_cluster_id += 1
        final_labels[i] = final_cluster_id
        print(f"  {point_ids[i]}({points[i][0]},{points[i][1]}) (noise) → Cluster {final_cluster_id + 1} (single-object cluster)")

total_clusters = final_cluster_id + 1
print()
print(f"Total clusters after treatment: {total_clusters}")
print()

for c in range(total_clusters):
    members = [point_ids[i] for i in range(n) if final_labels[i] == c]
    print(f"  Cluster {c + 1}: {{{', '.join(members)}}} (|c| = {len(members)})")
print()

# ============================================================
# Outlier Score Computation
# ============================================================
print("=" * 70)
print("Outlier Score Computation")
print("score(o) = dist(o, o') / |c|")
print("where o' is the nearest object in ANOTHER cluster")
print("=" * 70)
print()

cluster_sizes = np.zeros(total_clusters, dtype=int)
for c in range(total_clusters):
    cluster_sizes[c] = np.sum(final_labels == c)

scores = np.zeros(n)
nearest_other = [''] * n
nearest_other_dist = np.zeros(n)

for i in range(n):
    my_cluster = final_labels[i]
    my_size = cluster_sizes[my_cluster]
    min_dist = float('inf')
    nearest_j = -1

    for j in range(n):
        if final_labels[j] != my_cluster:
            if dist_matrix[i, j] < min_dist:
                min_dist = dist_matrix[i, j]
                nearest_j = j

    scores[i] = min_dist / my_size
    nearest_other[i] = point_ids[nearest_j]
    nearest_other_dist[i] = min_dist

    is_outlier = "YES ← OUTLIER" if scores[i] > 2 else "No"
    print(f"  {point_ids[i]}: cluster {my_cluster + 1}, |c| = {my_size}, "
          f"o' = {nearest_other[i]}, dist = {min_dist:.3f}, "
          f"score = {scores[i]:.3f} → {is_outlier}")

print()

# ============================================================
# Final Outlier Detection Result
# ============================================================
print("=" * 70)
print("OUTLIER DETECTION RESULT")
print("=" * 70)
print()

header = f"{'Point':<8} {'Cluster':<10} {'|c|':<6} {'o_oth':<6} {'dist':<12} {'score':<10} {'Outlier?':<10}"
print(header)
print("-" * 62)
for i in range(n):
    is_outlier = "YES" if scores[i] > 2 else "No"
    print(f"{point_ids[i]:<8} C{final_labels[i]+1:<9} {cluster_sizes[final_labels[i]]:<6} "
          f"{nearest_other[i]:<6} {nearest_other_dist[i]:<12.3f} {scores[i]:<10.3f} {is_outlier:<10}")

print()
outliers = [point_ids[i] for i in range(n) if scores[i] > 2]
print(f"Detected outliers: {{{', '.join(outliers)}}}")
print(f"Threshold: score > 2")

# ============================================================
# Save data for plotting
# ============================================================
np.savez('outlier_data.npz',
         points=points,
         point_ids=np.array(point_ids),
         labels=final_labels.astype(int),
         distance_matrix=dist_matrix,
         neighborhoods=np.array([np.array(nb) for nb in neighborhoods], dtype=object),
         eps=eps,
         minpts=minpts,
         num_clusters=total_clusters,
         cluster_sizes=cluster_sizes,
         scores=scores,
         nearest_other=np.array(nearest_other),
         nearest_other_dist=nearest_other_dist)

print("\nData saved to outlier_data.npz")
```

plots.py

```python
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
```

## Conclusion

In this assignment, we applied a clustering-based outlier detection method using DBSCAN on eight 2D objects. With ε = 1.5 and MinPts = 2 (Euclidean distance), DBSCAN produced two clusters — Cluster 1 {A, B, C, D} and Cluster 2 {F, G} — while E and H were left as noise and treated as single-object clusters.

The outlier scoring formula score(o) = dist(o, o') / |c| penalizes objects that are far from other clusters relative to their own cluster size. Object E(4,2) received a score of 2.236 (greater than the threshold of 2), making it the only detected outlier. Its high score comes from being isolated (|c| = 1) while being at a distance of √5 ≈ 2.236 from its nearest neighbor in another cluster. Object H(8,0) also sits in a single-object cluster but has a score of exactly 2.0, which does not exceed the threshold.
