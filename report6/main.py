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
