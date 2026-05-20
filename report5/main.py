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
    [2, 0],  # B
    [2, 1],  # C
    [3, 0],  # D
    [4, 2],  # E
    [5, 0],  # F
    [6, 0],  # G
    [8, 0],  # H
])

eps = 2
minpts = 3


def manhattan(a, b):
    return np.sum(np.abs(a - b))


# ============================================================
# Print header
# ============================================================
print("=" * 70)
print("DBSCAN Clustering")
print("=" * 70)
print(f"Points: {', '.join(f'{pid}({p[0]},{p[1]})' for pid, p in zip(point_ids, points))}")
print(f"Distance: Manhattan (L1)")
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
        dist_matrix[i, j] = manhattan(points[i], points[j])

print("=" * 70)
print("Manhattan Distance Matrix")
print("=" * 70)
header = f"{'':>6}"
for pid in point_ids:
    header += f"{pid:>6}"
print(header)
print("-" * (6 + 6 * n))
for i in range(n):
    row = f"{point_ids[i]:>6}"
    for j in range(n):
        row += f"{int(dist_matrix[i, j]):>6}"
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
# Final Result
# ============================================================
print()
print("=" * 70)
print("FINAL RESULT")
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

# Summary table
print(f"{'Point':<8} {'Type':<10} {'Cluster':<12}")
print("-" * 30)
for i in range(n):
    if labels[i] >= 0:
        ptype = "Core" if len(neighborhoods[i]) >= minpts else "Border"
        cluster_str = f"Cluster {labels[i] + 1}"
    else:
        ptype = "Noise"
        cluster_str = "-"
    print(f"{point_ids[i]:<8} {ptype:<10} {cluster_str:<12}")

# ============================================================
# Save data for plotting
# ============================================================
# Convert labels: noise(-2) → -1 for plotting convenience
plot_labels = labels.copy().astype(int)

np.savez('report5/dbscan_data.npz',
         points=points,
         point_ids=np.array(point_ids),
         labels=plot_labels,
         distance_matrix=dist_matrix,
         neighborhoods=np.array([np.array(nb) for nb in neighborhoods], dtype=object),
         eps=eps,
         minpts=minpts,
         num_clusters=num_clusters)

print("\nData saved to report5/dbscan_data.npz")
