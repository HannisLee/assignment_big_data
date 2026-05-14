import sys
import io
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============================================================
# Data
# ============================================================
point_ids = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
points = np.array([
    [2, 10],  # A
    [2, 5],   # B
    [8, 4],   # C
    [5, 8],   # D
    [7, 5],   # E
    [6, 4],   # F
    [1, 2],   # G
    [4, 9],   # H
])

k = 3
# Initial seed points: A(2,10), D(5,8), G(1,2)
seed_indices = [0, 3, 6]  # A, D, G
cluster_names = ['Cluster 1 (seed A)', 'Cluster 2 (seed D)', 'Cluster 3 (seed G)']

centroids = points[seed_indices].copy().astype(float)
assignments = np.full(len(points), -1, dtype=int)

print("=" * 70)
print("K-Means Clustering")
print("=" * 70)
print(f"Points: {', '.join(f'{pid}({p[0]},{p[1]})' for pid, p in zip(point_ids, points))}")
print(f"k = {k}")
print(f"Initial seeds: {', '.join(f'{point_ids[i]}({points[i][0]},{points[i][1]})' for i in seed_indices)}")
print()


def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


iteration = 0
while True:
    iteration += 1
    print("=" * 70)
    print(f"Iteration {iteration}")
    print("=" * 70)
    print(f"\nCurrent centroids:")
    for c_idx in range(k):
        print(f"  {cluster_names[c_idx]}: ({centroids[c_idx][0]:.4f}, {centroids[c_idx][1]:.4f})")
    print()

    # Compute distances
    print("Distance from each point to each centroid:")
    header = f"{'Point':<8}"
    for c_idx in range(k):
        short_name = f"C{c_idx+1}({centroids[c_idx][0]:.2f},{centroids[c_idx][1]:.2f})"
        header += f"{short_name:<22}"
    header += f"{'Assignment':<14}"
    print(header)
    print("-" * len(header))

    new_assignments = np.full(len(points), -1, dtype=int)
    distances = np.zeros((len(points), k))

    for i in range(len(points)):
        row = f"{point_ids[i]}({points[i][0]},{points[i][1]})    "
        for c_idx in range(k):
            d = euclidean(points[i], centroids[c_idx])
            distances[i, c_idx] = d
            row += f"{d:<22.4f}"
        nearest = np.argmin(distances[i])
        new_assignments[i] = nearest
        row += f"{cluster_names[nearest]}"
        print(row)

    print()

    # Show cluster membership
    print("Cluster assignments:")
    for c_idx in range(k):
        members = [point_ids[i] for i in range(len(points)) if new_assignments[i] == c_idx]
        print(f"  {cluster_names[c_idx]}: {', '.join(members)}")
    print()

    # Check convergence
    if np.array_equal(new_assignments, assignments):
        print(">>> No change in assignments. Algorithm CONVERGED.")
        break

    assignments = new_assignments.copy()

    # Update centroids
    print("Updating centroids (mean of assigned points):")
    for c_idx in range(k):
        mask = new_assignments == c_idx
        member_points = points[mask]
        new_centroid = member_points.mean(axis=0)
        old_c = centroids[c_idx].copy()
        centroids[c_idx] = new_centroid
        members_str = ', '.join([f"{point_ids[i]}({points[i][0]},{points[i][1]})" for i in range(len(points)) if new_assignments[i] == c_idx])
        print(f"  {cluster_names[c_idx]}: members = [{members_str}]")
        print(f"    old centroid = ({old_c[0]:.4f}, {old_c[1]:.4f})")
        print(f"    new centroid = ({new_centroid[0]:.4f}, {new_centroid[1]:.4f})")
    print()

# Final result
print()
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)
print(f"\nTotal iterations: {iteration}")
print()
print("Final cluster assignments:")
for c_idx in range(k):
    members = [point_ids[i] for i in range(len(points)) if assignments[i] == c_idx]
    print(f"  Cluster {c_idx+1}: {', '.join(members)} (centroid: ({centroids[c_idx][0]:.4f}, {centroids[c_idx][1]:.4f}))")

# Save data for plotting
np.savez('report4/kmeans_data.npz',
         points=points,
         point_ids=np.array(point_ids),
         assignments=assignments,
         centroids=centroids,
         initial_seeds=points[seed_indices],
         seed_indices=np.array(seed_indices))

print("\nData saved to report4/kmeans_data.npz")
