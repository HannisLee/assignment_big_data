"""
L7 Assignment — KNN Imputation for Missing BMI
Using k-Nearest Neighbors with weighted average to fill Ada's missing BMI.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
data = {
    "Ada":   {"gender": "Female", "age": 39, "position": "Associate Professor", "bmi": None, "drink": "Occasionally", "smoke": "No",    "tc": 172},
    "Alex":  {"gender": "Male",   "age": 53, "position": "Professor",            "bmi": 31.7, "drink": "Often",        "smoke": "Yes",   "tc": 255},
    "Bill":  {"gender": "Male",   "age": 25, "position": "Assistant Professor",  "bmi": 20.5, "drink": "No",           "smoke": "No",    "tc": 159},
    "Daisy": {"gender": "Female", "age": 28, "position": "Assistant Professor",  "bmi": 22.7, "drink": "Occasionally", "smoke": "No",    "tc": 166},
    "David": {"gender": "Male",   "age": 45, "position": "Professor",            "bmi": 30.2, "drink": "Often",        "smoke": "Yes",   "tc": 242},
    "John":  {"gender": "Male",   "age": 37, "position": "Associate Professor",  "bmi": 21.6, "drink": "No",           "smoke": "Yes",   "tc": 180},
    "Kate":  {"gender": "Female", "age": 48, "position": "Professor",            "bmi": 26.3, "drink": "Occasionally", "smoke": "Yes",   "tc": 181},
    "Lewis": {"gender": "Two-spirit", "age": 40, "position": "Associate Professor", "bmi": 24.4, "drink": "Occasionally", "smoke": "No", "tc": 192},
    "Lily":  {"gender": "Female", "age": 52, "position": "Professor",            "bmi": 28.0, "drink": "Occasionally", "smoke": "No",    "tc": 201},
    "Mary":  {"gender": "Female", "age": 43, "position": "Associate Professor",  "bmi": 28.6, "drink": "Often",        "smoke": "No",    "tc": 215},
}

k = 2
ada_tc = data["Ada"]["tc"]  # 172

# ── Step 1: Filter same gender (Female) ───────────────────────────────────────
females = {name: info for name, info in data.items()
           if info["gender"] == "Female" and name != "Ada"}

print("=" * 60)
print("Step 1: Female candidates (same gender as Ada)")
print("=" * 60)
print(f"Ada's Total Cholesterol = {ada_tc}\n")
print(f"{'Name':<8} {'Age':>4} {'BMI':>6} {'TC':>5} {'|TC - Ada TC|':>14}")
print("-" * 45)
for name, info in females.items():
    diff = abs(info["tc"] - ada_tc)
    print(f"{name:<8} {info['age']:>4} {info['bmi']:>6.1f} {info['tc']:>5} {diff:>14}")

# ── Step 2: Compute distances and find k nearest ─────────────────────────────
distances = {}
for name, info in females.items():
    distances[name] = abs(info["tc"] - ada_tc)

sorted_neighbors = sorted(distances.items(), key=lambda x: x[1])
k_nearest = sorted_neighbors[:k]

print(f"\n{'=' * 60}")
print(f"Step 2: k = {k} Nearest Neighbors (by |Total Cholesterol difference|)")
print("=" * 60)
for rank, (name, dist) in enumerate(k_nearest, 1):
    print(f"  Neighbor {rank}: {name} (TC = {females[name]['tc']}, distance = {dist})")

# ── Step 3: Weighted average ──────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("Step 3: Weighted Average for Ada's BMI")
print("=" * 60)

# Raw reciprocal weights
raw_weights = {name: 1.0 / dist for name, dist in k_nearest}
total_raw = sum(raw_weights.values())

print("\nRaw reciprocal weights:")
for name, w in raw_weights.items():
    print(f"  {name}: 1/{distances[name]} = {w:.6f}")
print(f"  Sum of raw weights = {total_raw:.6f}")

# Normalized weights (sum to 1)
norm_weights = {name: w / total_raw for name, w in raw_weights.items()}

print("\nNormalized weights (sum to 1):")
for name, w in norm_weights.items():
    print(f"  {name}: {w:.6f}")
print(f"  Sum = {sum(norm_weights.values()):.6f}")

# Weighted BMI
ada_bmi = sum(norm_weights[name] * females[name]["bmi"] for name in norm_weights)
print(f"\nAda's estimated BMI = ", end="")
terms = " + ".join(f"({w:.4f} × {females[n]['bmi']})" for n, w in norm_weights.items())
print(f"{terms}")
print(f"                   = {ada_bmi:.4f}")

# ── Visualization ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Plot 1: TC distances bar chart ---
ax1 = axes[0]
all_females_tc = [(name, info["tc"], abs(info["tc"] - ada_tc)) for name, info in females.items()]
all_females_tc.sort(key=lambda x: x[2])

names_sorted = [x[0] for x in all_females_tc]
tc_sorted = [x[1] for x in all_females_tc]
diff_sorted = [x[2] for x in all_females_tc]

neighbor_names = {n for n, _ in k_nearest}
colors = ["#e74c3c" if name in neighbor_names else "#3498db" for name in names_sorted]

bars = ax1.barh(names_sorted, diff_sorted, color=colors, edgecolor="white", linewidth=1.2)
ax1.axvline(x=0, color="gray", linewidth=0.8)
ax1.set_xlabel("|Total Cholesterol − Ada's TC (172)|", fontsize=12)
ax1.set_title("Distance to Ada (Female candidates)", fontsize=14, fontweight="bold")
ax1.invert_yaxis()

for bar, dist, tc_val in zip(bars, diff_sorted, tc_sorted):
    ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
             f"TC={tc_val}, Δ={dist}", va="center", fontsize=10)

# Legend
legend_elements = [Patch(facecolor="#e74c3c", label=f"k={k} nearest"),
                   Patch(facecolor="#3498db", label="Other candidates")]
ax1.legend(handles=legend_elements, loc="lower right", fontsize=11)

# --- Plot 2: Weighted BMI computation ---
ax2 = axes[1]
neighbor_labels = list(norm_weights.keys())
neighbor_bmis = [females[n]["bmi"] for n in neighbor_labels]
neighbor_weights = [norm_weights[n] for n in neighbor_labels]
neighbor_dists = [distances[n] for n in neighbor_labels]

x_pos = np.arange(len(neighbor_labels))
bar_width = 0.35

bars_bmi = ax2.bar(x_pos - bar_width / 2, neighbor_bmis, bar_width,
                    color="#2ecc71", edgecolor="white", linewidth=1.2, label="BMI")
bars_w = ax2.bar(x_pos + bar_width / 2, neighbor_weights, bar_width,
                   color="#e67e22", edgecolor="white", linewidth=1.2, label="Weight")

ax2.axhline(y=ada_bmi, color="#e74c3c", linestyle="--", linewidth=2, label=f"Ada's BMI ≈ {ada_bmi:.2f}")
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f"{n}\n(TC={females[n]['tc']}, Δ={neighbor_dists[i]})" for i, n in enumerate(neighbor_labels)], fontsize=11)
ax2.set_ylabel("Value", fontsize=12)
ax2.set_title("KNN Weighted Average for Ada's BMI", fontsize=14, fontweight="bold")
ax2.legend(fontsize=11)

# Annotate bars
for bar, val in zip(bars_bmi, neighbor_bmis):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
             f"{val:.1f}", ha="center", fontsize=11, fontweight="bold")
for bar, val in zip(bars_w, neighbor_weights):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{val:.4f}", ha="center", fontsize=11, fontweight="bold")

# Add formula text
formula = f"Ada BMI = {neighbor_weights[0]:.4f}×{neighbor_bmis[0]} + {neighbor_weights[1]:.4f}×{neighbor_bmis[1]} = {ada_bmi:.4f}"
ax2.text(0.5, -0.18, formula, transform=ax2.transAxes, ha="center", fontsize=12,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="gray"))

plt.tight_layout()
plt.savefig("knn_result.png", dpi=150, bbox_inches="tight")
print(f"\nPlot saved to knn_result.png")
print(f"\n{'=' * 60}")
print(f"RESULT: Ada's estimated BMI = {ada_bmi:.4f}")
print(f"{'=' * 60}")
