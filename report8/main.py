"""
L8 Assignment — Blocking Rules for Entity Linkage
Design f ∘ g blocking rules on the "phone" and "date of birth" attributes so
that (1) all matching records share a block (100% recall) and (2) the total
number of pairwise intrablock comparisons is minimized.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import numpy as np
import re
from collections import defaultdict

# ── Data (raw values, exactly as given in the assignment) ────────────────────
table_a = [
    {"id": "A1", "name": "Smith, John",  "phone": "445-881-4478",    "dob": "August 12, 1989", "state": "Maine"},
    {"id": "A2", "name": "Jennifer Tal", "phone": "+1-189-456-4513", "dob": "11/12/1965",      "state": "Tx"},
    {"id": "A3", "name": "Gates, Bill",  "phone": "(876)546-8165",   "dob": "June 15, 1972",   "state": "Kansas"},
    {"id": "A4", "name": "Alan Fitch",   "phone": "5493156648",      "dob": "2-6-1985",        "state": "Oh"},
    {"id": "A5", "name": "Jacob Alan",   "phone": "(205)1564896",    "dob": "1985 January 3",  "state": "Alabama"},
]
table_b = [
    {"id": "B1", "name": "John Smith",   "phone": "445-881-4478",    "dob": "08/12/1989", "state": "Maine"},
    {"id": "B2", "name": "Jennifer Tal", "phone": "189-456-4513",    "dob": "11/12/1965",  "state": "Tx"},
    {"id": "B3", "name": "Bill Gates",   "phone": "876-546-8165",    "dob": "06/15/1972",  "state": "Kansas"},
    {"id": "B4", "name": "Alan Fitch",   "phone": "549-315-6648",    "dob": "02/06/1985",  "state": "Oh"},
    {"id": "B5", "name": "Jacob Alan",   "phone": "205-156-4896",    "dob": "01/03/1985",  "state": "Alabama"},
]
all_records = table_a + table_b

# Ground-truth matching pairs (A_i ↔ B_i)
gold_pairs = {("A1", "B1"), ("A2", "B2"), ("A3", "B3"), ("A4", "B4"), ("A5", "B5")}


# ── Transformation functions g (inject normalization into the blocking rule) ─
def g_phone(raw):
    """Keep digits only; if 11 digits starting with '1', drop the country code."""
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 11 and digits[0] == "1":
        digits = digits[1:]
    return digits


_MONTHS = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
           "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12}


def g_dob(raw):
    """Parse month names and numeric MM/DD/YYYY (US convention) -> YYYYMMDD."""
    s = raw.strip().lower()
    month_val = None
    for name, num in _MONTHS.items():
        if name in s:
            month_val = num
            s = s.replace(name, " ")
            break
    nums = re.findall(r"\d+", s)
    if month_val is not None:
        year = next(n for n in nums if len(n) == 4)   # the 4-digit token is the year
        day = next(n for n in nums if len(n) != 4)    # the remaining token is the day
        m, d, y = month_val, int(day), int(year)
    else:
        # fully numeric -> assume MM/DD/YYYY (separators / or -), US convention
        m, d, y = int(nums[0]), int(nums[1]), int(nums[2])
    return f"{y:04d}{m:02d}{d:02d}"


# ── Blocking function f applied on top of g ──────────────────────────────────
def f_identity(x):
    """f = identity: use the full canonical value produced by g as the block key."""
    return x


def transform(attr):
    return g_phone if attr == "phone" else g_dob


# ── Blocking engine ──────────────────────────────────────────────────────────
def block_key(record, attr, use_transform):
    raw = record[attr]
    if use_transform:
        return f_identity(transform(attr)(raw))   # f ∘ g
    return raw                                      # raw-string blocking (no g)


def form_blocks(records, attr, use_transform):
    blocks = defaultdict(list)
    for r in records:
        blocks[block_key(r, attr, use_transform)].append(r["id"])
    return blocks


def cross_table_comparisons(blocks):
    """Σ_blocks (#A in block) × (#B in block): A–B pairs that must be compared."""
    total = 0
    for members in blocks.values():
        a = sum(1 for m in members if m.startswith("A"))
        b = sum(1 for m in members if m.startswith("B"))
        total += a * b
    return total


def recall(blocks, gold):
    """Fraction of gold pairs that share a block."""
    rec2key = {m: k for k, members in blocks.items() for m in members}
    found = sum(1 for a, b in gold if rec2key.get(a) == rec2key.get(b) and a in rec2key)
    return found / len(gold)


# ── Report helper ────────────────────────────────────────────────────────────
def banner(title):
    print("=" * 70)
    print(title)
    print("=" * 70)


# ── Show the transformation g on every record ────────────────────────────────
banner("Transformation g applied to raw values")
print(f"{'ID':<4} {'attr':<6} {'raw':<22} -> {'canonical (g)':<16}")
print("-" * 70)
for attr in ("phone", "dob"):
    for r in all_records:
        print(f"{r['id']:<4} {attr:<6} {r[attr]:<22} -> {transform(attr)(r[attr]):<16}")
    print("-" * 70)


# ── Evaluate the three strategies for each attribute ─────────────────────────
strategies = {
    "No blocking (|A|x|B|)": None,        # special: full cross-table baseline
    "Raw-string blocking":   ("raw", False),
    "f o g blocking":        ("fg", True),
}

results = {}   # results[attr][strategy_label] = (comparisons, recall, blocks)
for attr in ("phone", "dob"):
    banner(f"Attribute = {attr.upper()}")
    results[attr] = {}
    for label, spec in strategies.items():
        if spec is None:
            comps = len(table_a) * len(table_b)        # 5 x 5 = 25
            rec = 1.0
            blocks = {}
        else:
            _, use_t = spec
            blocks = form_blocks(all_records, attr, use_t)
            comps = cross_table_comparisons(blocks)
            rec = recall(blocks, gold_pairs)
        results[attr][label] = (comps, rec, blocks)
        print(f"  {label:<22} -> comparisons = {comps:<3}  recall = {rec*100:>5.1f}%")
    # detail for f o g
    fg_blocks = results[attr]["f o g blocking"][2]
    print(f"\n  f o g blocks for {attr} ({len(fg_blocks)} blocks):")
    for k, members in fg_blocks.items():
        print(f"     key = {k:<14}  members = {members}")
    print()

# ── Figure 1: comparisons vs. recall across strategies ───────────────────────
strategy_labels = ["No blocking\n(|A|x|B|)", "Raw-string\nblocking", "f o g\nblocking"]
keys = list(strategies.keys())
phone_cmps = [results["phone"][k][0] for k in keys]
dob_cmps = [results["dob"][k][0] for k in keys]
phone_rec = [results["phone"][k][1] for k in keys]
dob_rec = [results["dob"][k][1] for k in keys]

x = np.arange(len(strategy_labels))
w = 0.38
fig, ax = plt.subplots(figsize=(11, 6.5))
b1 = ax.bar(x - w / 2, phone_cmps, w, label="Phone", color="#3498db", edgecolor="white", linewidth=1.2)
b2 = ax.bar(x + w / 2, dob_cmps, w, label="Date of birth", color="#e67e22", edgecolor="white", linewidth=1.2)

ax.set_xticks(x)
ax.set_xticklabels(strategy_labels, fontsize=12)
ax.set_ylabel("# Intrablock comparisons (A-B pairs sharing a block)", fontsize=12)
ax.set_title("Blocking strategies — f ∘ g is optimal: 100% recall with the fewest comparisons",
             fontsize=13, fontweight="bold")
ax.set_ylim(0, 31)
ax.legend(fontsize=11, loc="upper right")

for bars, recs in [(b1, phone_rec), (b2, dob_rec)]:
    for bar, r in zip(bars, recs):
        h = bar.get_height()
        color = "#c0392b" if r < 1.0 else "#1e8449"
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                f"{int(h)}\nrecall {int(r * 100)}%", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color=color)

# subtly highlight the optimal strategy region
ax.axvspan(1.55, 2.45, color="#1e8449", alpha=0.07, zorder=0)
plt.tight_layout()
plt.savefig("blocking_comparisons.png", dpi=150, bbox_inches="tight")
print("Saved blocking_comparisons.png")


# ── Figure 2: block structure formed by f o g ────────────────────────────────
def draw_blocks(ax, attr):
    blocks = results[attr]["f o g blocking"][2]
    keys_list = list(blocks.keys())
    n = len(keys_list)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, n + 1.2)
    ax.axis("off")
    ax.set_title(f'Blocking on "{attr}" — f ∘ g forms {n} blocks (all matches co-blocked)',
                 fontsize=12.5, fontweight="bold")
    # column headers
    ax.text(0.5, n + 0.7, "Block key", ha="left", fontsize=11, fontweight="bold", color="#2c3e50")
    ax.text(5.4, n + 0.7, "Table A", ha="center", fontsize=11, fontweight="bold", color="#21618c")
    ax.text(8.2, n + 0.7, "Table B", ha="center", fontsize=11, fontweight="bold", color="#922b21")
    for i, k in enumerate(keys_list):
        y = n - 1 - i
        ax.add_patch(Rectangle((0.2, y - 0.35), 9.5, 0.7, fill=True,
                               facecolor="#f4f6f7", edgecolor="#bdc3c7", linewidth=1))
        members = blocks[k]
        a = ", ".join(m for m in members if m.startswith("A"))
        b = ", ".join(m for m in members if m.startswith("B"))
        ax.text(0.5, y, f"key = {k}", ha="left", va="center", fontsize=10.5,
                fontweight="bold", color="#2c3e50")
        ax.text(5.4, y, a, ha="center", va="center", fontsize=11.5,
                fontweight="bold", color="#21618c")
        ax.text(8.2, y, b, ha="center", va="center", fontsize=11.5,
                fontweight="bold", color="#922b21")


fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
draw_blocks(axes[0], "phone")
draw_blocks(axes[1], "dob")
legend = [Patch(facecolor="#21618c", label="Table A record"),
          Patch(facecolor="#922b21", label="Table B record (matches the A record in the same block)")]
fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=11, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
plt.savefig("blocking_blocks.png", dpi=150, bbox_inches="tight")
print("Saved blocking_blocks.png")


# ── Final summary ────────────────────────────────────────────────────────────
banner("RESULT")
for attr in ("phone", "dob"):
    comps = results[attr]["f o g blocking"][0]
    rec = results[attr]["f o g blocking"][1]
    print(f"  {attr:<5} f o g -> {comps} comparisons, recall = {rec*100:.0f}%  "
          f"(no-blocking baseline = 25)")
print("=" * 70)
