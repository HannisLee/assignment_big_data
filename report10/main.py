"""
Report 10 - Memory-Based Collaborative Filtering

This script reproduces the hand calculation in report.md and generates the
supporting SVG figures without external dependencies.
"""

from html import escape
from pathlib import Path


M = [
    [1, 0, 1, 1, 1, 0],
    [1, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 1],
    [1, 1, 0, 1, 0, 0],
]

TARGET_INDEX = 5
TOP_K = 2

C_BLUE = "#3498db"
C_ORANGE = "#e67e22"
C_GREEN = "#1e8449"
C_RED = "#c0392b"
C_GREY = "#7f8c8d"
C_DARK = "#2c3e50"
C_LIGHT = "#eef3f7"
C_LINE = "#d6dee6"


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def compute_recommendation():
    target = M[TARGET_INDEX]
    similarities = [dot(row, target) for row in M]

    other_users = [i for i in range(len(M)) if i != TARGET_INDEX]
    ranked_users = sorted(other_users, key=lambda i: (-similarities[i], i))
    top_users = ranked_users[:TOP_K]

    candidate_items = [
        j
        for j in range(len(target))
        if target[j] == 0 and any(M[i][j] == 1 for i in top_users)
    ]
    predictions = {
        j: sum(M[i][j] for i in top_users)
        for j in candidate_items
    }
    recommended_item = max(candidate_items, key=lambda j: (predictions[j], -j))

    return {
        "target": target,
        "similarities": similarities,
        "ranked_users": ranked_users,
        "top_users": top_users,
        "candidate_items": candidate_items,
        "predictions": predictions,
        "recommended_item": recommended_item,
    }


def svg_doc(width, height, body):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  text {{ font-family: Arial, Helvetica, sans-serif; fill: {C_DARK}; }}
  .title {{ font-size: 22px; font-weight: 700; }}
  .subtitle {{ font-size: 13px; fill: {C_GREY}; }}
  .label {{ font-size: 13px; }}
  .small {{ font-size: 12px; fill: {C_GREY}; }}
  .value {{ font-size: 13px; font-weight: 700; }}
</style>
<rect width="100%" height="100%" fill="white"/>
{body}
</svg>
"""


def rect(x, y, w, h, fill, stroke="none", sw=1, rx=0):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )


def text(x, y, value, cls="label", anchor="start", fill=None, weight=None):
    fill_attr = f' fill="{fill}"' if fill else ""
    weight_attr = f' font-weight="{weight}"' if weight else ""
    return (
        f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}"'
        f'{fill_attr}{weight_attr}>{escape(str(value))}</text>'
    )


def line(x1, y1, x2, y2, color=C_LINE, sw=1):
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="{sw}"/>'
    )


def polyline(points, color=C_LINE, sw=2):
    pts = " ".join(f"{x},{y}" for x, y in points)
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{sw}"/>'


def arrow(x1, y1, x2, y2, color=C_GREY):
    head = 8
    return "\n".join(
        [
            line(x1, y1, x2, y2, color, 2),
            f'<polygon points="{x2},{y2} {x2-head},{y2-head/2} {x2-head},{y2+head/2}" fill="{color}"/>',
        ]
    )


def write_svg(path, width, height, body):
    path.write_text(svg_doc(width, height, body), encoding="utf-8")


def make_matrix_heatmap(out_dir, result):
    target_idx = TARGET_INDEX
    top_users = set(result["top_users"])
    cell = 48
    left = 86
    top = 92
    width = 470
    height = 440
    parts = [
        text(28, 38, "Purchase Matrix", "title"),
        text(28, 61, "Rows are users, columns are items; target row and top-2 neighbors are highlighted.", "subtitle"),
    ]

    for j in range(6):
        parts.append(text(left + j * cell + cell / 2, top - 18, f"i{j+1}", "label", "middle"))

    for i, row in enumerate(M):
        y = top + i * cell
        if i == target_idx:
            label = "u6 target"
            color = C_RED
        elif i in top_users:
            label = f"u{i+1} top"
            color = C_GREEN
        else:
            label = f"u{i+1}"
            color = C_GREY
        parts.append(text(28, y + 30, label, "label", fill=color, weight="700" if i in top_users or i == target_idx else None))
        for j, value in enumerate(row):
            x = left + j * cell
            fill = C_BLUE if value else C_LIGHT
            parts.append(rect(x, y, cell - 4, cell - 4, fill, C_LINE, 1, 5))
            parts.append(text(x + (cell - 4) / 2, y + 29, value, "value", "middle", fill="white" if value else C_GREY))
        if i == target_idx:
            parts.append(rect(left - 4, y - 4, 6 * cell, cell + 4, "none", C_RED, 3, 7))
        elif i in top_users:
            parts.append(rect(left - 4, y - 4, 6 * cell, cell + 4, "none", C_GREEN, 3, 7))

    parts.extend(
        [
            rect(92, 396, 18, 18, C_BLUE, C_LINE, 1, 4),
            text(118, 410, "1 = bought", "small"),
            rect(220, 396, 18, 18, C_LIGHT, C_LINE, 1, 4),
            text(246, 410, "0 = not bought", "small"),
        ]
    )
    write_svg(out_dir / "purchase_matrix.svg", width, height, "\n".join(parts))


def make_bar_chart(out_dir, filename, title, subtitle, labels, values, highlight_labels):
    if isinstance(highlight_labels, str):
        highlight_labels = {highlight_labels}
    else:
        highlight_labels = set(highlight_labels)

    width = 620
    height = 390
    left = 76
    top = 92
    chart_w = 470
    chart_h = 210
    max_v = max(values) if values else 1
    bar_w = chart_w / len(values) * 0.56
    gap = chart_w / len(values)
    parts = [
        text(32, 38, title, "title"),
        text(32, 61, subtitle, "subtitle"),
        line(left, top, left, top + chart_h, C_LINE, 1),
        line(left, top + chart_h, left + chart_w, top + chart_h, C_LINE, 1),
    ]

    for tick in range(max_v + 1):
        y = top + chart_h - (tick / max_v) * chart_h if max_v else top + chart_h
        parts.append(line(left - 5, y, left + chart_w, y, "#edf1f5", 1))
        parts.append(text(left - 14, y + 4, tick, "small", "end"))

    for idx, (label, value) in enumerate(zip(labels, values)):
        cx = left + gap * idx + gap / 2
        bar_h = 0 if max_v == 0 else (value / max_v) * chart_h
        x = cx - bar_w / 2
        y = top + chart_h - bar_h
        is_highlight = label in highlight_labels
        color = C_ORANGE if is_highlight else C_BLUE
        if is_highlight:
            parts.append(rect(x - 5, y - 7, bar_w + 10, bar_h + 14, "#fff4e7", "none", 1, 8))
        parts.append(rect(x, y, bar_w, bar_h, color, "none", 1, 7))
        parts.append(text(cx, y - 10, value, "value", "middle", fill=color))
        parts.append(text(cx, top + chart_h + 26, label, "label", "middle"))

    parts.append(text(width / 2, 350, f"Highlighted result: {', '.join(sorted(highlight_labels))}", "subtitle", "middle"))
    write_svg(out_dir / filename, width, height, "\n".join(parts))


def make_similarity_chart(out_dir, result):
    labels = [f"user {i+1}" for i in range(5)]
    values = [result["similarities"][i] for i in range(5)]
    highlighted = [f"user {i+1}" for i in result["top_users"]]
    make_bar_chart(
        out_dir,
        "similarity_scores.svg",
        "Inner Product Similarity",
        "Similarity between each non-target user and target user 6.",
        labels,
        values,
        highlighted,
    )


def make_prediction_chart(out_dir, result):
    labels = [f"item {j+1}" for j in result["candidate_items"]]
    values = [result["predictions"][j] for j in result["candidate_items"]]
    recommended = f"item {result['recommended_item'] + 1}"
    make_bar_chart(
        out_dir,
        "prediction_scores.svg",
        "Candidate Item Prediction Scores",
        "Simple sum over the top-2 similar users: user 1 and user 4.",
        labels,
        values,
        recommended,
    )


def make_flow_diagram(out_dir, result):
    width = 820
    height = 300
    boxes = [
        (34, 96, 132, 82, "Target user", "u6 = [1,1,0,1,0,0]"),
        (206, 96, 132, 82, "Similarity", "inner product"),
        (378, 96, 132, 82, "Top-2 users", "user 1, user 4"),
        (550, 96, 132, 82, "Candidates", "item 3, 5, 6"),
        (704, 96, 82, 82, "Result", "item 5"),
    ]
    parts = [
        text(32, 40, "Collaborative Filtering Pipeline", "title"),
        text(32, 63, "From target user to top-1 recommendation.", "subtitle"),
    ]
    for idx, (x, y, w, h, title, detail) in enumerate(boxes):
        fill = "#fff4e7" if idx == len(boxes) - 1 else "#f7fafc"
        stroke = C_ORANGE if idx == len(boxes) - 1 else C_LINE
        parts.append(rect(x, y, w, h, fill, stroke, 2, 8))
        parts.append(text(x + w / 2, y + 31, title, "label", "middle", fill=C_DARK, weight="700"))
        parts.append(text(x + w / 2, y + 57, detail, "small", "middle", fill=C_ORANGE if idx == len(boxes) - 1 else C_GREY))
        if idx < len(boxes) - 1:
            x2 = boxes[idx + 1][0]
            parts.append(arrow(x + w + 12, y + h / 2, x2 - 12, y + h / 2))

    parts.append(polyline([(82, 220), (744, 220)], "#edf1f5", 2))
    parts.append(text(82, 248, "sim: user1=2, user4=2", "small", "middle"))
    parts.append(text(412, 248, "p(item5)=1+1=2", "small", "middle"))
    parts.append(text(744, 248, "highest score", "small", "middle"))
    write_svg(out_dir / "collaborative_filtering_steps.svg", width, height, "\n".join(parts))


def print_result(result):
    print("=" * 72)
    print("Memory-Based Collaborative Filtering")
    print("=" * 72)
    print(f"Target user: user {TARGET_INDEX + 1} = {result['target']}")
    print()
    print("Similarity scores:")
    for i, score in enumerate(result["similarities"]):
        if i == TARGET_INDEX:
            continue
        print(f"  user {i + 1}: {score}")
    print()
    print("Top-2 similar users:")
    for i in result["top_users"]:
        print(f"  user {i + 1}: similarity = {result['similarities'][i]}")
    print()
    print("Candidate item predictions:")
    for j in result["candidate_items"]:
        vals = [M[i][j] for i in result["top_users"]]
        print(f"  item {j + 1}: {vals[0]} + {vals[1]} = {result['predictions'][j]}")
    print()
    print(f"Recommendation: item {result['recommended_item'] + 1}")


def main():
    out_dir = Path(__file__).resolve().parent
    result = compute_recommendation()
    print_result(result)
    make_matrix_heatmap(out_dir, result)
    make_similarity_chart(out_dir, result)
    make_prediction_chart(out_dir, result)
    make_flow_diagram(out_dir, result)
    print()
    print("Generated figures:")
    for name in [
        "purchase_matrix.svg",
        "similarity_scores.svg",
        "prediction_scores.svg",
        "collaborative_filtering_steps.svg",
    ]:
        print(f"  {out_dir / name}")


if __name__ == "__main__":
    main()
