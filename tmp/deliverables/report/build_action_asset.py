from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"D:\AirSim\rl_drone_navigation")
OUT = ROOT / "tmp" / "deliverables" / "report" / "assets" / "action_distribution.png"
RUNS = {
    "DQN Scratch": ROOT
    / "experiments"
    / "airsimnh"
    / "dqn"
    / "scratch_33m_45k_seed7_stable_v3_scratch_validated_test_seed20007"
    / "results"
    / "evaluation_deterministic_trajectory.csv",
    "PPO Scratch": ROOT
    / "experiments"
    / "airsimnh"
    / "ppo"
    / "scratch_33m_45k_seed7_stable_v3_scratch_validated_test_seed20007"
    / "results"
    / "evaluation_deterministic_trajectory.csv",
    "PPO Curriculum": ROOT
    / "experiments"
    / "airsimnh"
    / "ppo"
    / "curriculum_stage03_33m_30k_seed7_stable_v3_stage3_pilot_validated_test_seed20007"
    / "results"
    / "evaluation_deterministic_trajectory.csv",
}
ACTIONS = ["forward", "left", "right", "up", "down", "hover"]
COLORS = {
    "forward": "#178F74",
    "left": "#146C94",
    "right": "#63A6C6",
    "up": "#D88C00",
    "down": "#C44536",
    "hover": "#7A7F87",
}


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def action_percentages(path: Path) -> dict[str, float]:
    counts = {action: 0 for action in ACTIONS}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            action = row["action_name"]
            if action in counts:
                counts[action] += 1
    total = max(sum(counts.values()), 1)
    return {action: 100.0 * count / total for action, count in counts.items()}


def build() -> None:
    width, height = 1600, 720
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    regular = font(r"C:\Windows\Fonts\arial.ttf", 28)
    small = font(r"C:\Windows\Fonts\arial.ttf", 24)
    bold = font(r"C:\Windows\Fonts\arialbd.ttf", 28)
    bar_left, bar_right = 300, 1540
    plot_width = bar_right - bar_left
    bar_height = 92
    row_tops = [80, 230, 380]

    for tick in range(0, 101, 20):
        x = bar_left + round(plot_width * tick / 100)
        draw.line((x, 45, x, 515), fill="#D7DCE1", width=2)
        label = str(tick)
        box = draw.textbbox((0, 0), label, font=small)
        draw.text((x - (box[2] - box[0]) / 2, 525), label, fill="#30363B", font=small)

    for (name, path), top in zip(RUNS.items(), row_tops):
        label_box = draw.textbbox((0, 0), name, font=regular)
        label_y = top + (bar_height - (label_box[3] - label_box[1])) / 2
        draw.text((bar_left - 22 - (label_box[2] - label_box[0]), label_y), name, fill="#20252A", font=regular)
        values = action_percentages(path)
        current_x = bar_left
        for action in ACTIONS:
            value = values[action]
            segment_width = plot_width * value / 100
            next_x = current_x + segment_width
            draw.rectangle((current_x, top, next_x, top + bar_height), fill=COLORS[action])
            if value >= 5:
                label = f"{value:.0f}%"
                box = draw.textbbox((0, 0), label, font=bold)
                text_color = "white" if action in {"forward", "left", "down", "hover"} else "#20252A"
                draw.text(
                    (
                        current_x + (segment_width - (box[2] - box[0])) / 2,
                        top + (bar_height - (box[3] - box[1])) / 2 - 2,
                    ),
                    label,
                    fill=text_color,
                    font=bold,
                )
            current_x = next_x

    axis_label = "Share of deterministic evaluation actions (%)"
    box = draw.textbbox((0, 0), axis_label, font=regular)
    draw.text(((width - (box[2] - box[0])) / 2, 575), axis_label, fill="#20252A", font=regular)

    legend_y = 645
    legend_item_width = 245
    legend_left = (width - legend_item_width * len(ACTIONS)) / 2
    for index, action in enumerate(ACTIONS):
        x = legend_left + index * legend_item_width
        draw.rectangle((x, legend_y, x + 32, legend_y + 24), fill=COLORS[action])
        draw.text((x + 43, legend_y - 4), action.title(), fill="#20252A", font=small)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUT, dpi=(220, 220))
    print(OUT)


if __name__ == "__main__":
    build()
