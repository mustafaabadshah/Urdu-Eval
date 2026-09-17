"""Script to generate animated demo GIF for UrduEval terminal execution."""

import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def create_demo_gif() -> None:
    width, height = 860, 480
    bg_color = (10, 14, 23)
    header_color = (24, 32, 47)
    card_color = (15, 23, 42)
    border_color = (40, 50, 70)
    text_white = (241, 245, 249)
    text_green = (52, 211, 153)
    text_cyan = (56, 189, 248)
    text_yellow = (251, 191, 36)
    text_dim = (148, 163, 184)
    text_red = (248, 113, 113)

    try:
        font = ImageFont.truetype("consola.ttf", 15)
        font_bold = ImageFont.truetype("consolab.ttf", 15)
        font_sm = ImageFont.truetype("consola.ttf", 13)
        font_lg = ImageFont.truetype("consolab.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
        font_bold = font
        font_sm = font
        font_lg = font

    frames = []

    def draw_window_frame():
        im = Image.new("RGB", (width, height), color=bg_color)
        draw = ImageDraw.Draw(im)
        # Window outer border
        draw.rounded_rectangle(
            [10, 10, width - 10, height - 10],
            radius=10,
            fill=bg_color,
            outline=border_color,
            width=1,
        )
        # Title bar
        draw.rounded_rectangle([10, 10, width - 10, 48], radius=10, fill=header_color)
        draw.rectangle([10, 38, width - 10, 48], fill=header_color)
        draw.line([10, 48, width - 10, 48], fill=border_color, width=1)
        # Dots
        draw.ellipse([26, 23, 38, 35], fill=(239, 68, 68))
        draw.ellipse([46, 23, 58, 35], fill=(245, 158, 11))
        draw.ellipse([66, 23, 78, 35], fill=(16, 185, 129))
        # Title text
        draw.text((width // 2 - 80, 21), "urdu-eval — zsh — 860x480", font=font_sm, fill=text_dim)
        return im, draw

    # Frame 1: Typing prompt
    im1, d1 = draw_window_frame()
    d1.text((30, 70), "researcher@node:~/urdu-eval$ ", font=font_bold, fill=text_green)
    d1.text((270, 70), "urdu-eval run --benchmark urdu-qa", font=font, fill=text_white)
    d1.rectangle([545, 70, 555, 87], fill=text_green)

    # Frame 2: Model setup
    im2, d2 = draw_window_frame()
    d2.text(
        (30, 70),
        "researcher@node:~/urdu-eval$ urdu-eval run --benchmark urdu-qa",
        font=font,
        fill=text_dim,
    )
    d2.rounded_rectangle(
        [30, 105, width - 30, 155], radius=6, fill=card_color, outline=border_color
    )
    d2.text((45, 115), "UrduEval v0.1.0", font=font_bold, fill=text_green)
    d2.text(
        (45, 133),
        "Provider: ollama  |  Model: llama3.1:8b  |  Benchmark: urdu-qa (15 items)",
        font=font_sm,
        fill=text_cyan,
    )

    # Frame 3: 40% Progress
    im3, d3 = draw_window_frame()
    d3.text(
        (30, 70),
        "researcher@node:~/urdu-eval$ urdu-eval run --benchmark urdu-qa",
        font=font,
        fill=text_dim,
    )
    d3.rounded_rectangle(
        [30, 105, width - 30, 155], radius=6, fill=card_color, outline=border_color
    )
    d3.text((45, 115), "UrduEval v0.1.0", font=font_bold, fill=text_green)
    d3.text(
        (45, 133),
        "Provider: ollama  |  Model: llama3.1:8b  |  Benchmark: urdu-qa (15 items)",
        font=font_sm,
        fill=text_cyan,
    )
    d3.text(
        (30, 180),
        "Evaluating: 40% [=============>                   ] 6/15 [00:01<00:02, 4.1 it/s]",
        font=font,
        fill=text_yellow,
    )

    # Frame 4: 100% Progress
    im4, d4 = draw_window_frame()
    d4.text(
        (30, 70),
        "researcher@node:~/urdu-eval$ urdu-eval run --benchmark urdu-qa",
        font=font,
        fill=text_dim,
    )
    d4.rounded_rectangle(
        [30, 105, width - 30, 155], radius=6, fill=card_color, outline=border_color
    )
    d4.text((45, 115), "UrduEval v0.1.0", font=font_bold, fill=text_green)
    d4.text(
        (45, 133),
        "Provider: ollama  |  Model: llama3.1:8b  |  Benchmark: urdu-qa (15 items)",
        font=font_sm,
        fill=text_cyan,
    )
    d4.text(
        (30, 180),
        "Evaluating: 100% [==================================] 15/15 [00:02<00:00, 5.8 it/s]",
        font=font,
        fill=text_green,
    )

    # Frame 5: Results Table & Diagnostics
    im5, d5 = draw_window_frame()
    d5.text(
        (30, 65),
        "researcher@node:~/urdu-eval$ urdu-eval run --benchmark urdu-qa",
        font=font_sm,
        fill=text_dim,
    )
    # Results Table Box
    d5.rounded_rectangle([30, 95, width - 30, 265], radius=6, fill=card_color, outline=border_color)
    d5.text((45, 108), "METRIC", font=font_bold, fill=text_white)
    d5.text((220, 108), "SCORE", font=font_bold, fill=text_white)
    d5.text((360, 108), "95% CONFIDENCE", font=font_bold, fill=text_white)
    d5.text((560, 108), "STATUS", font=font_bold, fill=text_white)
    d5.line([30, 130, width - 30, 130], fill=border_color)

    d5.text((45, 142), "Exact Match", font=font, fill=text_white)
    d5.text((220, 142), "60.0%", font=font_bold, fill=text_green)
    d5.text((360, 142), "[35.7% - 82.7%]", font=font_sm, fill=text_dim)
    d5.text((560, 142), "Passed", font=font_sm, fill=text_green)

    d5.text((45, 172), "Token F1", font=font, fill=text_white)
    d5.text((220, 172), "78.4%", font=font_bold, fill=text_cyan)
    d5.text((360, 172), "[58.2% - 91.1%]", font=font_sm, fill=text_dim)
    d5.text((560, 172), "Passed", font=font_sm, fill=text_green)

    d5.text((45, 202), "chrF++", font=font, fill=text_white)
    d5.text((220, 202), "74.2%", font=font_bold, fill=text_yellow)
    d5.text((360, 202), "[52.8% - 88.0%]", font=font_sm, fill=text_dim)
    d5.text((560, 202), "Passed", font=font_sm, fill=text_green)

    d5.text((45, 232), "Latency (mean)", font=font, fill=text_white)
    d5.text((220, 232), "142 ms", font=font, fill=text_dim)
    d5.text((360, 232), "p95: 198 ms", font=font_sm, fill=text_dim)
    d5.text((560, 232), "Fast", font=font_sm, fill=text_cyan)

    # Diagnostics Box
    d5.rounded_rectangle(
        [30, 280, width - 30, 370], radius=6, fill=card_color, outline=border_color
    )
    d5.text((45, 292), "Error Taxonomy Breakdown:", font=font_bold, fill=text_white)
    d5.text(
        (45, 318),
        "Correct: 9 (60%)  |  Reasoning Error: 3 (20%)  |  Wrong Script: 1 (6.7%)",
        font=font_sm,
        fill=text_yellow,
    )
    d5.text((45, 340), "Refusal: 1 (6.7%) |  Language Drift: 1 (6.7%)", font=font_sm, fill=text_dim)

    # Footer
    d5.text(
        (30, 390),
        "✓ Saved Checkpoint: results/run_20260917_urdu_qa/scores.json",
        font=font_sm,
        fill=text_green,
    )
    d5.text(
        (30, 412), "✓ Interactive HTML Report: results/report.html", font=font_sm, fill=text_cyan
    )
    d5.text(
        (30, 440), "researcher@node:~/urdu-eval$ urdu-eval leaderboard", font=font, fill=text_white
    )
    d5.rectangle([455, 440, 465, 457], fill=text_green)

    # Duplicate frames with varying duration for smooth video feel
    frames.extend([im1] * 2)
    frames.extend([im2] * 2)
    frames.extend([im3] * 2)
    frames.extend([im4] * 2)
    frames.extend([im5] * 8)

    assets_dir = Path("assets")
    assets_dir.mkdir(parents=True, exist_ok=True)
    out_path = assets_dir / "demo.gif"
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=500,
        loop=0,
        optimize=True,
    )
    print(f"Generated {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    create_demo_gif()
