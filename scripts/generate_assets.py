"""Generate high-resolution PNG assets and animated demo GIF for UrduEval."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

try:
    import arabic_reshaper
    from bidi.algorithm import get_display

    _reshaper = arabic_reshaper.ArabicReshaper(
        configuration={
            "language": "Urdu",
            "support_ligatures": True,
            "delete_harakat": False,
        }
    )

    def shape_urdu(text: str) -> str:
        """Reshape and apply bidi ordering to Urdu text for authentic cursive rendering."""
        return get_display(_reshaper.reshape(text))

except Exception:

    def shape_urdu(text: str) -> str:
        """Fallback when arabic_reshaper or python-bidi is not installed."""
        return text


def get_fonts():
    """Load system or default fonts gracefully."""
    try:
        font_sm = ImageFont.truetype("consola.ttf", 13)
        font_md = ImageFont.truetype("consola.ttf", 15)
        font_bold = ImageFont.truetype("consolab.ttf", 16)
        font_title = ImageFont.truetype("segoeuib.ttf", 36)
        font_lg = ImageFont.truetype("segoeuib.ttf", 20)
        font_badge = ImageFont.truetype("segoeui.ttf", 13)
    except Exception:
        try:
            font_sm = ImageFont.truetype("arial.ttf", 13)
            font_md = ImageFont.truetype("arial.ttf", 15)
            font_bold = ImageFont.truetype("arialbd.ttf", 16)
            font_title = ImageFont.truetype("arialbd.ttf", 36)
            font_lg = ImageFont.truetype("arialbd.ttf", 20)
            font_badge = ImageFont.truetype("arial.ttf", 13)
        except Exception:
            f = ImageFont.load_default()
            font_sm = font_md = font_bold = font_title = font_lg = font_badge = f

    # High-quality Urdu fonts (Tahoma provides full Urdu glyph ligature rendering on Windows)
    try:
        font_urdu_title = ImageFont.truetype("tahomabd.ttf", 34)
        font_urdu_md = ImageFont.truetype("tahoma.ttf", 15)
        font_urdu_bold = ImageFont.truetype("tahomabd.ttf", 15)
    except Exception:
        try:
            font_urdu_title = ImageFont.truetype("C:\\Windows\\Fonts\\tahomabd.ttf", 34)
            font_urdu_md = ImageFont.truetype("C:\\Windows\\Fonts\\tahoma.ttf", 15)
            font_urdu_bold = ImageFont.truetype("C:\\Windows\\Fonts\\tahomabd.ttf", 15)
        except Exception:
            font_urdu_title = font_title
            font_urdu_md = font_md
            font_urdu_bold = font_bold

    return {
        "sm": font_sm,
        "md": font_md,
        "bold": font_bold,
        "title": font_title,
        "lg": font_lg,
        "badge": font_badge,
        "urdu_title": font_urdu_title,
        "urdu_md": font_urdu_md,
        "urdu_bold": font_urdu_bold,
    }


def generate_banner_png(output_path: Path) -> None:
    """Generate high-res 1200x380 banner PNG with properly shaped Urdu script."""
    width, height = 1200, 380
    im = Image.new("RGB", (width, height), color=(8, 13, 23))
    draw = ImageDraw.Draw(im)
    fonts = get_fonts()

    # Outer border
    draw.rounded_rectangle(
        [10, 10, width - 10, height - 10],
        radius=14,
        fill=(11, 19, 36),
        outline=(30, 41, 59),
        width=2,
    )
    # Accent top border
    draw.rectangle([10, 10, width - 10, 15], fill=(16, 185, 129))

    # Decorative tech lines
    draw.line([50, 60, width - 50, 60], fill=(24, 34, 53), width=1)
    draw.line([50, 310, width - 50, 310], fill=(24, 34, 53), width=1)

    # Brand Icon Box
    draw.rounded_rectangle(
        [50, 80, 130, 160], radius=16, fill=(15, 23, 42), outline=(51, 65, 85), width=2
    )
    draw.text((70, 95), "UE", font=fonts["title"], fill=(52, 211, 153))

    # Titles (Latin brand name + properly shaped connected Urdu script)
    draw.text((150, 85), "UrduEval", font=fonts["title"], fill=(255, 255, 255))
    ue_len = int(draw.textlength("UrduEval", font=fonts["title"]))
    draw.text((150 + ue_len + 30, 84), shape_urdu("اردو ایویل"), font=fonts["urdu_title"], fill=(52, 211, 153))

    # Tagline
    draw.text(
        (150, 135),
        "The open, unified evaluation harness for Urdu & Roman Urdu AI.",
        font=fonts["lg"],
        fill=(148, 163, 184),
    )

    # Badges
    badges = [
        ("Urdu Script + Roman Urdu", (16, 185, 129)),
        ("Local Ollama & Cloud APIs", (6, 182, 212)),
        ("Custom JSONL Datasets", (99, 102, 241)),
        ("Interactive HTML Reports", (245, 158, 11)),
    ]
    bx = 150
    for text, dot_color in badges:
        bw = len(text) * 8 + 40
        draw.rounded_rectangle(
            [bx, 175, bx + bw, 210], radius=8, fill=(30, 41, 59), outline=(51, 65, 85), width=1
        )
        draw.ellipse([bx + 12, 188, bx + 22, 198], fill=dot_color)
        draw.text((bx + 30, 184), text, font=fonts["badge"], fill=(226, 232, 240))
        bx += bw + 15

    # Bottom KPI strip
    draw.rounded_rectangle(
        [50, 240, width - 50, 300], radius=10, fill=(10, 15, 29), outline=(30, 41, 59), width=1
    )
    kpis = [
        ("6+ Core Tasks", "QA, Reasoning, Trans, Roman"),
        ("7+ Metrics", "EM, F1, BLEU, chrF++, Judge"),
        ("0 Fabrications", "Honest dev benchmarks"),
        ("100% Private", "Local-first, zero telemetry"),
    ]
    kx = 80
    for title, subtitle in kpis:
        draw.text((kx, 250), title, font=fonts["bold"], fill=(56, 189, 248))
        draw.text((kx, 272), subtitle, font=fonts["sm"], fill=(100, 116, 139))
        kx += 275

    im.save(output_path, "PNG", optimize=True)
    print(f"Created {output_path} ({output_path.stat().st_size} bytes)")


def generate_architecture_png(output_path: Path) -> None:
    """Generate high-res 960x460 architecture diagram PNG."""
    width, height = 960, 460
    im = Image.new("RGB", (width, height), color=(10, 14, 26))
    draw = ImageDraw.Draw(im)
    fonts = get_fonts()

    draw.rounded_rectangle(
        [10, 10, width - 10, height - 10],
        radius=14,
        fill=(10, 14, 26),
        outline=(30, 41, 59),
        width=2,
    )
    # Header
    draw.text(
        (30, 30),
        "UrduEval End-to-End Evaluation Architecture",
        font=fonts["lg"],
        fill=(255, 255, 255),
    )
    draw.text(
        (30, 56),
        "Modular, extensible, local-first evaluation pipeline for Urdu & Roman Urdu LLMs",
        font=fonts["sm"],
        fill=(100, 116, 139),
    )

    cards = [
        (
            "1. Model Providers",
            "Local & Cloud Inference",
            [
                "• Ollama (Local & Free)",
                "• OpenAI & Claude",
                "• OpenRouter & HF",
                "• Custom HTTP / Mock",
            ],
            (6, 182, 212),
            30,
            90,
            270,
            160,
        ),
        (
            "2. Benchmark Registry",
            "Standard & Custom Suites",
            [
                "• urdu-qa & reasoning",
                "• urdu-translation & summary",
                "• urdu-roman & urdu-mmlu",
                "• Custom streaming *.jsonl",
            ],
            (16, 185, 129),
            340,
            90,
            270,
            160,
        ),
        (
            "3. Runner & Resilience",
            "Parallel & Fault-Tolerant",
            [
                "• ThreadPool Concurrency",
                "• SQLite Response Cache",
                "• Atomic Checkpoint/Resume",
                "• Exponential Backoff Retry",
            ],
            (99, 102, 241),
            650,
            90,
            270,
            160,
        ),
        (
            "4. Normalization Engine",
            "Linguistically Safe Profiles",
            [
                "• Raw Profile (Pure String)",
                "• Conservative (Kaf/Yeh/Aerab)",
                "• Standard (Digits/Punct)",
                "• Roman Urdu Phonetic Map",
            ],
            (236, 72, 153),
            30,
            270,
            270,
            160,
        ),
        (
            "5. Multi-Metric Engine",
            "String, N-Gram & Judge",
            [
                "• Exact Match & Token F1",
                "• BLEU-4, chrF & chrF++",
                "• ROUGE-L Overlap",
                "• Structured LLM Judge",
            ],
            (245, 158, 11),
            340,
            270,
            270,
            160,
        ),
        (
            "6. Analysis & Reports",
            "Diagnostics & Transparency",
            [
                "• Task-Specific Error Taxonomy",
                "• 95% Confidence Intervals",
                "• Raw vs Normalized Metrics",
                "• Standalone HTML & Leaderboard",
            ],
            (16, 185, 129),
            650,
            270,
            270,
            160,
        ),
    ]

    for title, subtitle, points, accent_color, x, y, w, h in cards:
        draw.rounded_rectangle(
            [x, y, x + w, y + h], radius=10, fill=(15, 23, 42), outline=(51, 65, 85), width=1
        )
        draw.rectangle([x, y, x + w, y + 4], fill=accent_color)
        draw.text((x + 16, y + 16), title, font=fonts["bold"], fill=(248, 250, 252))
        draw.text((x + 16, y + 36), subtitle, font=fonts["sm"], fill=(148, 163, 184))
        py = y + 60
        for pt in points:
            draw.text((x + 16, py), pt, font=fonts["sm"], fill=(226, 232, 240))
            py += 22

    im.save(output_path, "PNG", optimize=True)
    print(f"Created {output_path} ({output_path.stat().st_size} bytes)")


def generate_report_preview_png(output_path: Path) -> None:
    """Generate high-res 920x520 report preview PNG with authentic connected Urdu script."""
    width, height = 920, 520
    im = Image.new("RGB", (width, height), color=(11, 15, 25))
    draw = ImageDraw.Draw(im)
    fonts = get_fonts()

    draw.rounded_rectangle(
        [10, 10, width - 10, height - 10],
        radius=12,
        fill=(11, 15, 25),
        outline=(51, 65, 85),
        width=1,
    )
    # Browser bar
    draw.rounded_rectangle([10, 10, width - 10, 50], radius=10, fill=(30, 41, 59))
    draw.ellipse([25, 25, 37, 37], fill=(239, 68, 68))
    draw.ellipse([45, 25, 57, 37], fill=(245, 158, 11))
    draw.ellipse([65, 25, 77, 37], fill=(16, 185, 129))
    draw.rounded_rectangle([180, 20, 740, 42], radius=6, fill=(15, 23, 42), outline=(51, 65, 85))
    draw.text(
        (320, 24),
        "file:///results/report.html — UrduEval Standalone Interactive Report",
        font=fonts["sm"],
        fill=(148, 163, 184),
    )

    # Header
    draw.text((30, 68), "UrduEval Run Diagnostics Report", font=fonts["lg"], fill=(255, 255, 255))
    draw.text(
        (30, 94),
        "Run ID: run_20260917_urdu_qa | Benchmark: urdu-qa (v0.1.0) | Model: llama3.1 (ollama)",
        font=fonts["sm"],
        fill=(100, 116, 139),
    )

    # KPI cards
    kpis = [
        ("Exact Match", "60.0%", "[35.7% - 82.7%]", (16, 185, 129)),
        ("Token F1", "78.4%", "[58.2% - 91.1%]", (56, 189, 248)),
        ("chrF++", "74.2%", "[52.8% - 88.0%]", (245, 158, 11)),
        ("Mean Latency", "142 ms", "p95: 198 ms", (226, 232, 240)),
    ]
    kx = 30
    for title, val, ci, color in kpis:
        draw.rounded_rectangle(
            [kx, 125, kx + 200, 195], radius=8, fill=(19, 28, 49), outline=(30, 41, 59)
        )
        draw.text((kx + 16, 135), val, font=fonts["lg"], fill=color)
        draw.text((kx + 16, 160), title, font=fonts["bold"], fill=(226, 232, 240))
        draw.text((kx + 16, 178), ci, font=fonts["sm"], fill=(100, 116, 139))
        kx += 220

    # Filter Toolbar
    draw.rounded_rectangle([30, 215, 340, 250], radius=6, fill=(15, 23, 42), outline=(51, 65, 85))
    draw.text(
        (45, 226), "Search samples by prompt, answer, or ID...", font=fonts["sm"], fill=(100, 116, 139)
    )

    filters = [
        ("All (15)", (16, 185, 129)),
        ("Correct (9)", (30, 41, 59)),
        ("Reasoning Error (3)", (30, 41, 59)),
        ("Refusal (1)", (30, 41, 59)),
    ]
    fx = 360
    for name, fbg in filters:
        draw.rounded_rectangle([fx, 215, fx + 110, 250], radius=6, fill=fbg, outline=(51, 65, 85))
        draw.text((fx + 16, 226), name, font=fonts["sm"], fill=(255, 255, 255))
        fx += 120

    # Sample Card 1
    draw.rounded_rectangle(
        [30, 270, width - 30, 370], radius=8, fill=(17, 24, 39), outline=(30, 41, 59)
    )
    draw.rectangle([30, 270, 34, 370], fill=(16, 185, 129))
    draw.text(
        (45, 285),
        "#SAMPLE: urdu-qa-001   [CORRECT]   EM: 1.0 | F1: 1.0 | Latency: 128ms",
        font=fonts["bold"],
        fill=(52, 211, 153),
    )

    # Prompt 1
    px = 45
    draw.text((px, 312), "Input Prompt:  ", font=fonts["bold"], fill=(148, 163, 184))
    px += int(draw.textlength("Input Prompt:  ", font=fonts["bold"]))
    draw.text(
        (px, 311),
        shape_urdu("پاکستان کا دارالحکومت کون سا شہر ہے؟"),
        font=fonts["urdu_md"],
        fill=(226, 232, 240),
    )

    # Output 1
    ox = 45
    draw.text((ox, 338), "Model Output:  ", font=fonts["bold"], fill=(148, 163, 184))
    ox += int(draw.textlength("Model Output:  ", font=fonts["bold"]))

    u_out1 = shape_urdu("اسلام آباد")
    draw.text((ox, 337), u_out1, font=fonts["urdu_bold"], fill=(52, 211, 153))
    ox += int(draw.textlength(u_out1, font=fonts["urdu_bold"])) + 25

    ref_str = "|   Reference:  "
    draw.text((ox, 338), ref_str, font=fonts["md"], fill=(100, 116, 139))
    ox += int(draw.textlength(ref_str, font=fonts["md"]))

    u_ref1 = shape_urdu("اسلام آباد")
    draw.text((ox, 337), u_ref1, font=fonts["urdu_md"], fill=(148, 163, 184))

    # Sample Card 2
    draw.rounded_rectangle(
        [30, 390, width - 30, 490], radius=8, fill=(17, 24, 39), outline=(30, 41, 59)
    )
    draw.rectangle([30, 390, 34, 490], fill=(245, 158, 11))
    draw.text(
        (45, 405),
        "#SAMPLE: urdu-reasoning-004   [REASONING ERROR]   EM: 0.0 | F1: 0.33 | Latency: 186ms",
        font=fonts["bold"],
        fill=(251, 191, 36),
    )

    # Prompt 2
    px2 = 45
    draw.text((px2, 432), "Input Prompt:  ", font=fonts["bold"], fill=(148, 163, 184))
    px2 += int(draw.textlength("Input Prompt:  ", font=fonts["bold"]))
    draw.text(
        (px2, 431),
        shape_urdu("اگر احمد کے پاس ۵ سیب ہیں اور وہ ۲ کھا لیتا ہے، تو کتنے باقی ہیں؟"),
        font=fonts["urdu_md"],
        fill=(226, 232, 240),
    )

    # Output 2
    ox2 = 45
    draw.text((ox2, 458), "Model Output:  ", font=fonts["bold"], fill=(148, 163, 184))
    ox2 += int(draw.textlength("Model Output:  ", font=fonts["bold"]))

    u_out2 = shape_urdu("احمد کے پاس ۲ سیب باقی ہیں")
    draw.text((ox2, 457), u_out2, font=fonts["urdu_bold"], fill=(248, 113, 113))
    ox2 += int(draw.textlength(u_out2, font=fonts["urdu_bold"])) + 25

    exp_str = "|   Expected:  "
    draw.text((ox2, 458), exp_str, font=fonts["md"], fill=(100, 116, 139))
    ox2 += int(draw.textlength(exp_str, font=fonts["md"]))

    u_exp2 = shape_urdu("۳ سیب")
    draw.text((ox2, 457), u_exp2, font=fonts["urdu_md"], fill=(148, 163, 184))

    im.save(output_path, "PNG", optimize=True)
    print(f"Created {output_path} ({output_path.stat().st_size} bytes)")


def generate_rich_animated_gif(output_path: Path) -> None:
    """Generate a genuinely dynamic, 28-frame animated demo GIF with visible progression."""
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

    try:
        font = ImageFont.truetype("consola.ttf", 15)
        font_bold = ImageFont.truetype("consolab.ttf", 15)
        font_sm = ImageFont.truetype("consola.ttf", 13)
    except Exception:
        font = font_bold = font_sm = ImageFont.load_default()

    def base_window():
        im = Image.new("RGB", (width, height), color=bg_color)
        draw = ImageDraw.Draw(im)
        draw.rounded_rectangle(
            [10, 10, width - 10, height - 10],
            radius=10,
            fill=bg_color,
            outline=border_color,
            width=1,
        )
        draw.rounded_rectangle([10, 10, width - 10, 48], radius=10, fill=header_color)
        draw.rectangle([10, 38, width - 10, 48], fill=header_color)
        draw.line([10, 48, width - 10, 48], fill=border_color, width=1)
        draw.ellipse([26, 23, 38, 35], fill=(239, 68, 68))
        draw.ellipse([46, 23, 58, 35], fill=(245, 158, 11))
        draw.ellipse([66, 23, 78, 35], fill=(16, 185, 129))
        draw.text((width // 2 - 80, 21), "urdu-eval — bash — 860x480", font=font_sm, fill=text_dim)
        return im, draw

    frames = []

    # Sequence 1: Typing command
    cmd_text = "urdu-eval run --provider ollama --model llama3.1 --benchmark urdu-qa"
    for i in range(10, len(cmd_text) + 1, 6):
        im, d = base_window()
        d.text((30, 70), "researcher@ai:~/urdu-eval$ ", font=font_bold, fill=text_green)
        typed = cmd_text[:i]
        d.text((260, 70), typed, font=font, fill=text_white)
        tw = len(typed) * 9
        d.rectangle([260 + tw, 70, 270 + tw, 87], fill=text_green)
        frames.append(im)

    # Sequence 2: Banner initialization
    im, d = base_window()
    d.text((30, 70), f"researcher@ai:~/urdu-eval$ {cmd_text}", font=font, fill=text_dim)
    d.rounded_rectangle([30, 100, width - 30, 155], radius=6, fill=card_color, outline=border_color)
    d.text((45, 112), "UrduEval v0.1.2", font=font_bold, fill=text_green)
    d.text(
        (45, 132),
        "Provider: ollama  |  Model: llama3.1  |  Benchmark: urdu-qa (15 items)",
        font=font_sm,
        fill=text_cyan,
    )
    frames.extend([im] * 2)

    # Sequence 3: Progressive Evaluation Progress Bar (20%, 40%, 65%, 85%, 100%)
    for pct, cur, speed in [
        (20, 3, "6.8 it/s"),
        (40, 6, "6.5 it/s"),
        (65, 10, "6.4 it/s"),
        (85, 13, "6.2 it/s"),
        (100, 15, "6.3 it/s"),
    ]:
        im, d = base_window()
        d.text((30, 70), f"researcher@ai:~/urdu-eval$ {cmd_text}", font=font, fill=text_dim)
        d.rounded_rectangle(
            [30, 100, width - 30, 155], radius=6, fill=card_color, outline=border_color
        )
        d.text((45, 112), "UrduEval v0.1.2", font=font_bold, fill=text_green)
        d.text(
            (45, 132),
            "Provider: ollama  |  Model: llama3.1  |  Benchmark: urdu-qa (15 items)",
            font=font_sm,
            fill=text_cyan,
        )

        # Progress bar
        filled_len = int(pct * 35 / 100)
        bar_str = "━" * filled_len + " " * (35 - filled_len)
        bar_color = text_yellow if pct < 100 else text_green
        d.text(
            (30, 180),
            f"Evaluating: {pct}% [{bar_str}] {cur}/15 [00:02<00:00, {speed}]",
            font=font,
            fill=bar_color,
        )
        frames.extend([im] * 2)

    # Sequence 4: Complete results table + 95% Confidence Intervals + Error Taxonomy
    im, d = base_window()
    d.text((30, 65), f"researcher@ai:~/urdu-eval$ {cmd_text}", font=font_sm, fill=text_dim)

    # Metrics Table
    d.rounded_rectangle([30, 95, width - 30, 260], radius=6, fill=card_color, outline=border_color)
    d.text((45, 108), "METRIC", font=font_bold, fill=text_white)
    d.text((220, 108), "SCORE", font=font_bold, fill=text_white)
    d.text((360, 108), "95% CONFIDENCE", font=font_bold, fill=text_white)
    d.text((580, 108), "STATUS", font=font_bold, fill=text_white)
    d.line([30, 130, width - 30, 130], fill=border_color)

    d.text((45, 142), "Exact Match", font=font, fill=text_white)
    d.text((220, 142), "60.0%", font=font_bold, fill=text_green)
    d.text((360, 142), "[35.7% - 82.7%]", font=font_sm, fill=text_dim)
    d.text((580, 142), "✓ Evaluated", font=font_sm, fill=text_green)

    d.text((45, 170), "Token F1", font=font, fill=text_white)
    d.text((220, 170), "78.4%", font=font_bold, fill=text_cyan)
    d.text((360, 170), "[58.2% - 91.1%]", font=font_sm, fill=text_dim)
    d.text((580, 170), "✓ Evaluated", font=font_sm, fill=text_green)

    d.text((45, 198), "chrF++", font=font, fill=text_white)
    d.text((220, 198), "74.2%", font=font_bold, fill=text_yellow)
    d.text((360, 198), "[52.8% - 88.0%]", font=font_sm, fill=text_dim)
    d.text((580, 198), "✓ Evaluated", font=font_sm, fill=text_green)

    d.text((45, 226), "Latency (mean)", font=font, fill=text_white)
    d.text((220, 226), "142 ms", font=font, fill=text_dim)
    d.text((360, 226), "p95: 198 ms", font=font_sm, fill=text_dim)
    d.text((580, 226), "Fast (Local)", font=font_sm, fill=text_cyan)

    # Error Taxonomy
    d.rounded_rectangle([30, 275, width - 30, 355], radius=6, fill=card_color, outline=border_color)
    d.text((45, 286), "Diagnostic Failure Taxonomy:", font=font_bold, fill=text_white)
    d.text(
        (45, 310),
        "Correct: 9 (60.0%)  |  Reasoning Error: 3 (20.0%)  |  Script Slip: 1 (6.7%)",
        font=font_sm,
        fill=text_yellow,
    )
    d.text(
        (45, 330),
        "Model Refusal: 1 (6.7%)  |  Translation Drift: 1 (6.7%)",
        font=font_sm,
        fill=text_dim,
    )

    # Footers
    d.text(
        (30, 375),
        "✓ Saved Checkpoint: results/run_20260917_urdu_qa/scores.json",
        font=font_sm,
        fill=text_green,
    )
    d.text(
        (30, 395), "✓ Interactive HTML Report: results/report.html", font=font_sm, fill=text_cyan
    )
    d.text(
        (30, 425), "researcher@ai:~/urdu-eval$ urdu-eval leaderboard", font=font, fill=text_white
    )
    d.rectangle([460, 425, 470, 442], fill=text_green)

    frames.extend([im] * 8)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=350,
        loop=0,
        optimize=True,
    )
    print(f"Created animated {output_path} ({output_path.stat().st_size} bytes)")


def main():
    assets_dir = Path("assets")
    assets_dir.mkdir(parents=True, exist_ok=True)
    generate_banner_png(assets_dir / "banner.png")
    generate_architecture_png(assets_dir / "architecture.png")
    generate_report_preview_png(assets_dir / "report_preview.png")
    generate_rich_animated_gif(assets_dir / "demo.gif")


if __name__ == "__main__":
    main()
