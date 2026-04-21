#!/usr/bin/env python3
from __future__ import annotations

import base64
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET


W = 1920
H = 1080
TOPBAR_H = 120
LINE_H = 6
FOOTER_H = 4
MARGIN_X = 110
CONTENT_TOP = 170

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "proposal-defense-classic.drawio"

BLUE = "#1F497D"
GOLD = "#C9A84C"
BG = "#F7F8FA"
INK = "#1A1A2E"
MUTED = "#4A5568"
LINE = "#E2E8F0"
WHITE = "#FFFFFF"
CARD = "#FFFFFF"
CARD_ALT = "#F1F5F9"
DARK = "#0F172A"

DATE_TEXT = "2026 年 4 月"
TITLE_TEXT = "基于部件与结构条件约束的\n方块字字形生成方法研究"


class IdGen:
    def __init__(self) -> None:
        self._value = 2

    def next(self) -> str:
        current = self._value
        self._value += 1
        return str(current)


def image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def add_cell(root: ET.Element, cell_id: str, value: str, style: str, x: float, y: float, width: float, height: float) -> None:
    cell = ET.SubElement(
        root,
        "mxCell",
        {
            "id": cell_id,
            "value": value,
            "style": style,
            "vertex": "1",
            "parent": "1",
        },
    )
    ET.SubElement(
        cell,
        "mxGeometry",
        {
            "x": f"{x:.1f}",
            "y": f"{y:.1f}",
            "width": f"{width:.1f}",
            "height": f"{height:.1f}",
            "as": "geometry",
        },
    )


def rect_style(fill: str, stroke: str = "none", rounded: bool = False, dashed: bool = False) -> str:
    rounded_flag = "1" if rounded else "0"
    dashed_flag = "1" if dashed else "0"
    extra = "arcSize=10;" if rounded else ""
    return (
        f"rounded={rounded_flag};whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
        f"dashed={dashed_flag};{extra}"
    )


def text_style(
    *,
    size: int,
    color: str = INK,
    bold: bool = False,
    align: str = "left",
    valign: str = "top",
    family: str = "微软雅黑",
) -> str:
    font_style = "1" if bold else "0"
    return (
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        f"align={align};verticalAlign={valign};fontSize={size};fontStyle={font_style};"
        f"fontColor={color};fontFamily={family};spacingTop=2;spacingBottom=2;"
    )


def image_style(data_uri: str) -> str:
    return f"shape=image;html=1;imageAspect=0;aspect=fixed;verticalLabelPosition=bottom;verticalAlign=top;image={data_uri};"


def section_header(root: ET.Element, ids: IdGen, title: str) -> None:
    add_cell(root, ids.next(), "", rect_style(BG), 0, 0, W, H)
    add_cell(root, ids.next(), "", rect_style(BLUE), 0, 0, W, TOPBAR_H)
    add_cell(root, ids.next(), "", rect_style(GOLD), 0, TOPBAR_H, W, LINE_H)
    add_cell(root, ids.next(), title, text_style(size=30, color=WHITE, bold=True, valign="middle"), 0, 0, W, TOPBAR_H)
    add_cell(root, ids.next(), "", rect_style(LINE), 0, 1040, W, FOOTER_H)
    add_cell(root, ids.next(), DATE_TEXT, text_style(size=14, color=MUTED, align="right", valign="middle"), 1560, 1042, 270, 30)


def title_with_badge(root: ET.Element, ids: IdGen, title: str, x: float, y: float, width: float = 780) -> None:
    add_cell(root, ids.next(), "", rect_style(GOLD), x, y + 2, 10, 38)
    add_cell(root, ids.next(), title, text_style(size=25, bold=True, valign="middle"), x + 28, y, width, 44)


def add_card(root: ET.Element, ids: IdGen, x: float, y: float, width: float, height: float, title: str, body: str, *, fill: str = CARD) -> None:
    add_cell(root, ids.next(), "", rect_style(fill, LINE, rounded=True), x, y, width, height)
    add_cell(root, ids.next(), title, text_style(size=20, bold=True), x + 24, y + 18, width - 48, 36)
    add_cell(root, ids.next(), body, text_style(size=18, color=INK), x + 24, y + 62, width - 48, height - 84)


def add_small_label(root: ET.Element, ids: IdGen, text: str, x: float, y: float, width: float, *, fill: str = GOLD, color: str = INK) -> None:
    add_cell(root, ids.next(), "", rect_style(fill, fill, rounded=True), x, y, width, 34)
    add_cell(root, ids.next(), text, text_style(size=16, color=color, bold=True, align="center", valign="middle"), x, y, width, 34)


def add_image(root: ET.Element, ids: IdGen, image_path: Path, x: float, y: float, width: float, height: float, caption: str | None = None) -> None:
    add_cell(root, ids.next(), "", rect_style(DARK, LINE, rounded=True), x, y, width, height)
    add_cell(root, ids.next(), "", image_style(image_data_uri(image_path)), x + 14, y + 14, width - 28, height - 28 - (38 if caption else 0))
    if caption:
        add_cell(root, ids.next(), caption, text_style(size=16, color=WHITE, align="center", valign="middle"), x + 10, y + height - 40, width - 20, 24)


def add_arrow_text(root: ET.Element, ids: IdGen, text: str, x: float, y: float) -> None:
    add_cell(root, ids.next(), text, text_style(size=36, color=BLUE, bold=True, align="center", valign="middle"), x, y, 90, 80)


def deck_slides() -> list[dict]:
    sample_dir = ROOT.parent.parent.parent / "data" / "generated" / "sample_dataset"
    image_dir = sample_dir / "images" / "Hiragino_Sans_GB"
    component_dir = sample_dir / "components" / "Hiragino_Sans_GB"

    return [
        {
            "name": "封面",
            "builder": lambda root, ids: build_cover(root, ids, image_dir),
        },
        {
            "name": "目录",
            "builder": build_toc,
        },
        {
            "name": "研究背景",
            "builder": lambda root, ids: build_background(root, ids, image_dir),
        },
        {
            "name": "研究意义",
            "builder": build_significance,
        },
        {
            "name": "研究现状",
            "builder": build_related_work,
        },
        {
            "name": "任务定义",
            "builder": lambda root, ids: build_task_definition(root, ids, image_dir, component_dir),
        },
        {
            "name": "总体方案",
            "builder": build_overall_plan,
        },
        {
            "name": "核心方法",
            "builder": build_method_design,
        },
        {
            "name": "关键问题",
            "builder": build_key_questions,
        },
        {
            "name": "创新与可行性",
            "builder": build_innovation_feasibility,
        },
        {
            "name": "研究计划",
            "builder": build_schedule,
        },
        {
            "name": "致谢",
            "builder": build_closing,
        },
    ]


def build_cover(root: ET.Element, ids: IdGen, image_dir: Path) -> None:
    add_cell(root, ids.next(), "", rect_style(BG), 0, 0, W, H)
    add_cell(root, ids.next(), "", rect_style(BLUE), 0, 0, W, 120)
    add_cell(root, ids.next(), "", rect_style(GOLD), 0, 120, W, 6)
    add_cell(root, ids.next(), "广西民族大学硕士学位论文开题答辩", text_style(size=24, color=WHITE, bold=True, align="center", valign="middle"), 0, 28, W, 42)
    add_cell(root, ids.next(), TITLE_TEXT, text_style(size=34, color=INK, bold=True, valign="middle"), 150, 250, 980, 150)
    add_cell(root, ids.next(), "人工智能学院\n计算机科学与技术", text_style(size=24, color=MUTED), 150, 440, 400, 90)
    add_cell(root, ids.next(), "汇报人：黄子涵\n指导教师：白凤波\n时间：2026 年 4 月", text_style(size=22, color=INK), 150, 610, 420, 130)
    add_cell(root, ids.next(), "", rect_style(CARD, LINE, rounded=True), 1250, 210, 510, 610)
    add_cell(root, ids.next(), "研究对象示意", text_style(size=22, bold=True, align="center", valign="middle"), 1250, 238, 510, 40)
    add_image(root, ids, image_dir / "狗_glyph.png", 1325, 320, 150, 190, "左右结构")
    add_image(root, ids, image_dir / "困_glyph.png", 1490, 320, 150, 190, "全包围结构")
    add_image(root, ids, image_dir / "间_glyph.png", 1325, 540, 150, 190, "半包围结构")
    add_image(root, ids, image_dir / "明_glyph.png", 1490, 540, 150, 190, "组合生成")


def build_toc(root: ET.Element, ids: IdGen) -> None:
    section_header(root, ids, "目录")
    add_cell(root, ids.next(), "CONTENTS", text_style(size=58, color=BLUE, bold=True, align="center", valign="middle"), 1180, 220, 560, 80)
    items = [
        "01 研究背景与问题提出",
        "02 研究意义与应用场景",
        "03 国内外研究现状与研究空缺",
        "04 任务定义与研究方案",
        "05 方法设计、创新点与可行性",
        "06 研究计划与预期成果",
    ]
    y = 250
    for item in items:
        add_cell(root, ids.next(), "", rect_style(GOLD), 170, y + 12, 8, 30)
        add_cell(root, ids.next(), item, text_style(size=24, bold=True, valign="middle"), 200, y, 820, 54)
        y += 88
    add_card(
        root,
        ids,
        1130,
        360,
        590,
        290,
        "汇报逻辑",
        "从问题出发，而不是按表单逐项复述。\n先说明为什么这个题目值得做，再交代现有研究的不足，随后收束到任务定义、研究方案和后续计划。",
        fill=CARD_ALT,
    )


def build_background(root: ET.Element, ids: IdGen, image_dir: Path) -> None:
    section_header(root, ids, "01 研究背景与问题提出")
    title_with_badge(root, ids, "为什么把“字形生成”收束到部件和结构条件上？", 120, 182, 920)
    add_card(
        root,
        ids,
        120,
        260,
        910,
        530,
        "现实需求",
        "现代常用字已有成熟字库，但面对生僻字、异体字和未登录字符时，传统补字流程往往成本高、周期长，而且高度依赖人工经验。\n\n在古籍整理、地方志数字化、少数民族文字信息化等场景中，字符数量不一定大，但对结构正确性和快速试排的需求很强。",
    )
    add_card(
        root,
        ids,
        120,
        820,
        910,
        140,
        "问题提出",
        "与其把字当作普通图像去生成，不如直接利用方块字的构形规律，把部件和结构关系作为显式条件纳入模型。",
        fill=CARD_ALT,
    )
    add_cell(root, ids.next(), "", rect_style(CARD, LINE, rounded=True), 1080, 220, 710, 740)
    add_cell(root, ids.next(), "现有样例字形", text_style(size=22, bold=True, align="center", valign="middle"), 1080, 248, 710, 40)
    positions = [
        (1145, 330, "狗_glyph.png", "狗"),
        (1445, 330, "困_glyph.png", "困"),
        (1145, 610, "间_glyph.png", "间"),
        (1445, 610, "闪_glyph.png", "闪"),
    ]
    for x, y, filename, caption in positions:
        add_image(root, ids, image_dir / filename, x, y, 230, 220, caption)


def build_significance(root: ET.Element, ids: IdGen) -> None:
    section_header(root, ids, "02 研究意义与应用场景")
    title_with_badge(root, ids, "方法研究先行，应用价值作为自然外延", 120, 180, 900)
    add_card(root, ids, 120, 300, 520, 420, "字库补字与字体设计", "在目标字符缺失时，为字体设计和快速试排提供结构正确的候选字形，而不是完全依赖人工描绘。")
    add_card(root, ids, 700, 300, 520, 420, "古籍整理与异体字数字化", "许多字符没有稳定编码入口，但研究者通常能给出构形描述或局部部件图像，这使统一条件接口有了真实使用场景。")
    add_card(root, ids, 1280, 300, 520, 420, "低资源字符处理", "若模型能够在有限样本下维持结构约束，就有机会把现代汉字主任务平滑延伸到 OOV 字符、古壮字等边缘场景。")
    add_card(
        root,
        ids,
        200,
        780,
        1520,
        150,
        "研究定位",
        "本课题首先讨论的是“部件与结构条件如何驱动整字生成”，不是把系统包装成已经成熟落地的工程产品。",
        fill=CARD_ALT,
    )


def build_related_work(root: ET.Element, ids: IdGen) -> None:
    section_header(root, ids, "03 国内外研究现状与研究空缺")
    title_with_badge(root, ids, "相关研究已经很多，但尚未自然汇聚成统一任务框架", 120, 180, 1040)
    add_card(root, ids, 100, 300, 540, 430, "部件与结构表征", "零样本汉字识别、显式偏旁编码、IDS 对齐、formation tree 等工作说明：汉字可以被拆解，也值得以结构化方式表示。")
    add_card(root, ids, 690, 300, 540, 430, "字体迁移与字形生成", "GAN、Transformer 和扩散模型显著提升了字形生成质量，但多数研究默认目标字符身份已知，关注的是风格迁移而非“部件到整字”的组合生成。")
    add_card(root, ids, 1280, 300, 540, 430, "扩散模型与稀有字符处理", "近年的相关工作证明生成模型能够服务字体合成和低资源字符任务。\n\n但在这些研究中，结构约束通常还不是最核心的控制信号。")
    add_cell(root, ids.next(), "", rect_style(BLUE, BLUE, rounded=True), 140, 790, 1640, 145)
    add_cell(root, ids.next(), "研究空缺：现有工作分别回答了“如何表示部件”“如何迁移字体风格”“如何提升图像生成质量”，\n但还没有形成成熟的“部件 + 结构条件 -> 整字图像”的统一研究框架。", text_style(size=24, color=WHITE, bold=True, align="center", valign="middle"), 170, 820, 1580, 90)


def build_task_definition(root: ET.Element, ids: IdGen, image_dir: Path, component_dir: Path) -> None:
    section_header(root, ids, "04 任务定义与研究目标")
    title_with_badge(root, ids, "输入部件和结构条件，输出完整字形图像", 120, 182, 850)
    add_small_label(root, ids, "示例一：左右结构", 150, 265, 220)
    add_image(root, ids, component_dir / "狗_a.png", 150, 315, 160, 180, "部件 A：犭")
    add_arrow_text(root, ids, "+", 325, 360)
    add_image(root, ids, component_dir / "狗_b.png", 415, 315, 160, 180, "部件 B：句")
    add_arrow_text(root, ids, "→", 600, 360)
    add_small_label(root, ids, "结构标签：left_right", 710, 377, 250, fill=BLUE, color=WHITE)
    add_image(root, ids, image_dir / "狗_glyph.png", 1010, 300, 190, 220, "输出：狗")

    add_small_label(root, ids, "示例二：全包围结构", 150, 585, 250)
    add_image(root, ids, component_dir / "困_a.png", 150, 635, 160, 180, "部件 A：囗")
    add_arrow_text(root, ids, "+", 325, 680)
    add_image(root, ids, component_dir / "困_b.png", 415, 635, 160, 180, "部件 B：木")
    add_arrow_text(root, ids, "→", 600, 680)
    add_small_label(root, ids, "结构标签：full_surround", 710, 697, 280, fill=BLUE, color=WHITE)
    add_image(root, ids, image_dir / "困_glyph.png", 1030, 620, 190, 220, "输出：困")

    add_card(
        root,
        ids,
        1290,
        260,
        500,
        640,
        "研究目标",
        "1. 生成结果在结构上正确、在视觉上可辨识。\n\n2. 在随机切分之外，还要在未见整字组合上保持一定泛化能力。\n\n3. 条件接口同时兼容文本部件输入和图像部件输入，为后续 OOV 扩展验证留下空间。\n\n4. 任务不是规则拼接，也不是开放域文生图，而是一个边界清楚的结构化条件生成问题。",
        fill=CARD_ALT,
    )


def build_overall_plan(root: ET.Element, ids: IdGen) -> None:
    section_header(root, ids, "05 总体研究方案")
    title_with_badge(root, ids, "先把任务做小、做稳，再推进到主模型", 120, 180, 840)
    steps = [
        ("数据构建", "字体渲染 + 拆字表\n生成目标字、部件图、结构标签和样式信息"),
        ("基线模型", "先训练条件 U-Net\n验证任务是否能稳定跑通"),
        ("表征压缩", "训练 VAE\n把字形映射到潜空间"),
        ("主模型", "在潜空间中做条件扩散\n比较结构一致性与生成质量"),
        ("评测与扩展", "整字留出评测\n加少量 OOV/古壮字扩展验证"),
    ]
    x = 120
    widths = [300, 300, 280, 300, 420]
    for idx, (title, body) in enumerate(steps):
        width = widths[idx]
        add_cell(root, ids.next(), "", rect_style(CARD, LINE, rounded=True), x, 350, width, 360)
        add_cell(root, ids.next(), f"{idx + 1:02d}", text_style(size=42, color=BLUE, bold=True, align="center", valign="middle"), x + 20, 385, 72, 56)
        add_cell(root, ids.next(), title, text_style(size=24, bold=True), x + 100, 390, width - 130, 40)
        add_cell(root, ids.next(), body, text_style(size=18), x + 26, 470, width - 52, 170)
        if idx < len(steps) - 1:
            add_arrow_text(root, ids, "→", x + width + 10, 470)
        x += width + 55
    add_card(
        root,
        ids,
        200,
        780,
        1520,
        140,
        "推进原则",
        "主任务以现代汉字为核心，OOV 或古壮字只作为方法外延验证。研究路线优先保证问题定义清楚、数据协议稳定和基线可复现。",
        fill=CARD_ALT,
    )


def build_method_design(root: ET.Element, ids: IdGen) -> None:
    section_header(root, ids, "06 核心方法设计")
    title_with_badge(root, ids, "统一条件接口 + 显式结构约束", 120, 180, 760)
    add_card(root, ids, 120, 290, 560, 560, "统一条件接口", "文本部件输入适合现代汉字的数据构造与词表管理。\n\n图像部件输入更适合未登录字符、古壮字和异体字场景。\n\n结构标签与样式信息与两类输入共同进入同一条件对象，避免形成彼此割裂的两套方法。")
    add_card(root, ids, 720, 290, 520, 260, "条件 U-Net 基线", "从部件图像、部件 ID、结构标签和样式信息中提取条件，尽早建立一个稳定、可分析的生成基线。")
    add_card(root, ids, 720, 590, 520, 260, "VAE + 潜空间扩散", "先用 VAE 压缩字形，再在潜空间中做条件去噪生成，降低计算压力，同时比较主模型与基线在结构一致性上的差异。")
    add_card(root, ids, 1280, 290, 520, 560, "显式结构约束", "布局热图不只是可视化，而是作为空间先验进入模型。\n\n结构监督和后续结构探针共同服务于一个目标：避免模型只记住整字外观，而没有真正学会部件之间该怎样组合。", fill=CARD_ALT)


def build_key_questions(root: ET.Element, ids: IdGen) -> None:
    section_header(root, ids, "07 拟解决的关键问题")
    title_with_badge(root, ids, "真正困难的地方，不是“能不能出图”，而是“是否学会怎样组合”", 120, 180, 1140)
    rows = [
        ("多模态条件如何统一", "文本部件与图像部件在接口层和编码层同时保持兼容，避免训练过程出现条件割裂。"),
        ("结构错误如何显式控制", "通过布局热图、结构监督和辅助探针约束部件比例、位置关系和包围结构，降低“看起来像字但结构不对”的风险。"),
        ("组合泛化如何被证明", "把整字留出切分放到关键位置，结合结构正确性分析和典型案例，区分真正泛化与训练样本记忆。"),
    ]
    y = 305
    for problem, response in rows:
        add_cell(root, ids.next(), "", rect_style(CARD, LINE, rounded=True), 120, y, 520, 190)
        add_cell(root, ids.next(), problem, text_style(size=24, color=BLUE, bold=True), 150, y + 28, 460, 40)
        add_cell(root, ids.next(), "", rect_style(GOLD), 690, y + 70, 120, 10)
        add_cell(root, ids.next(), "", rect_style(CARD_ALT, LINE, rounded=True), 850, y, 930, 190)
        add_cell(root, ids.next(), response, text_style(size=20, color=INK), 885, y + 38, 860, 120)
        y += 225


def build_innovation_feasibility(root: ET.Element, ids: IdGen) -> None:
    section_header(root, ids, "08 创新点与可行性分析")
    title_with_badge(root, ids, "创新要讲清楚，可行性也要讲扎实", 120, 180, 780)
    add_card(root, ids, 120, 300, 760, 620, "创新点", "1. 把文本部件输入和图像部件输入统一到同一条件接口中。\n\n2. 将结构标签、布局热图和结构监督显式引入生成过程，而不是完全依赖模型隐式学习。\n\n3. 将整字留出切分放到更关键的位置，用组合泛化而不是单纯随机切分来检验方法是否成立。")
    add_card(root, ids, 940, 300, 840, 620, "可行性基础", "数据来源明确：公开字体与拆字表可以支持自动构造首版样本。\n\n研究路线可控：先做条件 U-Net 基线，再推进到 VAE 与潜空间扩散。\n\n工程基础已具备：仓库中已有 build_dataset、train_baseline、train_diffusion、run_demo 等脚本，说明这项工作不是从零起步。\n\n风险应对清楚：先低分辨率跑通链路，再逐步提升标准分辨率，并把 OOV 验证控制在小样本轻量适配范围内。", fill=CARD_ALT)


def build_schedule(root: ET.Element, ids: IdGen) -> None:
    section_header(root, ids, "09 研究计划与预期成果")
    title_with_badge(root, ids, "按阶段推进，保证每一步都能形成可复核结果", 120, 180, 860)
    add_cell(root, ids.next(), "", rect_style(CARD, LINE, rounded=True), 120, 300, 1110, 620)
    phases = [
        ("2026.04-2026.06", "文献梳理与任务收口", "明确部件集合、结构标签、数据协议和实验划分方式。"),
        ("2026.07-2026.09", "基线模型训练", "先在较低分辨率下跑通训练、保存、加载和推理全链路。"),
        ("2026.10-2026.12", "主模型与核心对比", "完成 VAE 与潜空间扩散训练，开展整字留出条件下的核心实验。"),
        ("2027.01-2027.02", "扩展验证与演示整合", "选取少量 OOV 样本进行图像条件适配与演示整合。"),
        ("2027.03-2027.05", "论文写作与答辩准备", "完成图表整理、结果复核、论文写作和答辩材料收束。"),
    ]
    y = 350
    for idx, (period, stage, desc) in enumerate(phases):
        add_cell(root, ids.next(), "", rect_style(BLUE if idx == 0 else CARD_ALT, BLUE if idx == 0 else LINE, rounded=True), 165, y + 10, 240, 66)
        add_cell(root, ids.next(), period, text_style(size=18, color=WHITE if idx == 0 else BLUE, bold=True, align="center", valign="middle"), 165, y + 10, 240, 66)
        add_cell(root, ids.next(), stage, text_style(size=22, bold=True), 450, y, 360, 38)
        add_cell(root, ids.next(), desc, text_style(size=18), 450, y + 40, 690, 62)
        if idx < len(phases) - 1:
            add_cell(root, ids.next(), "", rect_style(GOLD), 278, y + 90, 14, 34)
        y += 108
    add_card(
        root,
        ids,
        1290,
        300,
        500,
        620,
        "预期成果",
        "1. 形成覆盖数据准备、模型训练、评测和推理的研究原型。\n\n2. 完成条件 U-Net 基线与潜空间扩散主模型的系统对比。\n\n3. 建立以整字留出切分为核心的评测协议。\n\n4. 通过小规模 OOV 扩展验证统一条件接口的可迁移性。\n\n5. 完成硕士开题后的持续研究与论文写作材料。",
        fill=CARD_ALT,
    )


def build_closing(root: ET.Element, ids: IdGen) -> None:
    add_cell(root, ids.next(), "", rect_style(BG), 0, 0, W, H)
    add_cell(root, ids.next(), "", rect_style(BLUE), 0, 0, W, 120)
    add_cell(root, ids.next(), "", rect_style(GOLD), 0, 120, W, 6)
    add_cell(root, ids.next(), "感谢各位老师的聆听与指导", text_style(size=42, color=INK, bold=True, align="center", valign="middle"), 300, 330, 1320, 70)
    add_cell(root, ids.next(), "Q & A", text_style(size=62, color=BLUE, bold=True, align="center", valign="middle"), 560, 470, 800, 100)
    add_cell(root, ids.next(), "汇报人：黄子涵    指导教师：白凤波    2026 年 4 月", text_style(size=22, color=MUTED, align="center", valign="middle"), 360, 690, 1200, 44)
    add_cell(root, ids.next(), "请各位老师批评指正", text_style(size=24, color=INK, align="center", valign="middle"), 540, 770, 840, 40)


def build_diagram(slide_name: str, builder) -> ET.Element:
    diagram = ET.Element("diagram", {"id": slide_name, "name": slide_name})
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        {
            "dx": "0",
            "dy": "0",
            "grid": "1",
            "gridSize": "10",
            "guides": "1",
            "tooltips": "1",
            "connect": "1",
            "arrows": "1",
            "fold": "1",
            "page": "1",
            "pageScale": "1",
            "pageWidth": str(W),
            "pageHeight": str(H),
            "math": "0",
            "shadow": "0",
        },
    )
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})
    ids = IdGen()
    builder(root, ids)
    return diagram


def main() -> None:
    mxfile = ET.Element("mxfile", {"host": "app.diagrams.net", "modified": "2026-04-21T00:00:00.000Z", "agent": "Codex", "version": "24.7.17"})
    for slide in deck_slides():
        mxfile.append(build_diagram(slide["name"], slide["builder"]))
    tree = ET.ElementTree(mxfile)
    ET.indent(tree, space="  ")
    tree.write(OUT_PATH, encoding="utf-8", xml_declaration=True)
    print(f"Saved drawio source to {OUT_PATH}")


if __name__ == "__main__":
    main()
