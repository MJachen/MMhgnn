from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "server_results"
OUT_DIR = ROOT / "reports" / "experiment_report_2026-06-24"
FIG_DIR = OUT_DIR / "figures"
REPORT_PATH = OUT_DIR / "MMHGNN_阶段性实验报告_2026-06-24.docx"
FULL_CSV_PATH = OUT_DIR / "experiment_results_full.csv"

RUN_ORDER = [
    "brats2020_public_eval",
    "public_eval_current_calibrated",
    "public_eval_fixed_05",
    "public_eval_fixed_01",
    "public_calib_safe_seed42",
    "missing_curriculum_seed42",
    "no_t1ce_focus_seed42",
]

RUN_CODES = {
    "brats2020_public_eval": "B0",
    "public_eval_current_calibrated": "B0-R",
    "public_eval_fixed_05": "T0.5",
    "public_eval_fixed_01": "T0.1",
    "public_calib_safe_seed42": "CS",
    "missing_curriculum_seed42": "MC",
    "no_t1ce_focus_seed42": "NF",
}

RUN_NAMES = {
    "brats2020_public_eval": "初始公开集基线",
    "public_eval_current_calibrated": "基线校准复评",
    "public_eval_fixed_05": "固定阈值 0.5",
    "public_eval_fixed_01": "固定阈值 0.1",
    "public_calib_safe_seed42": "受约束安全校准",
    "missing_curriculum_seed42": "缺失模态课程训练",
    "no_t1ce_focus_seed42": "无 T1ce 定向微调",
}

COMBO_ORDER = [
    "t2",
    "t1ce",
    "t1",
    "flair",
    "t2_t1ce",
    "t2_t1",
    "t2_flair",
    "t1ce_t1",
    "t1ce_flair",
    "t1_flair",
    "t2_t1ce_t1",
    "t2_t1ce_flair",
    "t2_t1_flair",
    "t1ce_t1_flair",
    "t2_t1ce_t1_flair",
]

COMBO_LABELS = {
    "t2": "T2",
    "t1ce": "T1ce",
    "t1": "T1",
    "flair": "FLAIR",
    "t2_t1ce": "T2+T1ce",
    "t2_t1": "T2+T1",
    "t2_flair": "T2+FLAIR",
    "t1ce_t1": "T1ce+T1",
    "t1ce_flair": "T1ce+FLAIR",
    "t1_flair": "T1+FLAIR",
    "t2_t1ce_t1": "T2+T1ce+T1",
    "t2_t1ce_flair": "T2+T1ce+FLAIR",
    "t2_t1_flair": "T2+T1+FLAIR",
    "t1ce_t1_flair": "T1ce+T1+FLAIR",
    "t2_t1ce_t1_flair": "Full",
}

FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\msyh.ttf"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
]
FONT_PATH = next((path for path in FONT_CANDIDATES if path.exists()), None)

NAVY = "18324A"
BLUE = "2E74B5"
TEAL = "2D7F7A"
GOLD = "A87816"
RED = "A33A3A"
LIGHT_BLUE = "E8F0F7"
LIGHT_GRAY = "F2F4F7"
PALE_GOLD = "FFF4D6"
PALE_RED = "FCE8E6"
WHITE = "FFFFFF"
MUTED = "626B73"
GRID = "D7DDE3"


def pil_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend([Path(r"C:\Windows\Fonts\msyhbd.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf")])
    if FONT_PATH:
        candidates.append(FONT_PATH)
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def load_results() -> pd.DataFrame:
    frames = []
    for run in RUN_ORDER:
        path = RESULTS_ROOT / run / "evaluate_only" / "metrics.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing metrics file: {path}")
        frame = pd.read_csv(path)
        frame.insert(0, "run", run)
        frame.insert(1, "run_code", RUN_CODES[run])
        frame.insert(2, "run_name_cn", RUN_NAMES[run])
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["combo"] = pd.Categorical(data["combo"], COMBO_ORDER, ordered=True)
    data = data.sort_values(["run", "combo"]).reset_index(drop=True)
    return data


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_candidate_predictions() -> tuple[int, int]:
    left_root = RESULTS_ROOT / "missing_curriculum_seed42" / "evaluate_only"
    right_root = RESULTS_ROOT / "no_t1ce_focus_seed42" / "evaluate_only"
    equal = 0
    total = 0
    for combo in COMBO_ORDER:
        left = left_root / combo / "predictions.csv"
        right = right_root / combo / "predictions.csv"
        if left.exists() and right.exists():
            total += 1
            equal += int(sha256(left) == sha256(right))
    return equal, total


def compute_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    core_no_t1ce = {"t2", "flair", "t2_flair"}
    for run in RUN_ORDER:
        frame = data[data["run"] == run].copy()
        core = frame[frame["combo"].astype(str).isin(core_no_t1ce)]
        full = frame[frame["combo"].astype(str) == "t2_t1ce_t1_flair"].iloc[0]
        rows.append(
            {
                "run": run,
                "code": RUN_CODES[run],
                "name": RUN_NAMES[run],
                "mean_bac": frame["bal_acc"].mean(),
                "mean_auc": frame["auc"].mean(),
                "core_no_t1ce_bac": core["bal_acc"].mean(),
                "core_no_t1ce_auc": core["auc"].mean(),
                "full_bac": full["bal_acc"],
                "full_auc": full["auc"],
            }
        )
    return pd.DataFrame(rows)


def metric(data: pd.DataFrame, run: str, combo: str, key: str) -> float:
    row = data[(data["run"] == run) & (data["combo"].astype(str) == combo)]
    return float(row.iloc[0][key])


def hex_rgb(value: str) -> tuple[int, int, int]:
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def blend(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def save_logic_chain(path: Path) -> None:
    width, height = 1800, 520
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = pil_font(42, bold=True)
    box_title = pil_font(28, bold=True)
    box_body = pil_font(22)
    draw.text((70, 35), "实验逻辑链：从可复现评估到缺失模态鲁棒性验证", font=title_font, fill=hex_rgb(NAVY))

    boxes = [
        ("1 评估基础", "公开 BraTS 配置\n固定划分与同步流程", LIGHT_BLUE, BLUE),
        ("2 行为基线", "15 种模态组合\n定位 T1ce 主导", "EAF5F2", TEAL),
        ("3 校准诊断", "分组阈值 vs 固定阈值\n区分排序与决策问题", PALE_GOLD, GOLD),
        ("4 鲁棒性训练", "课程缺失与无 T1ce 微调\n候选结果已产出", "F7ECEC", RED),
        ("5 协议审计", "发现预测重复\n修复选择与哈希校验", LIGHT_GRAY, NAVY),
    ]
    left, top, box_w, box_h, gap = 70, 150, 295, 250, 46
    for index, (heading, body, fill, accent) in enumerate(boxes):
        x0 = left + index * (box_w + gap)
        x1 = x0 + box_w
        draw.rounded_rectangle((x0, top, x1, top + box_h), radius=18, fill=hex_rgb(fill), outline=hex_rgb(accent), width=4)
        draw.rectangle((x0, top, x1, top + 12), fill=hex_rgb(accent))
        draw.text((x0 + 22, top + 35), heading, font=box_title, fill=hex_rgb(NAVY))
        y = top + 105
        for line in body.split("\n"):
            draw.text((x0 + 22, y), line, font=box_body, fill=(45, 53, 60))
            y += 46
        if index < len(boxes) - 1:
            ax0, ay = x1 + 8, top + box_h // 2
            ax1 = x1 + gap - 8
            draw.line((ax0, ay, ax1, ay), fill=hex_rgb(MUTED), width=5)
            draw.polygon([(ax1, ay), (ax1 - 17, ay - 12), (ax1 - 17, ay + 12)], fill=hex_rgb(MUTED))
    image.save(path, quality=95)


def save_heatmap(data: pd.DataFrame, path: Path) -> None:
    runs = RUN_ORDER
    width, height = 2400, 1250
    margin_left, margin_top, margin_right, margin_bottom = 360, 190, 70, 330
    grid_w = width - margin_left - margin_right
    grid_h = height - margin_top - margin_bottom
    cell_w = grid_w / len(COMBO_ORDER)
    cell_h = grid_h / len(runs)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((65, 40), "15 种模态可用性下的平衡准确率（BAC）", font=pil_font(44, True), fill=hex_rgb(NAVY))
    draw.text((65, 100), "颜色越深表示 BAC 越高；MC 与 NF 数值相同，不能视作独立改进证据。", font=pil_font(24), fill=hex_rgb(MUTED))

    low, mid, high = hex_rgb("F6E7E7"), hex_rgb("FFF4D6"), hex_rgb("D8EEE8")
    matrix = data.pivot(index="run", columns="combo", values="bal_acc")
    for row_index, run in enumerate(runs):
        y0 = margin_top + row_index * cell_h
        draw.text((65, y0 + cell_h * 0.27), f"{RUN_CODES[run]}  {RUN_NAMES[run]}", font=pil_font(22, row_index in (0, 5, 6)), fill=hex_rgb(NAVY))
        for col_index, combo in enumerate(COMBO_ORDER):
            value = float(matrix.loc[run, combo])
            if value <= 0.65:
                color = blend(low, mid, (value - 0.4) / 0.25)
            else:
                color = blend(mid, high, (value - 0.65) / 0.2)
            x0 = margin_left + col_index * cell_w
            draw.rectangle((x0, y0, x0 + cell_w, y0 + cell_h), fill=color, outline=hex_rgb(WHITE), width=2)
            label = f"{value:.3f}"
            bbox = draw.textbbox((0, 0), label, font=pil_font(18, value >= 0.78))
            draw.text((x0 + (cell_w - (bbox[2] - bbox[0])) / 2, y0 + (cell_h - (bbox[3] - bbox[1])) / 2 - 3), label, font=pil_font(18, value >= 0.78), fill=(35, 40, 45))

    for col_index, combo in enumerate(COMBO_ORDER):
        x = margin_left + col_index * cell_w + cell_w * 0.5
        label = COMBO_LABELS[combo]
        rotated = Image.new("RGBA", (220, 55), (255, 255, 255, 0))
        rd = ImageDraw.Draw(rotated)
        rd.text((0, 0), label, font=pil_font(19), fill=hex_rgb(NAVY))
        rotated = rotated.rotate(45, expand=True, resample=Image.Resampling.BICUBIC)
        image.paste(rotated, (int(x - 12), int(margin_top + grid_h + 15)), rotated)
    image.save(path, quality=95)


def save_key_comparison(data: pd.DataFrame, path: Path) -> None:
    combos = ["t2_t1ce_t1_flair", "t1ce_flair", "t1ce", "t2_flair", "flair", "t2"]
    labels = [COMBO_LABELS[item] for item in combos]
    baseline = [metric(data, "brats2020_public_eval", combo, "bal_acc") for combo in combos]
    candidate = [metric(data, "missing_curriculum_seed42", combo, "bal_acc") for combo in combos]
    width, height = 1700, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((65, 35), "关键模态组合：基线与课程训练候选结果", font=pil_font(42, True), fill=hex_rgb(NAVY))
    draw.text((65, 90), "候选模型改善 Full/T1ce+FLAIR 的阈值后 BAC，但无 T1ce 核心组合未改善。", font=pil_font(24), fill=hex_rgb(MUTED))
    chart_left, chart_top, chart_right, chart_bottom = 125, 180, width - 80, height - 130
    draw.line((chart_left, chart_bottom, chart_right, chart_bottom), fill=(80, 85, 90), width=3)
    draw.line((chart_left, chart_top, chart_left, chart_bottom), fill=(80, 85, 90), width=3)
    for tick in range(0, 10):
        value = tick / 10
        y = chart_bottom - value * (chart_bottom - chart_top)
        draw.line((chart_left, y, chart_right, y), fill=hex_rgb("E6EAEE"), width=2)
        draw.text((52, y - 13), f"{value:.1f}", font=pil_font(18), fill=hex_rgb(MUTED))
    group_w = (chart_right - chart_left) / len(combos)
    bar_w = group_w * 0.27
    for i, label in enumerate(labels):
        center = chart_left + group_w * (i + 0.5)
        for j, (value, color) in enumerate([(baseline[i], BLUE), (candidate[i], GOLD)]):
            x0 = center + (j - 1) * bar_w + 7
            x1 = x0 + bar_w - 12
            y0 = chart_bottom - value * (chart_bottom - chart_top)
            draw.rectangle((x0, y0, x1, chart_bottom), fill=hex_rgb(color))
            draw.text((x0 - 2, y0 - 31), f"{value:.3f}", font=pil_font(18, True), fill=hex_rgb(color))
        bbox = draw.textbbox((0, 0), label, font=pil_font(20))
        draw.text((center - (bbox[2] - bbox[0]) / 2, chart_bottom + 22), label, font=pil_font(20), fill=hex_rgb(NAVY))
    draw.rectangle((width - 475, 44, width - 445, 74), fill=hex_rgb(BLUE))
    draw.text((width - 430, 43), "B0 基线", font=pil_font(21), fill=hex_rgb(NAVY))
    draw.rectangle((width - 275, 44, width - 245, 74), fill=hex_rgb(GOLD))
    draw.text((width - 230, 43), "MC 候选", font=pil_font(21), fill=hex_rgb(NAVY))
    image.save(path, quality=95)


def threshold_group(combo: str) -> str:
    parts = set(combo.split("_"))
    if "t1ce" in parts:
        return "含 T1ce"
    if "t1" in parts:
        return "无 T1ce/含 T1"
    return "无 T1ce/无 T1"


def save_calibration_comparison(data: pd.DataFrame, path: Path) -> None:
    runs = ["public_eval_current_calibrated", "public_eval_fixed_05", "public_eval_fixed_01", "public_calib_safe_seed42"]
    groups = ["含 T1ce", "无 T1ce/含 T1", "无 T1ce/无 T1"]
    colors = [BLUE, GOLD, TEAL, RED]
    values: dict[str, list[float]] = {}
    for run in runs:
        frame = data[data["run"] == run].copy()
        frame["group"] = frame["combo"].astype(str).map(threshold_group)
        means = frame.groupby("group", observed=True)["bal_acc"].mean()
        values[run] = [float(means[group]) for group in groups]

    width, height = 1700, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((65, 35), "校准策略对不同模态组的影响", font=pil_font(42, True), fill=hex_rgb(NAVY))
    draw.text((65, 90), "固定阈值不改变 AUC，只改变决策点；分组校准对无 T1ce 组合更关键。", font=pil_font(24), fill=hex_rgb(MUTED))
    left, top, right, bottom = 125, 195, width - 80, height - 130
    draw.line((left, bottom, right, bottom), fill=(80, 85, 90), width=3)
    draw.line((left, top, left, bottom), fill=(80, 85, 90), width=3)
    for tick in range(0, 10):
        value = tick / 10
        y = bottom - value * (bottom - top)
        draw.line((left, y, right, y), fill=hex_rgb("E6EAEE"), width=2)
        draw.text((52, y - 13), f"{value:.1f}", font=pil_font(18), fill=hex_rgb(MUTED))
    group_w = (right - left) / len(groups)
    bar_w = group_w * 0.16
    for group_index, group in enumerate(groups):
        center = left + group_w * (group_index + 0.5)
        for run_index, run in enumerate(runs):
            value = values[run][group_index]
            x0 = center + (run_index - 2) * bar_w + 8
            x1 = x0 + bar_w - 10
            y0 = bottom - value * (bottom - top)
            draw.rectangle((x0, y0, x1, bottom), fill=hex_rgb(colors[run_index]))
            draw.text((x0 - 1, y0 - 28), f"{value:.3f}", font=pil_font(17), fill=hex_rgb(colors[run_index]))
        bbox = draw.textbbox((0, 0), group, font=pil_font(21))
        draw.text((center - (bbox[2] - bbox[0]) / 2, bottom + 24), group, font=pil_font(21), fill=hex_rgb(NAVY))
    legend_x = 930
    for i, run in enumerate(runs):
        x = legend_x + (i % 2) * 330
        y = 43 + (i // 2) * 45
        draw.rectangle((x, y, x + 26, y + 26), fill=hex_rgb(colors[i]))
        draw.text((x + 38, y - 2), f"{RUN_CODES[run]} {RUN_NAMES[run]}", font=pil_font(19), fill=hex_rgb(NAVY))
    image.save(path, quality=95)


def save_mechanism_analysis(data: pd.DataFrame, path: Path) -> None:
    full_row = data[(data["run"] == "brats2020_public_eval") & (data["combo"].astype(str) == "t2_t1ce_t1_flair")].iloc[0]
    gates = [float(full_row[f"avg_gate_{mod}"]) for mod in ["t1", "t1ce", "t2", "flair"]]
    edge_values = []
    for combo in ["t2_t1ce_t1_flair", "t2_flair"]:
        path_csv = RESULTS_ROOT / "brats2020_public_eval" / "evaluate_only" / combo / "edge_type_drop_metrics.csv"
        edge = pd.read_csv(path_csv)
        full = float(edge[edge["setting"] == "full"].iloc[0]["bal_acc"])
        dropped = float(edge[edge["setting"] == "drop_anatomy"].iloc[0]["bal_acc"])
        edge_values.append((COMBO_LABELS[combo], full, dropped))

    width, height = 1700, 850
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((65, 35), "模型机制证据：模态门控与解剖边", font=pil_font(42, True), fill=hex_rgb(NAVY))
    draw.text((65, 90), "当前基线主要依赖 T1ce；当 T1ce 缺失时，解剖边对 BAC 的贡献更明显。", font=pil_font(24), fill=hex_rgb(MUTED))

    draw.text((80, 170), "Full 组合的平均模态门控", font=pil_font(28, True), fill=hex_rgb(NAVY))
    x0, y0, total_w, bar_h = 85, 245, 690, 90
    gate_colors = ["9BB7D4", BLUE, "7FB9B5", GOLD]
    cursor = x0
    for mod, value, color in zip(["T1", "T1ce", "T2", "FLAIR"], gates, gate_colors):
        segment = total_w * value
        draw.rectangle((cursor, y0, cursor + segment, y0 + bar_h), fill=hex_rgb(color))
        cursor += segment
    legend_y = 380
    for i, (mod, value, color) in enumerate(zip(["T1", "T1ce", "T2", "FLAIR"], gates, gate_colors)):
        y = legend_y + i * 64
        draw.rectangle((90, y, 125, y + 35), fill=hex_rgb(color))
        draw.text((145, y - 2), f"{mod}: {value * 100:.1f}%", font=pil_font(24, i == 1), fill=hex_rgb(NAVY))

    draw.text((930, 170), "移除解剖边后的 BAC 变化", font=pil_font(28, True), fill=hex_rgb(NAVY))
    left, top, right, bottom = 970, 255, 1620, 680
    draw.line((left, bottom, right, bottom), fill=(80, 85, 90), width=3)
    draw.line((left, top, left, bottom), fill=(80, 85, 90), width=3)
    group_w = (right - left) / 2
    bar_w = 90
    for i, (label, full, dropped) in enumerate(edge_values):
        center = left + group_w * (i + 0.5)
        for j, (value, color) in enumerate([(full, TEAL), (dropped, RED)]):
            x = center + (j - 1) * bar_w + 18
            y = bottom - value * (bottom - top)
            draw.rectangle((x, y, x + bar_w - 24, bottom), fill=hex_rgb(color))
            draw.text((x - 3, y - 31), f"{value:.3f}", font=pil_font(20, True), fill=hex_rgb(color))
        bbox = draw.textbbox((0, 0), label, font=pil_font(22))
        draw.text((center - (bbox[2] - bbox[0]) / 2, bottom + 23), label, font=pil_font(22), fill=hex_rgb(NAVY))
    draw.rectangle((1060, 735, 1090, 765), fill=hex_rgb(TEAL))
    draw.text((1105, 733), "完整图", font=pil_font(20), fill=hex_rgb(NAVY))
    draw.rectangle((1280, 735, 1310, 765), fill=hex_rgb(RED))
    draw.text((1325, 733), "去除解剖边", font=pil_font(20), fill=hex_rgb(NAVY))
    image.save(path, quality=95)


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin_name, margin_value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        tag = "w:" + margin_name
        node = tc_mar.find(qn(tag))
        if node is None:
            node = OxmlElement(tag)
            tc_mar.append(node)
        node.set(qn("w:w"), str(margin_value))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_table_geometry(table, widths_dxa: list[int], indent_dxa: int = 120) -> None:
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(sum(widths_dxa)))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), str(indent_dxa))
    tbl_ind.set(qn("w:type"), "dxa")
    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths_dxa:
        grid_col = OxmlElement("w:gridCol")
        grid_col.set(qn("w:w"), str(width))
        grid.append(grid_col)
    for row in table.rows:
        for index, cell in enumerate(row.cells):
            width = widths_dxa[index]
            cell.width = Inches(width / 1440)
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(width))
            tc_w.set(qn("w:type"), "dxa")
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_run_font(run, size: float | None = None, bold: bool | None = None, color: str | None = None, italic: bool | None = None) -> None:
    run.font.name = "Microsoft YaHei"
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), "Microsoft YaHei")
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), "Microsoft YaHei")
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def style_paragraph(paragraph, size: float = 10.5, color: str = "222222", bold: bool = False, align=None) -> None:
    if align is not None:
        paragraph.alignment = align
    for run in paragraph.runs:
        set_run_font(run, size=size, bold=bold, color=color)


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run()
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = " PAGE "
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char1)
    run._r.append(instr_text)
    run._r.append(fld_char2)
    set_run_font(run, 8.5, color=MUTED)


def configure_styles(doc: Document) -> None:
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Microsoft YaHei"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Microsoft YaHei")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Microsoft YaHei")
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15
    for name, size, color, before, after in [
        ("Heading 1", 16, BLUE, 16, 8),
        ("Heading 2", 13, BLUE, 12, 6),
        ("Heading 3", 11.5, NAVY, 8, 4),
    ]:
        style = styles[name]
        style.font.name = "Microsoft YaHei"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Microsoft YaHei")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Microsoft YaHei")
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True


def configure_section(section, landscape: bool = False) -> None:
    if landscape:
        section.orientation = WD_ORIENT.LANDSCAPE
        section.page_width = Inches(11)
        section.page_height = Inches(8.5)
        section.left_margin = Inches(0.62)
        section.right_margin = Inches(0.62)
        section.top_margin = Inches(0.62)
        section.bottom_margin = Inches(0.62)
    else:
        section.orientation = WD_ORIENT.PORTRAIT
        section.page_width = Inches(8.5)
        section.page_height = Inches(11)
        section.left_margin = Inches(0.82)
        section.right_margin = Inches(0.82)
        section.top_margin = Inches(0.72)
        section.bottom_margin = Inches(0.72)
    section.header_distance = Inches(0.32)
    section.footer_distance = Inches(0.32)
    header = section.header
    hp = header.paragraphs[0]
    hp.text = "MMHGNN 阶段性实验报告  |  公开 BraTS 缺失模态鲁棒性"
    style_paragraph(hp, size=8.5, color=MUTED)
    footer = section.footer
    fp = footer.paragraphs[0]
    prefix = fp.add_run("内部汇报  ·  2026-06-24  ·  ")
    set_run_font(prefix, 8.5, color=MUTED)
    add_page_number(fp)


def add_title_block(doc: Document) -> None:
    kicker = doc.add_paragraph()
    kicker.paragraph_format.space_after = Pt(5)
    run = kicker.add_run("阶段性研究汇报")
    set_run_font(run, 10, True, GOLD)
    title = doc.add_paragraph()
    title.paragraph_format.space_after = Pt(5)
    run = title.add_run("MMHGNN 阶段性实验报告")
    set_run_font(run, 24, True, NAVY)
    subtitle = doc.add_paragraph()
    subtitle.paragraph_format.space_after = Pt(12)
    run = subtitle.add_run("公开 BraTS 多模态缺失鲁棒性、阈值校准与协议审计")
    set_run_font(run, 13, False, MUTED)
    meta = doc.add_table(rows=3, cols=2)
    meta_data = [
        ("汇报日期", "2026 年 6 月 24 日"),
        ("实验对象", "公开 BraTS，固定划分 train/val/test = 292/36/37"),
        ("核心口径", "BAC、AUC、SEN、SPE；测试集 29 阳性 / 8 阴性"),
    ]
    for row, values in zip(meta.rows, meta_data):
        for index, value in enumerate(values):
            row.cells[index].text = value
            set_cell_shading(row.cells[index], LIGHT_GRAY if index == 0 else WHITE)
            for p in row.cells[index].paragraphs:
                style_paragraph(p, size=9.5, color=NAVY if index == 0 else "222222", bold=index == 0)
    set_table_geometry(meta, [1600, 7760])


def add_callout(doc: Document, title: str, text: str, fill: str, accent: str) -> None:
    table = doc.add_table(rows=1, cols=1)
    cell = table.cell(0, 0)
    set_cell_shading(cell, fill)
    paragraph = cell.paragraphs[0]
    title_run = paragraph.add_run(title + "  ")
    set_run_font(title_run, 10.5, True, accent)
    body_run = paragraph.add_run(text)
    set_run_font(body_run, 10.5, False, "222222")
    set_table_geometry(table, [9360])
    after = doc.add_paragraph()
    after.paragraph_format.space_after = Pt(0)


def add_figure(doc: Document, image_path: Path, caption: str, width: float = 6.45) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    paragraph.add_run().add_picture(str(image_path), width=Inches(width))
    caption_p = doc.add_paragraph()
    caption_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_p.paragraph_format.space_after = Pt(8)
    run = caption_p.add_run(caption)
    set_run_font(run, 8.8, False, MUTED)


def fill_table(table, headers: list[str], rows: list[list[str]], widths: list[int], font_size: float = 8.5, header_fill: str = LIGHT_GRAY, first_col_left: bool = True) -> None:
    for index, header in enumerate(headers):
        cell = table.rows[0].cells[index]
        cell.text = header
        set_cell_shading(cell, header_fill)
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            style_paragraph(p, size=font_size, color=NAVY, bold=True)
    set_repeat_table_header(table.rows[0])
    for row_data in rows:
        cells = table.add_row().cells
        for index, value in enumerate(row_data):
            cells[index].text = str(value)
            if len(table.rows) % 2 == 0:
                set_cell_shading(cells[index], "FAFBFC")
            for p in cells[index].paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if index == 0 and first_col_left else WD_ALIGN_PARAGRAPH.CENTER
                style_paragraph(p, size=font_size, color="222222", bold=False)
    set_table_geometry(table, widths)


def add_summary_table(doc: Document, summary: pd.DataFrame) -> None:
    headers = ["代码", "实验", "平均 BAC", "平均 AUC", "无 T1ce 核心 BAC", "Full BAC", "Full AUC", "判定"]
    verdicts = {
        "B0": "可靠基线",
        "B0-R": "复评一致",
        "T0.5": "阈值消融",
        "T0.1": "阈值消融",
        "CS": "校准候选",
        "MC": "待协议重跑",
        "NF": "结果重复，待重跑",
    }
    rows = []
    for item in summary.itertuples(index=False):
        rows.append(
            [
                item.code,
                item.name,
                f"{item.mean_bac:.3f}",
                f"{item.mean_auc:.3f}",
                f"{item.core_no_t1ce_bac:.3f}",
                f"{item.full_bac:.3f}",
                f"{item.full_auc:.3f}",
                verdicts[item.code],
            ]
        )
    table = doc.add_table(rows=1, cols=len(headers))
    fill_table(table, headers, rows, [630, 1850, 1000, 1000, 1500, 930, 930, 1520], font_size=7.8)


def add_run_definition_table(doc: Document) -> None:
    rows = [
        ["B0", "同一基线模型 + checkpoint 分组阈值", "建立 15 种模态组合的行为基线"],
        ["B0-R", "同一 checkpoint 的复评", "验证评估流程与结果复现性"],
        ["T0.5 / T0.1", "同一模型，忽略 checkpoint 校准并固定阈值", "隔离阈值对 BAC/SEN/SPE 的影响；AUC 应保持不变"],
        ["CS", "阈值限制在 [0.05, 0.95]，并采用接近 0.5 的平局规则", "抑制极端阈值，改善小验证组的决策稳定性"],
        ["MC", "缺失模态课程训练", "提升对随机缺失模式的整体鲁棒性"],
        ["NF", "在 MC 后定向采样 T2、FLAIR、T2+FLAIR", "重点提升无 T1ce 条件，同时保护 Full 性能"],
    ]
    table = doc.add_table(rows=1, cols=3)
    fill_table(table, ["实验", "与前一实验的关系", "要回答的问题"], rows, [1050, 4060, 4250], font_size=8.4)


def add_full_results_table(doc: Document, data: pd.DataFrame) -> None:
    headers = ["Run", "组合", "Thr", "ACC", "AUC", "BAC", "SEN", "SPE", "TN", "FP", "FN", "TP"]
    rows = []
    for run in RUN_ORDER:
        frame = data[data["run"] == run].copy()
        frame["combo"] = frame["combo"].astype(str)
        frame["order"] = frame["combo"].map({combo: i for i, combo in enumerate(COMBO_ORDER)})
        frame = frame.sort_values("order")
        for row in frame.itertuples(index=False):
            rows.append(
                [
                    RUN_CODES[run],
                    COMBO_LABELS[str(row.combo)],
                    f"{float(row.applied_threshold):.3f}",
                    f"{float(row.acc):.3f}",
                    f"{float(row.auc):.3f}",
                    f"{float(row.bal_acc):.3f}",
                    f"{float(row.sen):.3f}",
                    f"{float(row.spe):.3f}",
                    str(int(row.tn)),
                    str(int(row.fp)),
                    str(int(row.fn)),
                    str(int(row.tp)),
                ]
            )
    table = doc.add_table(rows=1, cols=len(headers))
    fill_table(table, headers, rows, [620, 1750, 700, 680, 680, 680, 680, 680, 520, 520, 520, 520], font_size=6.5)


def build_report(data: pd.DataFrame, summary: pd.DataFrame, equal_predictions: tuple[int, int]) -> None:
    doc = Document()
    configure_styles(doc)
    configure_section(doc.sections[0], landscape=False)
    add_title_block(doc)

    baseline_full_bac = metric(data, "brats2020_public_eval", "t2_t1ce_t1_flair", "bal_acc")
    baseline_full_auc = metric(data, "brats2020_public_eval", "t2_t1ce_t1_flair", "auc")
    candidate_full_bac = metric(data, "missing_curriculum_seed42", "t2_t1ce_t1_flair", "bal_acc")
    candidate_full_auc = metric(data, "missing_curriculum_seed42", "t2_t1ce_t1_flair", "auc")
    baseline_t2_flair = metric(data, "brats2020_public_eval", "t2_flair", "bal_acc")
    candidate_t2_flair = metric(data, "missing_curriculum_seed42", "t2_flair", "bal_acc")

    doc.add_heading("1. 汇报结论", level=1)
    add_callout(
        doc,
        "阶段判断",
        "评估与诊断链路已经打通，基线结论清楚；校准实验回答了阈值稳定性问题；缺失模态训练已产出候选结果，但由于 MC 与 NF 的预测文件逐项重复，且后续代码专门修复了训练协议，这两组结果必须重跑后才能作为方法改进证据。",
        LIGHT_BLUE,
        BLUE,
    )
    conclusions = [
        f"基线在 Full 与 T1ce+FLAIR 下均达到 BAC={baseline_full_bac:.3f}；Full AUC={baseline_full_auc:.3f}。四模态并未优于 T1ce+FLAIR，说明当前模型主要由 T1ce 驱动。",
        f"无 T1ce 时，基线最好的代表组合是 T2+FLAIR（BAC={baseline_t2_flair:.3f}）。门控与边消融显示：FLAIR 是主要补充信号，解剖边在缺少 T1ce 时更重要。",
        f"MC 候选的 Full BAC 从 {baseline_full_bac:.3f} 升至 {candidate_full_bac:.3f}，但 Full AUC 从 {baseline_full_auc:.3f} 降至 {candidate_full_auc:.3f}；T2+FLAIR BAC 反而降至 {candidate_t2_flair:.3f}。这不是一致的鲁棒性提升。",
        f"MC 与 NF 在 {equal_predictions[0]}/{equal_predictions[1]} 个可核对模态组合上的 predictions.csv 哈希完全一致，因此 NF 的定向微调效果未被现有结果证明。",
    ]
    for item in conclusions:
        p = doc.add_paragraph(style="List Number")
        p.add_run(item)
        style_paragraph(p, size=10.5)

    doc.add_heading("2. 实验之间的逻辑链", level=1)
    add_figure(doc, FIG_DIR / "experiment_logic.png", "图 1  实验演进不是并列堆叠，而是上一阶段发现问题、下一阶段针对性回答。")
    p = doc.add_paragraph()
    p.add_run("逻辑解释：").bold = True
    p.add_run("先保证数据、划分和远程结果可复现；再用 15 种模态组合建立模型行为基线；基线暴露出 T1ce 主导和极端阈值问题，因此开展固定阈值/安全校准；随后进入缺失模态课程训练与无 T1ce 定向微调；最后通过预测哈希和联合验证发现协议缺陷，并完成修复。")
    style_paragraph(p, size=10.5)

    doc.add_heading("3. 已完成任务与实验设计", level=1)
    add_run_definition_table(doc)
    p = doc.add_paragraph()
    p.add_run("工程侧已完成：").bold = True
    p.add_run("公开 BraTS YAML 配置、数据配置检查、固定划分、服务器评估打包与本地拉取、逐病例预测导出、阈值校准诊断，以及修复后的联合验证/模型哈希/预测哈希审计。当前主分支仍停留在 2d87862；协议修复位于实验分支提交 1471a6b，尚待合并和服务器重跑。")
    style_paragraph(p, size=10.5)

    doc.add_heading("4. 完整结果概览", level=1)
    add_summary_table(doc, summary)
    p = doc.add_paragraph()
    run = p.add_run("口径说明：")
    set_run_font(run, 8.8, True, MUTED)
    run = p.add_run("“无 T1ce 核心”取 T2、FLAIR、T2+FLAIR 三组均值，与修复后协议的联合验证口径一致。B0 与 B0-R 复评一致属于预期；MC 与 NF 完全一致不属于预期。")
    set_run_font(run, 8.8, False, MUTED)

    add_figure(doc, FIG_DIR / "balanced_accuracy_heatmap.png", "图 2  所有 7 个运行在 15 种模态组合下的 BAC。", width=6.55)

    doc.add_heading("5. 结果分析", level=1)
    doc.add_heading("5.1 基线揭示 T1ce 主导，而不是四模态协同增益", level=2)
    p = doc.add_paragraph()
    p.add_run(f"Full 与 T1ce+FLAIR 的 BAC 都是 {baseline_full_bac:.3f}，两者 AUC 也相同（{baseline_full_auc:.3f}）。T1ce 单模态的 BAC 为 {metric(data, 'brats2020_public_eval', 't1ce', 'bal_acc'):.3f}、AUC 为 {metric(data, 'brats2020_public_eval', 't1ce', 'auc'):.3f}。因此，当前模型的主要判别来源是 T1ce，FLAIR 提供补充，但 T1/T2 并未形成稳定的额外收益。")
    style_paragraph(p, size=10.5)
    add_figure(doc, FIG_DIR / "mechanism_analysis.png", "图 3  Full 组合门控与解剖边消融。门控比例是模型行为证据，不应解释为因果贡献率。")

    doc.add_heading("5.2 校准实验把“排序能力”和“分类决策”分开", level=2)
    p = doc.add_paragraph()
    p.add_run("B0-R、T0.5、T0.1 使用同一模型概率，15 个组合的 AUC 保持不变，而 BAC/SEN/SPE 随阈值变化。这说明当前无 T1ce 失败的一部分来自阈值不稳，而不是排序能力完全消失。CS 通过限制阈值范围和改变平局规则，改善了部分 FLAIR 相关组合，但 T2 单模态仍退化为全阴性，校准不能代替表征学习。")
    style_paragraph(p, size=10.5)
    add_figure(doc, FIG_DIR / "calibration_group_comparison.png", "图 4  三类模态组的平均 BAC 随校准策略变化。")

    doc.add_heading("5.3 课程训练候选结果未形成一致的鲁棒性提升", level=2)
    add_figure(doc, FIG_DIR / "key_combo_comparison.png", "图 5  B0 与 MC 候选的关键组合 BAC 对比。")
    p = doc.add_paragraph()
    p.add_run(f"MC 的 Full BAC 提升 {candidate_full_bac - baseline_full_bac:+.3f}，但 Full AUC 下降 {candidate_full_auc - baseline_full_auc:+.3f}；T2+FLAIR BAC 下降 {candidate_t2_flair - baseline_t2_flair:+.3f}。这更像阈值后混淆矩阵的局部改善，而不是稳定的排序能力与缺失模态鲁棒性共同提升。由于 MC/NF 预测完全相同，现阶段不能宣称定向微调有效。")
    style_paragraph(p, size=10.5)

    doc.add_heading("6. 当前风险与下一步", level=1)
    risks = [
        ("P0", "按 1471a6b 协议重新训练 MC/NF seed42，并运行 compare_protocol_runs.py；只有 checkpoint 权重变化、定向微调被选中、四个无 T1ce 预测哈希均改变时，才进入结果比较。"),
        ("P1", "至少补 3 个随机种子，报告均值±标准差。当前 test=37 且负类仅 8 例，单个样本即可让 SPE 变化 0.125，单次结果不够稳定。"),
        ("P1", "模型选择同时约束无 T1ce AUC、无 T1ce BAC 和 Full BAC，避免通过牺牲完整模态或只移动阈值获得表面提升。"),
        ("P2", "保留解剖边；优先测试原型边、mask-aware head 和定向采样强度，而不是先删除对无 T1ce 组合已有帮助的结构。"),
    ]
    table = doc.add_table(rows=1, cols=3)
    fill_table(table, ["优先级", "行动", "完成标准"], [[priority, action, "产出可审计结果包与比较表"] for priority, action in risks], [900, 6660, 1800], font_size=8.4)
    add_callout(
        doc,
        "导师汇报建议",
        "本阶段的主要成果不是已经证明无 T1ce 微调有效，而是完成了可复现评估、找到了 T1ce 主导与阈值敏感问题，并通过结果哈希发现了协议缺陷。下一阶段目标明确：在修复后的联合验证约束下，证明无 T1ce 性能能够提升且 Full 性能不明显退化。",
        PALE_GOLD,
        GOLD,
    )

    landscape = doc.add_section(WD_SECTION.NEW_PAGE)
    configure_section(landscape, landscape=True)
    doc.add_heading("附录 A：完整逐运行结果", level=1)
    p = doc.add_paragraph()
    p.add_run("每个运行包含 15 种非空模态组合。Thr 为应用阈值；BAC 为平衡准确率。完整原始汇总同时保存为 experiment_results_full.csv。")
    style_paragraph(p, size=9)
    add_full_results_table(doc, data)

    portrait = doc.add_section(WD_SECTION.NEW_PAGE)
    configure_section(portrait, landscape=False)
    doc.add_heading("附录 B：实验代码与结果有效性", level=1)
    rows = [
        ["2d87862", "当前 main / B0", "公开数据配置与基础评估", "可作为基线"],
        ["699e608", "CS / MC / NF 现有结果", "增加校准与缺失模态实验", "MC/NF 需重跑"],
        ["1471a6b", "修复后协议", "固定 split、联合验证、选择护栏、权重/预测哈希", "代码完成，结果待产出"],
    ]
    table = doc.add_table(rows=1, cols=4)
    fill_table(table, ["提交", "对应实验", "作用", "当前结论"], rows, [1350, 2200, 3890, 1920], font_size=8.8)
    p = doc.add_paragraph()
    p.add_run("结论边界：").bold = True
    p.add_run("本报告只基于已同步到 server_results 的结果。由于协议修复后的训练结果尚未生成，MC/NF 只能作为问题定位材料，不能作为最终性能提升证据。")
    style_paragraph(p, size=10.5)

    doc.save(REPORT_PATH)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load_results()
    data.to_csv(FULL_CSV_PATH, index=False, encoding="utf-8-sig")
    summary = compute_summary(data)
    summary.to_csv(OUT_DIR / "experiment_summary.csv", index=False, encoding="utf-8-sig")
    equal_predictions = compare_candidate_predictions()
    save_logic_chain(FIG_DIR / "experiment_logic.png")
    save_heatmap(data, FIG_DIR / "balanced_accuracy_heatmap.png")
    save_key_comparison(data, FIG_DIR / "key_combo_comparison.png")
    save_calibration_comparison(data, FIG_DIR / "calibration_group_comparison.png")
    save_mechanism_analysis(data, FIG_DIR / "mechanism_analysis.png")
    build_report(data, summary, equal_predictions)
    print(f"report={REPORT_PATH}")
    print(f"rows={len(data)}")
    print(f"candidate_prediction_hash_matches={equal_predictions[0]}/{equal_predictions[1]}")


if __name__ == "__main__":
    main()
