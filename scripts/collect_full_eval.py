"""Validate and tabulate the 13-scene baseline versus quality SOTA run."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean


ROOT = Path(__file__).resolve().parents[1]
SCENES = json.loads((ROOT / "baseline/scenes_full.json").read_text())
EXPERIMENTS = {
    "baseline": ROOT / "results/baseline_A_baseline_B",
    "sota": ROOT / "results/baseline_A_topology_opacity7000_B",
}


def load(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def time_fraction(ratio):
    if ratio <= 0.10:
        return 1.0
    if ratio >= 1.0:
        return 0.0
    return (1.0 - ratio) / 0.90


def read_result(root, scene):
    path = root / scene["group"] / scene["scene"]
    a = load(path / "work/module_a.json")
    r = load(path / "resplat/resplat.json")
    b = load(path / "module_b/module_b.json")
    metrics = load(path / "module_b/metrics.json")
    split = load(path / "work/split.json")
    # The three legacy baseline JSON files predate the explicit seed field.
    # Their unchanged trainer calls safe_state(), which fixes Python, NumPy,
    # and PyTorch to zero.  Reject any explicit nonzero seed.
    if b.get("seed", 0) != 0:
        raise ValueError(f"nonzero seed: {path}")
    return {
        "path": str(path), "group": scene["group"], "scene": scene["scene"],
        "psnr": metrics["mean"]["psnr"], "ssim": metrics["mean"]["ssim"],
        "lpips": metrics["mean"]["lpips"], "time_A": a["total_seconds"],
        "time_resplat": r["total_seconds"], "time_B": b["total_seconds"],
        "time_total": a["total_seconds"] + r["total_seconds"] + b["total_seconds"],
        "num_test": metrics["num_views"], "iterations": b["iterations"],
        "split": split,
    }


def avg(rows):
    keys = ("psnr", "ssim", "lpips", "time_A", "time_resplat", "time_B",
            "time_total", "budget_ratio", "time_points")
    return {key: fmean(row[key] for row in rows) for key in keys}


def delta(value, base, digits):
    return f"{value:.{digits}f} ({value - base:+.{digits}f})"


def raw_table(rows, baseline=None):
    if baseline is None:
        lines = [
            "| dataset | scene | PSNR | SSIM | LPIPS | A(s) | ReSplat(s) | B(s) | total(s) | 30k train budget(s) | B/budget | time points(/20) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| {row['group']} | {row['scene']} | {row['psnr']:.3f} | {row['ssim']:.4f} | "
                f"{row['lpips']:.4f} | {row['time_A']:.3f} | {row['time_resplat']:.2f} | "
                f"{row['time_B']:.2f} | {row['time_total']:.3f} | {row['budget']:.2f} | "
                f"{100 * row['budget_ratio']:.2f}% | {row['time_points']:.3f} |")
        return "\n".join(lines)

    lines = [
        "| dataset | scene | PSNR (Δ) | SSIM (Δ) | LPIPS (Δ) | A(s) (Δ) | ReSplat(s) (Δ) | B(s) (Δ) | total(s) (Δ) | 30k train budget(s) | B/budget (Δpp) | time points(/20) (Δ) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row, base in zip(rows, baseline):
        lines.append(
            f"| {row['group']} | {row['scene']} | {delta(row['psnr'], base['psnr'], 3)} | "
            f"{delta(row['ssim'], base['ssim'], 4)} | {delta(row['lpips'], base['lpips'], 4)} | "
            f"{delta(row['time_A'], base['time_A'], 3)} | "
            f"{delta(row['time_resplat'], base['time_resplat'], 2)} | "
            f"{delta(row['time_B'], base['time_B'], 2)} | "
            f"{delta(row['time_total'], base['time_total'], 3)} | {row['budget']:.2f} | "
            f"{100 * row['budget_ratio']:.2f}% ({100 * (row['budget_ratio'] - base['budget_ratio']):+.2f}) | "
            f"{delta(row['time_points'], base['time_points'], 3)} |")
    return "\n".join(lines)


def aggregate_table(baseline, sota):
    by_group = defaultdict(lambda: ([], []))
    for base, candidate in zip(baseline, sota):
        by_group[base["group"]][0].append(base)
        by_group[base["group"]][1].append(candidate)
    entries = [(group, avg(pair[0]), avg(pair[1])) for group, pair in by_group.items()]
    # Scene-macro and dataset-macro are both reported because DL3DV has more scenes.
    entries.append(("all scenes (13-scene macro)", avg(baseline), avg(sota)))
    base_dataset = {key: fmean(base[key] for _, base, _ in entries[:3]) for key in avg(baseline)}
    sota_dataset = {key: fmean(candidate[key] for _, _, candidate in entries[:3]) for key in avg(sota)}
    entries.append(("all datasets (3-dataset macro)", base_dataset, sota_dataset))
    lines = [
        "| aggregate | baseline PSNR | SOTA PSNR (Δ) | baseline SSIM | SOTA SSIM (Δ) | baseline LPIPS | SOTA LPIPS (Δ) | baseline B(s) | SOTA B(s) (Δ) | baseline time pts | SOTA time pts (Δ) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, base, candidate in entries:
        lines.append(
            f"| {name} | {base['psnr']:.3f} | {delta(candidate['psnr'], base['psnr'], 3)} | "
            f"{base['ssim']:.4f} | {delta(candidate['ssim'], base['ssim'], 4)} | "
            f"{base['lpips']:.4f} | {delta(candidate['lpips'], base['lpips'], 4)} | "
            f"{base['time_B']:.2f} | {delta(candidate['time_B'], base['time_B'], 2)} | "
            f"{base['time_points']:.3f} | {delta(candidate['time_points'], base['time_points'], 3)} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", type=Path, default=ROOT / "md/conclusion.md")
    parser.add_argument("--json", type=Path, default=ROOT / "results/full_corpus_eval.json")
    args = parser.parse_args()
    rows = {name: [read_result(root, scene) for scene in SCENES]
            for name, root in EXPERIMENTS.items()}
    for base, candidate, scene in zip(rows["baseline"], rows["sota"], SCENES):
        if base["split"] != candidate["split"]:
            raise ValueError(f"baseline/SOTA split differs: {scene['group']}/{scene['scene']}")
        if base["iterations"] != 10000 or candidate["iterations"] != 7000:
            raise ValueError(f"unexpected iteration budget: {scene['group']}/{scene['scene']}")
        ref = load(EXPERIMENTS["baseline"] / scene["group"] / scene["scene"] /
                   "reference_3dgs/module_b.json")
        if ref.get("iterations") != 30000:
            raise ValueError(f"invalid reference: {scene['group']}/{scene['scene']}")
        budget = ref["train_seconds"]
        for row in (base, candidate):
            row["budget"] = budget
            row["budget_ratio"] = row["time_B"] / budget
            row["time_points"] = 20.0 * time_fraction(row["budget_ratio"])
            del row["split"]

    payload = {
        "scoring": {
            "weights": {"psnr": 0.2, "ssim": 0.2, "lpips": 0.4, "time_B": 0.2},
            "time_rule": "20 points at ratio<=0.10; 0 at ratio>=1; linear between",
            "quality_normalization": None,
        },
        "baseline": rows["baseline"], "sota": rows["sota"],
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n")
    overall_base = avg(rows["baseline"])
    overall_sota = avg(rows["sota"])
    psnr_better = sum(candidate["psnr"] > base["psnr"]
                      for base, candidate in zip(rows["baseline"], rows["sota"]))
    ssim_better = sum(candidate["ssim"] > base["ssim"]
                      for base, candidate in zip(rows["baseline"], rows["sota"]))
    lpips_better = sum(candidate["lpips"] < base["lpips"]
                       for base, candidate in zip(rows["baseline"], rows["sota"]))
    b_faster = sum(candidate["time_B"] < base["time_B"]
                   for base, candidate in zip(rows["baseline"], rows["sota"]))
    base_full_time = sum(row["time_points"] == 20 for row in rows["baseline"])
    sota_full_time = sum(row["time_points"] == 20 for row in rows["sota"])
    document = f"""# 三数据集全集：baseline 与质量 SOTA

日期：2026-10-30。baseline 为 `uniform16(A) + fastgs10000(B)`；SOTA 为
`uniform16(A) + topology_opacity7000(B)`。13 个场景均固定 seed=0、单次运行，
同场景 baseline/SOTA 的拆分与分辨率逐项一致。三个旧 baseline 结果的阶段
JSON 尚未显式保存 seed 字段，但其训练入口从初始提交起通过 `safe_state()` 固定
Python/NumPy/PyTorch seed=0；其余结果均显式记录 seed=0。B 预算使用同机、同场景经典
3DGS 30k 的纯训练时间（`train_seconds`），不把参考模型加载时间计入预算。

时间分严格按本轮约定计算：若 `r=B实际耗时/30k训练预算`，`r<=10%` 得20分，
`r>=100%` 得0分，中间为 `20*(1-r)/0.9`。权重为 PSNR/SSIM/LPIPS/时间
=20/20/40/20；但仓库及当前要求未给 PSNR、SSIM、LPIPS 各自映射到 `[0,1]`
的公式，因此本文不虚构综合总分，先给出全部原始质量值、delta 与可精确计算的
时间分。取得质量归一化公式后，JSON 可直接重算总分。

## Baseline 原始结果

{raw_table(rows['baseline'])}

## SOTA 原始结果及相对 baseline 的 delta

括号内均为同场景 `SOTA - baseline`；LPIPS、耗时和 B/budget 的负值表示改善。

{raw_table(rows['sota'], rows['baseline'])}

## 数据集与全集汇总

各数据集内部按场景等权；同时报告13场景等权和三个数据集等权，避免 DL3DV 的
6个场景在唯一汇总中天然获得更高权重。

{aggregate_table(rows['baseline'], rows['sota'])}

## 结论

SOTA 在 {lpips_better}/13 场景降低 LPIPS、{b_faster}/13 场景缩短 B 耗时、
{psnr_better}/13 场景提高 PSNR、{ssim_better}/13 场景提高 SSIM。13 场景等权
均值相对 baseline 的 PSNR 为 {overall_sota['psnr'] - overall_base['psnr']:+.3f} dB，
SSIM 为 {overall_sota['ssim'] - overall_base['ssim']:+.4f}，LPIPS 为
{overall_sota['lpips'] - overall_base['lpips']:+.4f}，B 耗时为
{overall_sota['time_B'] - overall_base['time_B']:+.2f} s，时间分为
{overall_sota['time_points'] - overall_base['time_points']:+.3f}/20。相对均值上，LPIPS
降低 {100 * (overall_base['lpips'] - overall_sota['lpips']) / overall_base['lpips']:.2f}%，
B 耗时降低 {100 * (overall_base['time_B'] - overall_sota['time_B']) / overall_base['time_B']:.2f}%；
时间满分场景由 baseline 的 {base_full_time}/13 增至 SOTA 的 {sota_full_time}/13。

逐场景例外是 `trashbin` 的 PSNR -0.356 dB、SSIM -0.0238，以及
`2dad2d...` 的 SSIM -0.0015；因此不能表述为所有质量指标在每个场景都改善。

质量三项的20/20/40权重不能直接乘原始 dB、SSIM 和 LPIPS；在主办方质量
归一化公式缺失时，本文不把量纲不同的原始值拼成伪“综合总分”。当前可精确
报告原始质量、逐场景 delta、B/budget 和按指定曲线计算的时间分；取得质量
归一化公式后可由保存的 JSON 直接补算总分。
"""
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(document)
    print(f"wrote {args.markdown}")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
