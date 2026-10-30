"""Refresh the v1 main table from completed, separate full-pipeline runs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TARGET = ROOT / "md/main_table_v1.md"
SCENES = json.loads((ROOT / "baseline/scenes.json").read_text())
START = "<!-- V1_RESULTS_START -->"
END = "<!-- V1_RESULTS_END -->"
BACKENDS = ("terminal_guard", "topology_freeze")


def read(path):
    return json.loads(path.read_text())


def row(path):
    if not (path / "report.json").exists():
        return None
    a = read(path / "work/module_a.json")
    r = read(path / "resplat/resplat.json")
    b = read(path / "module_b/module_b.json")
    m = read(path / "module_b/metrics.json")
    manifest = read(path / "v1_run.json")
    if manifest.get("exit_code") != 0:
        raise RuntimeError(f"incomplete run has a report: {path}")
    assert m["num_views"] == len(m["per_view"])
    assert a["num_context"] == 16 if "num_context" in a else len(read(path / "work/split.json")["context"]) == 16
    return {"metrics": m["mean"], "a": a["total_seconds"],
            "resplat": r["total_seconds"], "b": b["total_seconds"],
            "total": a["total_seconds"]+r["total_seconds"]+b["total_seconds"],
            "ply_sha256": b["init_ply_sha256"], "seed": b["seed"],
            "gpu": manifest["gpu_physical_index"], "commit": manifest["git_commit"],
            "backend": b.get("adapter"), "iterations": b["iterations"]}


def cell(value, delta, decimals):
    return f"{value:.{decimals}f} ({delta:+.{decimals}f})"


def main():
    base = {}
    for group, info in SCENES.items():
        rel = Path(group) / info["scene"]
        base[group] = read(RESULTS / "baseline_A_baseline_B" / rel /
                           "module_b/metrics.json")["mean"]

    lines = ["## v1 新模块实验", "",
             "原10000步本机基线见上节；`no_reset7000`是`main_table.md`中已有的v0模块，本轮只作为配对控制。",
             "主表每个新模块、每个场景只保留一次预先指定的seed=0首跑（r0）。同seed重复运行放在下方的复验摘要，不计作新实验行。",
             "以下每行来自独立完整的A→ReSplat→B→测试渲染→评估；",
             "括号内同时给出相对原10000步基线的Δ与相同seed的本轮`no_reset7000`控制Δ，顺序为`Δ原/Δ控制`。",
             "三场景均固定16 context、7000步、相同官方权重和留出集。时间为各自同一次运行分段求和；",
             "B/30k使用历史同机同分辨率纯训练参考，双GPU同时运行可能改变时间，单独报告GPU编号。",
             "原始PLY、PNG和JSON保留在本机`results/`。", ""]
    for backend in BACKENDS:
        records = []
        path = RESULTS / f"baseline_A_{backend}7000_seed0r0_B"
        for group, info in SCENES.items():
            item = row(path / group / info["scene"])
            if item:
                records.append((path.name, group, info["scene"], item))
        if not records:
            continue
        lines.extend([f"### uniform16(A) + {backend}7000(B)", ""])
        if backend == "terminal_guard":
            lines.extend(["单项改动：只跳过第7000步的增密/剪枝及其透明度0.8限幅；总步数、学习率日程、",
                          "周期重置关闭状态与`no_reset7000`控制相同。", ""])
        elif backend == "topology_freeze":
            lines.extend(["单项改动：第5600步后停止增密，后1400步继续优化已有高斯；",
                          "总步数与学习率日程同`no_reset7000`控制。", ""])
        lines.append("| GPU | 场景 | PSNR (Δ原/Δ控制) | SSIM (Δ原/Δ控制) | LPIPS (Δ原/Δ控制) | A(s) | ReSplat(s) | B(s) (Δ控制) | 总(s) | B/30k | 输入PLY同控制 |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for name, group, scene, item in records:
            original = base[group]
            control_path = RESULTS / name.replace(backend, "no_reset", 1) / group / scene
            control = row(control_path) if control_path.exists() else None
            old_metrics = original

            def metric(key, digits):
                value = item["metrics"][key]
                vs_control = value-control["metrics"][key] if control else float("nan")
                return f"{value:.{digits}f} ({value-old_metrics[key]:+.{digits}f}/{vs_control:+.{digits}f})"

            reference = read(RESULTS / "baseline_A_baseline_B" / group / scene /
                             "reference_3dgs/module_b.json")
            budget = reference["train_seconds"]
            b_delta = item["b"]-control["b"] if control else float("nan")
            same_ply = "是" if control and item["ply_sha256"] == control["ply_sha256"] else "待核/否"
            lines.append("| " + " | ".join([
                str(item["gpu"]), group + "/" + scene[:12],
                metric("psnr", 3), metric("ssim", 4), metric("lpips", 4),
                f'{item["a"]:.2f}', f'{item["resplat"]:.2f}',
                cell(item["b"], b_delta, 2), f'{item["total"]:.2f}',
                f'{100*item["b"]/budget:.2f}%', same_ply]) + " |")
        commits = ", ".join(f"`{revision}`" for revision in sorted({
            rec[3]["commit"][:12] for rec in records}))
        lines.extend(["", f"运行记录中的提交：{commits}；逐行提交见本机 `v1_run.json`。",
                      "本表为单次结果；重复波动见下方复验摘要。", ""])
        if backend == "terminal_guard" and len(records) == 3:
            lines.extend(["DL3DV的LPIPS较配对控制低0.0028；HW低约0.0010；counter在四位小数口径不变。",
                          "B用时仅DL3DV缩短，另两场景增加；本次单项结果不能宣称跨场景加速，",
                          "也不能仅凭DL3DV一次改善确认通用质量收益。", ""])

    lines.extend(["### 已有配置 `no_reset7000` 的复验摘要（非新模块）", "",
                  "历史配置与结果见`main_table.md`。为评估本轮运行波动，使用相同seed=0独立执行两次完整管线；",
                  "`r0`/`r1`是第一次/第二次运行，不代表不同算法或不同seed。以下每个场景只占一行。",
                  "双GPU调度时，某场景两次可能落在不同GPU，因此B用时仅作波动参考。", "",
                  "| 场景 | v0 PSNR / LPIPS | 本轮r0 PSNR / LPIPS / B(s) | 本轮r1 PSNR / LPIPS / B(s) | 输入PLY一致 |",
                  "|---|---|---|---|---|"])
    for group, info in SCENES.items():
        scene = info["scene"]
        rel = Path(group) / scene
        old = read(RESULTS / "baseline_A_no_reset7000_B" / rel / "module_b/metrics.json")["mean"]
        control = [row(RESULTS / f"baseline_A_no_reset7000_seed0r{i}_B" / rel) for i in (0, 1)]

        def summary(item):
            return (f'{item["metrics"]["psnr"]:.3f} / {item["metrics"]["lpips"]:.4f} / '
                    f'{item["b"]:.2f}') if item else "—"

        same = ("是" if all(control) and control[0]["ply_sha256"] == control[1]["ply_sha256"]
                else "待核")
        lines.append(f'| {group}/{scene[:12]} | {old["psnr"]:.3f} / {old["lpips"]:.4f} | '
                     f'{summary(control[0])} | {summary(control[1])} | {same} |')
    lines.append("")

    source = TARGET.read_text()
    content = "\n".join(lines).rstrip()+"\n"
    if START in source:
        pre = source.split(START)[0]
        post = source.split(END)[1]
        output = pre+START+"\n"+content+END+post
    else:
        output = source.replace("待首轮实验填入。每个重复运行使用唯一目录；`results/` 中的原始 JSON、PLY、PNG 与日志留在本机。",
                                START+"\n"+content+END)
    TARGET.write_text(output)
    print(f"updated {TARGET} with one main-table row per completed v1 module and scene")


if __name__ == "__main__":
    main()
