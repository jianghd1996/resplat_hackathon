"""Refresh the v1 main table from completed, separate full-pipeline runs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TARGET = ROOT / "md/main_table_v1.md"
SCENES = json.loads((ROOT / "baseline/scenes.json").read_text())
START = "<!-- V1_RESULTS_START -->"
END = "<!-- V1_RESULTS_END -->"
BACKENDS = ("no_reset", "terminal_guard", "topology_freeze")


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

    lines = ["## v1 完整管线实验", "",
             "原10000步本机基线见上节。以下每行来自独立完整的A→ReSplat→B→测试渲染→评估；",
             "括号内同时给出相对原10000步基线的Δ与相同seed/重复号的v1 `no_reset7000`控制Δ，顺序为`Δ原/Δ控制`。",
             "控制自身的`Δ控制`为0；无对应控制的候选不进入配对比较。",
             "三场景均固定16 context、7000步、相同官方权重和留出集。时间为各自同一次运行分段求和；",
             "B/30k使用历史同机同分辨率纯训练参考，双GPU同时运行可能改变时间，单独报告GPU编号。",
             "原始PLY、PNG和JSON保留在本机`results/`。", ""]
    for backend in BACKENDS:
        records = []
        for path in sorted(RESULTS.glob(f"baseline_A_{backend}7000_seed*_B")):
            for group, info in SCENES.items():
                item = row(path / group / info["scene"])
                if item:
                    records.append((path.name, group, info["scene"], item))
        if not records:
            continue
        lines.extend([f"### uniform16(A) + {backend}7000(B)", ""])
        lines.append("| 运行 | GPU | 场景 | PSNR (Δ原/Δ控制) | SSIM (Δ原/Δ控制) | LPIPS (Δ原/Δ控制) | A(s) | ReSplat(s) | B(s) (Δ控制) | 总(s) | B/30k | 输入PLY同控制 |")
        lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
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
                name.split("_seed")[-1][:-2], str(item["gpu"]), group + "/" + scene[:12],
                metric("psnr", 3), metric("ssim", 4), metric("lpips", 4),
                f'{item["a"]:.2f}', f'{item["resplat"]:.2f}',
                cell(item["b"], b_delta, 2), f'{item["total"]:.2f}',
                f'{100*item["b"]/budget:.2f}%', same_ply]) + " |")
        lines.extend(["", f"运行代码提交：`{records[0][3]['commit'][:12]}`。",
                      "指标与时间为单次原始测量；同seed重复与跨seed波动需要在结论中单独说明。", ""])

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
    print(f"updated {TARGET} from {len([1 for b in BACKENDS for p in RESULTS.glob(f'baseline_A_{b}7000_seed*_B') for g,i in SCENES.items() if (p/g/i['scene']/'report.json').exists()])} completed scene-runs")


if __name__ == "__main__":
    main()
