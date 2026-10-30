"""Write v1 new-module results in the exact column order used by main_table.md.

One fixed seed (FastGS safe_state=0), one completed run per module and scene.
Archived diagnostic repeats are never read by this collector.
"""
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TARGET = ROOT / "md/main_table_v1.md"
SCENES = json.loads((ROOT / "baseline/scenes.json").read_text())
START = "<!-- V1_RESULTS_START -->"
END = "<!-- V1_RESULTS_END -->"
BACKENDS = ("terminal_guard", "topology_freeze", "opacity_state", "topology_opacity")


def read(path):
    return json.loads(path.read_text())


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def collect(path):
    a = read(path / "work/module_a.json")
    r = read(path / "resplat/resplat.json")
    b = read(path / "module_b/module_b.json")
    m = read(path / "module_b/metrics.json")
    split = read(path / "work/split.json")
    assert len(split["context"]) == 16
    assert m["num_views"] == len(split["test"]) == len(m["per_view"])
    assert b.get("seed", 0) == 0
    return {"metric": m["mean"], "A": a["total_seconds"],
            "ReSplat": r["total_seconds"], "B": b["total_seconds"],
            "total": a["total_seconds"] + r["total_seconds"] + b["total_seconds"],
            "ply_sha256": b.get("init_ply_sha256")}


def fmt(value, old, precision):
    return f"{value:.{precision}f} ({value-old:+.{precision}f})"


def main():
    lines = ["## v1 新模块单项实验", "",
             "统一使用 FastGS `safe_state` 的 seed=0，每个模块与场景仅记录一次完整运行。",
             "本表沿用上方基线的列顺序：括号内 Δ 为该行减本机原10000步 `baseline(A)+baseline(B)`；",
             "三个阶段和总耗时来自该行同一次运行。`no_reset7000` 的既有结果保留在 `main_table.md`，",
             "这里只在文字中用于解释机制，不作为 v1 新模块重新列行。",
             "数据、位姿、留出集、官方权重与分辨率保持一致；HW 使用预建 PyCOLMAP 位姿。",
             "两张 RTX 3090 运行任务，表格不区分 GPU；同机经典30k参考沿用已实测值。", ""]
    completed = 0
    for backend in BACKENDS:
        root = RESULTS / f"baseline_A_{backend}7000_B"
        if not root.is_dir():
            continue
        rows = []
        for group, info in SCENES.items():
            scene = info["scene"]
            path = root / group / scene
            if not (path / "report.json").exists():
                continue
            historical_split = read(RESULTS / "baseline_A_no_reset7000_B" / group / scene /
                                    "work/split.json")
            current_split = read(path / "work/split.json")
            for key in ("train", "test", "context", "resolution"):
                assert current_split[key] == historical_split[key], (
                    "split or context differs from control", group, scene, key)
            candidate = collect(path)
            baseline = collect(RESULTS / "baseline_A_baseline_B" / group / scene)
            reference = read(RESULTS / "baseline_A_baseline_B" / group / scene /
                             "reference_3dgs/module_b.json")
            assert candidate["ply_sha256"] is not None
            rows.append((group, scene, candidate, baseline, reference, path))
        if not rows:
            continue
        completed += len(rows)
        lines.extend([f"### baseline(A) + {backend}7000(B)", ""])
        if backend == "terminal_guard":
            lines.extend(["B 代码为 `baseline/utils_B/terminal_guard.py`：沿用 `no_reset7000` 的7000步、",
                          "位置学习率日程和关闭周期性opacity重置设置，仅跳过第7000步增密/剪枝事件。",
                          "参数跨三场景相同。", ""])
        else:
            if backend == "topology_freeze":
                lines.extend(["B 代码为 `baseline/utils_B/topology_freeze.py`：沿用 `no_reset7000` 的7000步日程，",
                              "仅将密集化截止改为5600步，随后1400步优化已有高斯。参数跨三场景相同。", ""])
            else:
                if backend == "opacity_state":
                    lines.extend(["B 代码为 `baseline/utils_B/opacity_state.py`：保持7000步、密集化、0.8 opacity限幅",
                                  "和关闭周期性重置不变；只为未触发限幅的高斯保留opacity Adam动量。参数跨三场景相同。", ""])
                else:
                    lines.extend(["B 代码为 `baseline/utils_B/topology_opacity.py`：组合已完成的两个单项，",
                                  "在5600步停止密集化，并为未触发0.8限幅的高斯保留opacity Adam动量；",
                                  "总预算仍为7000步，参数跨三场景相同。", ""])
        lines.extend(["| **group** | **scene** | **PSNR (Δ)** | **SSIM (Δ)** | **LPIPS (Δ)** | **A(s) (Δ)** | **ReSplat(s) (Δ)** | **B(s) (Δ)** | **总耗时(s) (Δ)** | **经典30k参考** |",
                      "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"])
        for group, scene, new, old, ref, path in rows:
            historical_ply = (RESULTS / "baseline_A_no_reset7000_B" / group / scene /
                              "resplat/resplat_world.ply")
            assert new["ply_sha256"] == file_sha256(historical_ply), (
                "ReSplat input differs from v0 no_reset control", group, scene)
            fields = [group, (scene[:8]+"…" if group == "dl3dv" else scene),
                      fmt(new["metric"]["psnr"], old["metric"]["psnr"], 3),
                      fmt(new["metric"]["ssim"], old["metric"]["ssim"], 4),
                      fmt(new["metric"]["lpips"], old["metric"]["lpips"], 4),
                      fmt(new["A"], old["A"], 3),
                      fmt(new["ReSplat"], old["ReSplat"], 3),
                      fmt(new["B"], old["B"], 3),
                      f'**{fmt(new["total"], old["total"], 3)}**',
                      f'{ref["total_seconds"]}s / {ref["test_psnr_quick"]}']
            lines.append("| " + " | ".join(fields) + " |")
        lines.extend(["", "源数据：本机 `results/` 中该模块的各场景 `work/module_a.json`、",
                      "`resplat/resplat.json`、`module_b/module_b.json`、`module_b/metrics.json`；",
                      "结果目录和 `v1_run.json` 记录了代码提交及实际运行命令。", ""])
        if backend == "topology_freeze" and len(rows) == 3:
            lines.extend(["相对 `main_table.md` 的 `no_reset7000`，三场景 PSNR 分别变化",
                          "+0.936/+0.040/+0.397 dB，LPIPS 分别变化 -0.0108/-0.0020/-0.0078；",
                          "最终高斯数反而增加 6863/5193/8614（约2.75%/1.46%/2.64%）。",
                          "因此现有证据支持“固定拓扑后的连续恢复”，不支持把收益解释为更小模型。", ""])
        if backend == "opacity_state" and len(rows) == 3:
            lines.extend(["相对 `main_table.md` 的 `no_reset7000`，三场景 LPIPS 分别变化",
                          "-0.0099/-0.0038/-0.0129，但最终高斯数增加约23.5%/17.6%/45.4%，",
                          "B 用时增加10.33/3.45/8.50秒；counter的B约为经典30k纯训练时间的11.0%。",
                          "该单项属于质量优先候选，尚未保持10%时间档。", ""])
    a_root = RESULTS / "covisibility_guarded_A_no_reset7000_B"
    if a_root.is_dir():
        rows = []
        for group, info in SCENES.items():
            scene = info["scene"]
            path = a_root / group / scene
            if not (path / "report.json").exists():
                continue
            historical_split = read(RESULTS / "baseline_A_no_reset7000_B" / group / scene /
                                    "work/split.json")
            current_split = read(path / "work/split.json")
            for key in ("train", "test", "resolution"):
                assert current_split[key] == historical_split[key]
            candidate = collect(path)
            baseline = collect(RESULTS / "baseline_A_baseline_B" / group / scene)
            reference = read(RESULTS / "baseline_A_baseline_B" / group / scene /
                             "reference_3dgs/module_b.json")
            rows.append((group, scene, candidate, baseline, reference, path))
        if rows:
            completed += len(rows)
            lines.extend(["### covisibility_guarded(A) + no_reset7000(B)", "",
                          "A 代码为 `baseline/utils_A/covisibility_guarded.py`：从均匀16帧开始，保护序列端点和分区，",
                          "最多交换2帧；只接受训练轨迹并集增加、每视图支持p10不降、平均最近相机距离不超过",
                          "均匀策略1.1倍且最大序号间隙不超过2倍的候选。缺少训练轨迹时回退均匀16帧。",
                          "B 固定为既有 `no_reset7000`，只测试A。", "",
                          "| **group** | **scene** | **PSNR (Δ)** | **SSIM (Δ)** | **LPIPS (Δ)** | **A(s) (Δ)** | **ReSplat(s) (Δ)** | **B(s) (Δ)** | **总耗时(s) (Δ)** | **经典30k参考** |",
                          "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"])
            for group, scene, new, old, ref, path in rows:
                fields = [group, (scene[:8]+"…" if group == "dl3dv" else scene),
                          fmt(new["metric"]["psnr"], old["metric"]["psnr"], 3),
                          fmt(new["metric"]["ssim"], old["metric"]["ssim"], 4),
                          fmt(new["metric"]["lpips"], old["metric"]["lpips"], 4),
                          fmt(new["A"], old["A"], 3), fmt(new["ReSplat"], old["ReSplat"], 3),
                          fmt(new["B"], old["B"], 3), f'**{fmt(new["total"], old["total"], 3)}**',
                          f'{ref["total_seconds"]}s / {ref["test_psnr_quick"]}']
                lines.append("| " + " | ".join(fields) + " |")
            lines.extend(["", "源数据与校验口径同上；三场景参数统一，选择器不读取测试图像或测试指标。", ""])
    if not completed:
        lines.append("正式单次实验尚未完成；早期重复运行只用于诊断，未入主表。")
    text = TARGET.read_text()
    if START not in text or END not in text:
        raise RuntimeError("v1 result markers missing")
    TARGET.write_text(text.split(START)[0] + START + "\n" + "\n".join(lines).rstrip() +
                      "\n" + END + text.split(END, 1)[1])
    print(f"updated {TARGET}: {completed} official scene rows")


if __name__ == "__main__":
    main()
