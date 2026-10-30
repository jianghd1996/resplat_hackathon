# 我们的 A/B 插件实验入口

本仓库固定使用赛题 ReSplat 代码与预训练权重，只替换前处理 A 与接续优化 B。当前可运行的组合是 `uniform16`（baseline A）和 `fastgs`（baseline B）。A 对训练视角按图像名称排序后用 `linspace` 均匀选最多 16 张；B 从 ReSplat 导出的 PLY 用 FastGS 训练 10000 步。接口分别在 `baseline/utils_A/` 与 `baseline/utils_B/`；后续 agent 的验证与记录流程见 `md/agent_workflow.md`，技术优先级见 `md/plan_v0.md`，历史和复现表格见 `md/main_table.md`。

## 数据与环境

仓库根目录下三组数据的输入格式如下；`baseline/scenes.json` 当前各选一个场景。

| 数据组 | 本机目录 | 原始相机/位姿输入 | 图像 | 默认场景 |
|---|---|---|---|---|
| DL3DV | `datasets/DL3DV/<hash>/` | `transforms.json` | `images_4/`，960×540 | `001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f` |
| hw | `datasets/hw/<scene>/` | `frames_info.txt` 对应 JPG；PyCOLMAP 建图/去畸变后使用 `sparse/0/` + `images_undistorted/` | 原始 `images/`；不要混入额外 PNG | `billboard` |
| mipnerf360 | `datasets/mipnerf360/<scene>/` | `sparse/0/{cameras,images}.bin` | `images/` | `counter` |

当前服务器仓库位于 `/mnt/DataPart/jianghongda/wangyanbin_resplat/resplat_hackathon`。两个阶段均使用 `/mnt/DataPart/jianghongda/env/resplat/bin/python`：其中 PyTorch 2.7.0+cu126、PyCOLMAP 3.10.0 和 FastGS CUDA 扩展已可导入。hw 不依赖 COLMAP CLI；仅当没有 `frames_info.txt` 且没有现成位姿时，现有 A 才走 COLMAP CLI。权重在仓库根目录的 `pretrained/`；正式脚本会优先查找 `resplat/pretrained/`，找不到时自动使用根目录权重，也可用 `CHECKPOINT` 显式指定。

## 正式三场景训练和测试

确认 `billboard/sparse/0/pycolmap_undistorted.json` 与配套的
`images_undistorted/` 均存在后运行；缺少位姿时先按 `baseline/README.md` 的
PyCOLMAP 命令建图：

```bash
cd /mnt/DataPart/jianghongda/wangyanbin_resplat/resplat_hackathon
cd baseline
RESPLAT_ENV=/mnt/DataPart/jianghongda/env/resplat FASTGS_ENV=/mnt/DataPart/jianghongda/env/resplat \
  bash run_train.sh --module-a uniform16 --module-b fastgs
RESPLAT_ENV=/mnt/DataPart/jianghongda/env/resplat FASTGS_ENV=/mnt/DataPart/jianghongda/env/resplat \
  bash run_eval.sh --module-a uniform16 --module-b fastgs
python collect_results.py --experiment baseline_A_baseline_B
```

`run_train.sh` / `run_eval.sh` 无数据组参数时按 `baseline/scenes.json` 跑三场景；末尾加 `hw` 等组名可只跑对应场景。训练和评测写入仓库根目录 `results/baseline_A_baseline_B/<group>/<scene>/`，其中 `work/` 是 A 的标准化 COLMAP 输入，`resplat/` 为固定模型输出，`module_b/` 有训练 PLY、测试 PNG 与 `metrics.json`。汇总为 `results/baseline_A_baseline_B/summary.{json,md}`。整个 `results/` 被 Git 忽略，实验记录留在服务器。

## 官方要求的单场景一键入口

`baseline/run_pipeline.py` 接受任意三种格式的场景目录，并在指定输出目录完成 A → ReSplat → B → 测试渲染 → 指标计算；输出 `model.ply`、`module_b/test_renders/*.png`、`module_b/metrics.json` 和 `report.json`。例如：

```bash
cd /mnt/DataPart/jianghongda/wangyanbin_resplat/resplat_hackathon
/mnt/DataPart/jianghongda/env/resplat/bin/python baseline/run_pipeline.py \
  --scene datasets/hw/billboard \
  --output results/baseline_A_baseline_B/hw/billboard \
  --module-a uniform16 --module-b fastgs \
  --resplat-python /mnt/DataPart/jianghongda/env/resplat/bin/python \
  --fastgs-python /mnt/DataPart/jianghongda/env/resplat/bin/python
```

只需把 `--scene` 与 `--output` 换成 `datasets/DL3DV/<hash>` / `results/baseline_A_baseline_B/dl3dv/<hash>` 或 `datasets/mipnerf360/counter` / `results/baseline_A_baseline_B/mipnerf360/counter`。默认 16 个 context、B 训练 10000 步；如需显式指定权重可加 `--checkpoint pretrained/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth`。实验输出遵循 `xx_A_xx_B` 目录命名；新插件加入后先在 CLI 注册，再使用对应的组合目录。两种入口使用同一 A 选择器和 B 适配器。

当前 billboard 的旧 `tmp` 报告也可以只在服务器端归档：

```bash
mkdir -p results/baseline_A_baseline_B/hw/billboard
cp /mnt/DataPart/jianghongda/wangyanbin_resplat/tmp/billboard_verify.z28Bi5/billboard_report.json \
  results/baseline_A_baseline_B/hw/billboard/report.json
python baseline/collect_results.py --experiment baseline_A_baseline_B
```

收集命令打印 Markdown 表格和 JSON，可复制给维护总表的人；它也会读取常规 `metrics.json` 和各阶段计时文件。计时口径需同时记录位姿是否预先缓存：复用 `sparse/0` 的 A 时间不包含首次 PyCOLMAP 建图。

## 本机实验命令

本机两个 Conda 环境分别为 `/home/wangyb/miniconda3/envs/resplat` 和
`/home/wangyb/miniconda3/envs/fastgs`。从仓库根目录运行，例如单 A 的 FPS：

```bash
cd baseline
RESPLAT_ENV=/home/wangyb/miniconda3/envs/resplat \
FASTGS_ENV=/home/wangyb/miniconda3/envs/fastgs \
  bash run_train.sh --module-a fps --module-b fastgs
RESPLAT_ENV=/home/wangyb/miniconda3/envs/resplat \
FASTGS_ENV=/home/wangyb/miniconda3/envs/fastgs \
  bash run_eval.sh --module-a fps --module-b fastgs
python collect_results.py --experiment fps_A_baseline_B --format both
```

短预算 B 通过 `ITERATIONS` 统一设置训练和评估步数，例如 `ITERATIONS=7000`
配合 `--module-a uniform16 --module-b short_budget`，结果写到
`results/baseline_A_short_budget7000_B/`。`short_schedule` 在同样步数上只把
位置学习率衰减终点缩至该预算，结果写到
`results/baseline_A_short_schedule7000_B/`。两个模块都保留原始失败结果及
阶段 JSON；详细假设、判据和实测结论见 `md/plan_v0.md` 与 `md/main_table.md`。
P2 单 A 可用 `--module-a quality_filter` 或 `--module-a coverage_local`
与 `--module-b fastgs` 分别测试。P2 最终剪枝探针使用 `ITERATIONS=7000`
及 `--module-a uniform16 --module-b regularized`，结果写入
`results/baseline_A_regularized7000_B/`；它从同样的 ReSplat PLY 开始，
剪枝耗时计入 B。跨场景结论以总表实际完成的实验为准。
收集脚本会从同机同场景的 `baseline_A_baseline_B` 目录读取 30k 参考并在
JSON 的 `reference_source` 标明来源；各实验的 A、ReSplat、B 时间始终从
该实验自身的阶段 JSON 读取，同时输出该行相对同机基线的 Δ。
