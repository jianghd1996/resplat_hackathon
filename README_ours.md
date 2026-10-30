# 我们的 A/B 插件实验入口

本仓库固定使用赛题 ReSplat 代码与预训练权重，只替换前处理 A 与接续优化 B。当前可运行的组合是 `uniform16`（baseline A）和 `fastgs`（baseline B）。A 对训练视角按图像名称排序后用 `linspace` 均匀选最多 16 张；B 从 ReSplat 导出的 PLY 用 FastGS 训练 10000 步。接口分别在 `baseline/utils_A/` 与 `baseline/utils_B/`，技术优先级见 `md/plan_v0.md`，历史和复现表格见 `md/main_table.md`。

## 数据与环境

仓库根目录下三组数据的输入格式如下；`baseline/scenes.json` 当前各选一个场景。

| 数据组 | 本机目录 | 原始相机/位姿输入 | 图像 | 默认场景 |
|---|---|---|---|---|
| DL3DV | `datasets/DL3DV/<hash>/` | `transforms.json` | `images_4/`，960×540 | `001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f` |
| hw | `datasets/hw/<scene>/` | `frames_info.txt` 对应 JPG；PyCOLMAP 建图/去畸变后使用 `sparse/0/` + `images_undistorted/` | 原始 `images/`；不要混入额外 PNG | `billboard` |
| mipnerf360 | `datasets/mipnerf360/<scene>/` | `sparse/0/{cameras,images}.bin` | `images/` | `counter` |

当前服务器仓库位于 `/mnt/DataPart/jianghongda/wangyanbin_resplat/resplat_hackathon`。两个阶段均使用 `/mnt/DataPart/jianghongda/env/resplat/bin/python`：其中 PyTorch 2.7.0+cu126、PyCOLMAP 3.10.0 和 FastGS CUDA 扩展已可导入。hw 不依赖 COLMAP CLI；仅当没有 `frames_info.txt` 且没有现成位姿时，现有 A 才走 COLMAP CLI。权重在仓库根目录的 `pretrained/`；正式脚本会优先查找 `resplat/pretrained/`，找不到时自动使用根目录权重，也可用 `CHECKPOINT` 显式指定。

## 正式三场景训练和测试

待正在运行的 hw 重建完成、`billboard/sparse/0/pycolmap_undistorted.json` 与 `images_undistorted/` 均存在后：

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
