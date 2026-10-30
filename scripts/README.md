# 脚本

以下命令在仓库根目录运行。权重和数据集不提交到 Git。

## 下载与检查资源

```bash
bash scripts/assets/download.sh weight
bash scripts/assets/download.sh datasets
bash scripts/assets/check.sh
```

16 视角 ReSplat 权重从[作者公开的模型仓库](https://huggingface.co/haofeixu/resplat/blob/main/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth)下载，并用完整 SHA256 校验。文件保存在根目录 `pretrained/`，同时可从 `resplat/pretrained/` 读取。数据集按 README 中的主办方链接交给 `gdown` 下载；如有可访问的新链接，可设置 `DATASET_URL`。场景路径以 `baseline/scenes.json` 为准。

## 远端运行

另一台机器使用已经配置好的 `resplat` Conda 环境：

```bash
bash scripts/runs/prepare_hw.sh
bash scripts/runs/prepare_hw.sh --apply
bash scripts/runs/baseline.sh all
bash scripts/runs/baseline.sh train dl3dv
bash scripts/runs/baseline.sh eval dl3dv
```

`prepare_hw.sh` 默认预览；加 `--apply` 为缺少位姿的 hw 场景用 PyCOLMAP
生成 `sparse/0/` 和配套的 `images_undistorted/`。已有正确配对的场景会跳过，
旧的直接转换模型会归档到 `datasets/hw_backup/`。可在末尾指定场景名。

如果两个阶段使用不同环境，可设置 `FASTGS_ENV=fastgs`。运行脚本会先检查所选场景和权重，避免缺数据时被基础脚本静默跳过。
