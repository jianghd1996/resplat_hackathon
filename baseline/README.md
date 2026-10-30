# ReSplat 竞赛 Baseline（A → ReSplat → B）

经典三段式基线：**模块 A**（位姿获取 + 均匀 16 视角选择）→ **ReSplat**（16 视角前馈重建）→
**模块 B**（全训练视角 3DGS 接续训练 10000 步）。每阶段独立计时，最终输出
PSNR / SSIM / LPIPS + 总耗时。仓库根目录 `datasets/` 三组数据各跑一个场景，一键运行。

## 一键运行

```bash
# 0) 仅在缺少 CUDA 扩展的新机器上运行：bash setup.sh

# 1) 一键端到端训练：A -> ReSplat -> B（三组数据各一个场景）
bash run_train.sh --module-a uniform16 --module-b fastgs  # 全部；末尾可指定 dl3dv hw

# 2) 一键评测：渲染测试视角 + PSNR/SSIM/LPIPS + 时间汇总表
bash run_eval.sh --module-a uniform16 --module-b fastgs  # 同上可指定组

# 可选）经典 3DGS 30k 从头训练参考（选手需要打败的时间线）
bash run_reference.sh dl3dv
```

产物：
```
../results/baseline_A_baseline_B/<group>/<scene>/
  work/                         # 模块 A 产物（COLMAP + split.json + module_a.json）
  resplat/                       # resplat_world.ply/.npz + resplat.json（计时）
  module_b/                      # 训练模型 + module_b.json（计时）+ metrics.json
  reference_3dgs/                # （可选）30k 参考
../results/baseline_A_baseline_B/summary.{json,md}  # 汇总表
```

## 环境

- **resplat 环境**（模块 A / ReSplat / 评测指标）：按 `../resplat/README.md` 安装
  （python 3.12, torch 2.7, gsplat 1.5.3, pointops, PyCOLMAP 3.10），并安装官方权重：
  `resplat/pretrained/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth`
  （选手从主办方提供的链接下载，放入 `../resplat/pretrained/`，或用
  `run_resplat.py --checkpoint` 指定路径）
- **FastGS / 模块 B**：需要 `simple_knn`、`fused_ssim` 和
  `diff_gaussian_rasterization_fastgs` CUDA 扩展。新机器可按 `setup.sh` 准备；
  该脚本还会从指定 FastGS 提交获取光栅器源码。
- **数据集**：放在仓库根目录的 `datasets/DL3DV`、`datasets/hw`、
  `datasets/mipnerf360`；实际场景由 `baseline/scenes.json` 指定。

当前服务器的数据目录（共 13 个场景；以下数字为已观察到的本地目录结构）：

```text
datasets/
├── DL3DV/                 6 个场景，<场景哈希>/images_4/ + transforms.json
│                          每场景约 305–381 张，图像 960×540
├── hw/                    4 个场景：car、basketball、billboard、trashbin
│   └── <场景>/             images/ + 原始 frames_info.txt
│       ├── sparse/0/       当前供模块 A 读取的相机模型
│       └── images_undistorted/  与新 PyCOLMAP 模型配对的去畸变图像
└── mipnerf360/            3 个场景：bicycle、bonsai、counter
    └── <场景>/             images/ + sparse/0/{cameras,images}.bin
```

hw 的 `images/` 每场景还含一张与原始 JPG 分辨率不同的 PNG；默认重建只选择
`frames_info.txt` 时间戳对应的 JPG。旧的 `sparse_0/` 和 `.pycolmap/` 是历史
工作文件，不是模块 A 默认读取路径。`baseline/scenes.json` 当前各选一个场景：
DL3DV 的 `001dccbc…5f`、hw 的 `billboard`、mipnerf360 的 `counter`。
`images_undistorted/` 只有在运行新重建工具后才会出现；若仍有
`sparse/0/conversion_summary.json`，说明该场景还是旧的直接转换结果。

### 当前服务器环境（供本地 agent 使用）

- 仓库在 `/mnt/DataPart/jianghongda/wangyanbin_resplat/resplat_hackathon`；
  已配置的 Python 是 `/mnt/DataPart/jianghongda/env/resplat/bin/python`，
  其中 PyTorch 为 `2.7.0+cu126`、PyCOLMAP 为 `3.10.0`，GPU 可用。
  这同一环境也能导入上述三个 FastGS CUDA 扩展，因此训练和评测均可用它。
- `/mnt/DataPart/jianghongda/env/3dgs` 是另一个环境（PyTorch `2.6.0+cu124`），
  未安装 `simple_knn`，不能直接作为本仓库的 FastGS 环境。服务器有
  `/usr/local/cuda-12.8/bin/nvcc`。Conda 列表里还有同名的其他 `resplat`
  环境，请用绝对路径选择上面的环境。
- 本机的模型权重位于仓库根目录 `pretrained/`；`run_resplat.py` 默认查找
  `resplat/pretrained/`。直接调用时传 `--checkpoint` 指向根目录中的权重；
  `run_train.sh` 会自动在两处查找，也可设置 `CHECKPOINT`。
- hw 的默认位姿重建使用 PyCOLMAP，不需要 `colmap` CLI；输入为
  `frames_info.txt` 对应的原始 JPG，并执行特征提取、匹配、建图和去畸变。
  无位姿且没有 `frames_info.txt` 的其他数据，仍走 COLMAP CLI 分支。

在该服务器运行一键脚本时，明确指定两个环境为同一绝对路径：

```bash
cd /mnt/DataPart/jianghongda/wangyanbin_resplat/resplat_hackathon
cd baseline
RESPLAT_ENV=/mnt/DataPart/jianghongda/env/resplat \
FASTGS_ENV=/mnt/DataPart/jianghongda/env/resplat \
bash run_train.sh hw
```

## 流水线设计（baseline 的具体选择）

### 模块 A（`module_a/preprocess.py`，resplat 环境）

1. **位姿获取（有则跳过，无则估计）**，按来源优先级：
   - `sparse/0/{cameras,images}.{bin,txt}`（mipnerf360）→ 直接使用
   - `transforms.json`（DL3DV）→ 转 COLMAP（OpenGL→OpenCV 轴翻转 + 内参缩放，不估计）
   - `frames_info.txt`（hw 采集数据）→ **用 PyCOLMAP 重新建图并去畸变**。
     只用 `frames_info.txt` 中的时间戳筛选原始 JPG；不使用其位置值。
     PyCOLMAP GPU SIFT 不可用时自动改用 CPU。已有 `sparse/0` 若带
     `conversion_summary.json`（直接转换的旧位姿），模块 A 会忽略并重建。
     已建好的去畸变模型可用 `images_undistorted/` 和
     `sparse/0/pycolmap_undistorted.json` 配对保存；不能只替换相机文件。
   - 什么都没有 → 仍使用 COLMAP CLI 估计位姿
2. **分辨率**：长边 ≤ 960 且 H/W 为 64 的倍数（LANCZOS 缩放，内参逐轴缩放），
   全流程（ReSplat/B/评测）统一使用该分辨率；**竖屏输入长边上限压到 768**
   （参考 `benchmarks/gs_compare_20261020` 的策略：竖屏 ViT token 数约为
   等像素横屏的 1.9×，960 长边会使跨视角 cost-volume 超 24GB）
3. **竖屏处理**：图像**保持原方向不旋转**；`resplat/src/model/encoder/
   unimatch/mv_unimatch.py` 中的 `assert ori_h <= ori_w` 按该 benchmark
   同款方式放宽为竖屏 no-op（mono-ViT 的 resize 逻辑与宽高比无关，
   该补丁对横屏输入行为不变）
4. **划分**：llffhold=8（与经典 3DGS 一致，每 8 张留 1 张作测试）；
   16 个 context 视角从**训练集**中 `np.linspace` 均匀选出（不接触测试视角）

已用 `frames_info_to_colmap.py` 直接转换过的 hw 场景，可以先用
`python baseline/rebuild_hw_sparse.py --root datasets/hw --backup-dir /path/to/backup`
预览。确认后加 `--apply`：脚本只处理带 `sparse/0/conversion_summary.json`
标记的旧模型，从原始 JPG 重新建图，把旧 `sparse/` 归档到指定目录，并保存
新的 `sparse/0/` 与配套的 `images_undistorted/`。`sparse_0/` 不会被改动。

### ReSplat 阶段（`run_resplat.py`，resplat 环境）

- `dl3dv_16v_540x960` 官方权重；context = A 选出的 16 张，
  refinement target = 同样 16 张（自监督精修，不向 ReSplat 输入额外视角）
- `infer_colmap.build_batch` 会把位姿归一化到中间 context 视角（pivot），
  高斯位于该归一化坐标系；导出时需要**完整逆变换**（本 baseline 的关键修正，
  已用渲染 PSNR 平价验证）：
  1. means：`X_world = pivot_c2w · X_norm`
  2. rotations：ReSplat 的 decoder 用**预计算协方差**渲染——
     `cov = R_ctx_norm[v] · R_local · Σ · R_localᵀ · R_ctx_norm[v]ᵀ`，
     即每个高斯的世界旋转 = **源 context 视角的世界旋转 × 局部旋转**
     （高斯按 v-major 排序，第 g 个的源视角 = g // (N/16)）
  3. SH 系数：光栅器在世界系求视线方向，刚体旋转后系数需做
     **Wigner-D 旋转**（按 pivot 旋转，逐阶块对角）
  4. far 平面剔除 + opacity<0.005 剪枝（对齐 ReSplat 渲染行为与
     模块 B 自身的剪枝阈值）
- 导出**经典 3DGS PLY 布局**（f_rest 按 channel-major 展平，与
  save_ply/load_ply 往返一致），同时保存 npz 便于调试

### 模块 B（`module_b/train_from_ply.py`，fastgs 环境）

- 经典 3DGS（Inria graphdeco）训练循环：L1+SSIM(0.2)、自适应致密化
  （grad 0.0002、interval 100、opacity reset 3000）、SH warmup
- **全训练视角**（llffhold 划分的 train 集）接续训练 **10000 步**，
  超参按 30k 经典配置折半（position_lr_max_steps=10k、densify_until_iter=5k）
- 初始化 = ReSplat 世界系 PLY（`--init_ply`）；`--init_random` 则从随机
  视锥点云开始（供 30k 参考计时用）

### 评测（`render_test.py` + `evaluate.py`）

- fastgs 环境渲染测试视角 → resplat 环境**官方 ReSplat 指标实现**
  （PSNR/SSIM/VGG-LPIPS，与榜单同源）对比 GT → `metrics.json`
- `summarize.py` 汇总各阶段计时与指标

## 历史基线结果（RTX 4090 24GB）

| group | scene | PSNR | SSIM | LPIPS | A(s) | resplat(s) | B(s) | **total(s)** | 经典30k参考(s) | 参考 PSNR |
|---|---|---|---|---|---|---|---|---|---|---|
| dl3dv | 001dccbc…5f | 20.47 | 0.652 | 0.457 | 31.5 | 18.1 | 213.7 | **263.2** | 435.0 | 19.49 |
| hw | billboard | 22.68 | 0.799 | 0.159 | — | — | — | **143** | 915 | 16.75 |
| mipnerf360 | counter | 14.43 | 0.453 | 0.688 | 54.5 | 18.9 | 283.8 | **357.3** | 736.6 | 11.77 |

`billboard` 行取自仓库根目录 README 已记录的结果；该记录没有保存 A、ReSplat、
B 的分段耗时，因此用 `—` 表示，不能用之前 `basketball` 的分段耗时填入。
其余两行是较早的复跑记录。仓库未提交这些历史运行的 `metrics.json` 和日志；
在当前服务器复跑时，应以新生成的 `metrics.json`、`module_a.json`、
`resplat.json` 和 `module_b.json` 为准，不要把不同运行的时间相加。

- 表中三个场景的总耗时都低于所列经典 3DGS 30k 参考耗时，PSNR 也更高。
- mipnerf360 的绝对指标较低，反映 ReSplat 在 DL3DV 视频数据之外的分布差异
  （官方 `infer_colmap.py` 直接跑 counter 也只有 13.6）。

## 关键调研结论（写代码前的验证工作）

1. **hw `frames_info.txt` 的位置字段不可用**：四元数经独立 COLMAP 重建交叉验证为
   W2C（OpenCV 约定，相对旋转误差 0.54°），但 `posX/Y/Z` 与任何刚体对齐都不符
   （全局/局部 umeyama 残差 ~99%，PnP 单帧验证失败）——因此 baseline 对 hw 重新跑
   COLMAP（~96s，注册 79/79 张，重投影 0.58px）
2. 实际 jpg 为 720×960（`frames_info.txt` 声明的 1080×1440 是传感器尺寸）；
   baseline 按逐轴内参缩放自动兼容
3. 3DGS 经典代码在 sparse 只有 cameras/images（无 points3D）时崩溃——
   vendored 副本已修补（本管线的初始化本就不需要 SfM 点云）

## 选手可改进的方向（baseline 刻意保守之处）

- A：更快的位姿（免 COLMAP：利用 frames_info 的有效四元数 + 三角化恢复位置；
  或学习式位姿）、更聪明的 16 视角选择（FPS/覆盖率/信息量）
- ReSplat：更多 refinement 轮次、把训练视角作为 refinement target、分辨率/分块策略
- B：更少步数（如 3k-5k + 更高学习率）、蒸馏 ReSplat 初始化、致密化预算削减
- A+B 融合：增量式（ReSplat 出粗模型 → B 边训边补视角）
