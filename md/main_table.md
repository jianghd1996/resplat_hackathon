# A/B 插件实验总表

固定 ReSplat 为赛题提供的代码与预训练权重。模块 A 从训练视角选取前馈输入、建立/读取位姿并规整图像；模块 B 从 ReSplat 导出的标准 PLY 接续训练并输出可渲染 PLY。评测用相同留出视角计算 PSNR、SSIM、VGG-LPIPS；总耗时采用 A + ReSplat + B，渲染/评测耗时另计。不同硬件和“是否复用已建位姿”的时间不能混作同一次实验。

## baseline(A) + baseline(B)：仓库既有基线记录

A 设计：`baseline/module_a/preprocess.py` 读取 COLMAP / DL3DV `transforms.json`，hw 使用 PyCOLMAP 建图并去畸变；在训练视角上按索引均匀选取最多 16 张。策略代码为 `baseline/utils_A/uniform16.py`，参数 `--module-a uniform16`。B 设计：`baseline/module_b/train_from_ply_fastgs.py` 从 ReSplat PLY 开始 FastGS 接续 10000 步，适配器为 `baseline/utils_B/fastgs.py`，参数 `--module-b fastgs`。

下表来自仓库根目录 README 的**历史基线记录**，原始逐阶段 JSON 未提交；它不能代表当前服务器新位姿流程的计时。经典 30k 参考列为耗时 / PSNR。

| **group** | **scene** | **PSNR** | **SSIM** | **LPIPS** | **总耗时** | **经典30k参考** |
|---|---|---:|---:|---:|---:|---|
| dl3dv | 001dccbc… | 32.57 | 0.950 | 0.153 | **89s** | 547s / 29.91 |
| hw | billboard | 22.68 | 0.799 | 0.159 | **143s** | 915s / 16.75 |
| mipnerf360 | counter | 27.77 | 0.888 | 0.207 | **116s** | 531s / 27.08 |

## 本机复现：baseline(A) + baseline(B)，2026-10-30

A 使用 `baseline/utils_A/uniform16.py` 的训练视角均匀 16 帧，B 使用
`baseline/utils_B/fastgs.py` 从相同 ReSplat PLY 接续 10000 步。代码为
`fc96611`（基于 `526a818`），官方权重 SHA256 为
`a72dc6d0d56f7ccc50d145c17c83d866d4aaa2c13adb1a22f7a84481001e1a4c`。
本机 GPU 为 RTX 3090 24 GB；ReSplat 环境 PyTorch 2.7.0+cu128，FastGS 环境
PyTorch 2.4.1+cu121。三场景分别在 42、8、30 个测试视角上评测；位姿来源依次为
`transforms.json`、预先用 PyCOLMAP 3.10 重建并去畸变的 `colmap.bin`、已有的
`colmap.bin`。HW 的 A 时间只计复用位姿后的预处理，不含首次建图。三场景的
图像/训练/测试数量与分辨率依次为 DL3DV 330/288/42、512×960，HW
64/56/8、768×512，`counter` 240/210/30、576×960。三个阶段的
计时与指标均来自各场景同一次运行，原始 JSON、PLY 和 PNG 保存在本机
`results/baseline_A_baseline_B/`，其中链接到运行时的 `baseline/work/` 与
`baseline/results/`；收集自 `baseline/collect_results.py`。本机经典 30k 参考尚未实测。

| **group** | **scene** | **PSNR** | **SSIM** | **LPIPS** | **A(s)** | **ReSplat(s)** | **B(s)** | **总耗时** | **经典30k参考** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| dl3dv | 001dccbc… | 32.843 | 0.9505 | 0.1518 | 47.2585 | 31.02 | 72.04 | **150.318s** | — |
| hw | billboard | 22.571 | 0.7983 | 0.1597 | 8.5261 | 29.74 | 73.84 | **112.106s** | — |
| mipnerf360 | counter | 27.754 | 0.8881 | 0.2071 | 84.5212 | 34.41 | 79.46 | **198.391s** | — |

## fps(A) + baseline(B)：相机中心最远点选帧

A 代码为 `baseline/utils_A/fps.py`（提交 `6fbd2b6`）：仅在训练相机中心上
确定性选 16 帧，起点为距均值最远者，之后逐次选择离当前已选集合最远者；
并列时按训练序列顺序选择。B 仍为原 FastGS 10000 步。所有场景共用该参数，
官方 ReSplat、权重、分辨率、训练/测试划分与上面的本机复现相同。GPU 为 RTX 3090
24 GB，HW 使用预建 PyCOLMAP 位姿；原始报告、PLY、PNG 和 `run_metadata.json`
在本机 `results/fps_A_baseline_B/`。以下由 `baseline/collect_results.py` 从各阶段
JSON 收集；经典 30k 本机参考仍待测。

| **group** | **scene** | **PSNR** | **SSIM** | **LPIPS** | **A(s)** | **ReSplat(s)** | **B(s)** | **总耗时** | **经典30k参考** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| dl3dv | 001dccbc… | 32.672 | 0.9501 | 0.1516 | 47.0227 | 30.66 | 74.92 | **152.603s** | — |
| hw | billboard | 22.649 | 0.7987 | 0.1590 | 8.5656 | 25.65 | 77.10 | **111.316s** | — |
| mipnerf360 | counter | 27.666 | 0.8843 | 0.2156 | 82.0922 | 33.89 | 79.74 | **195.722s** | — |

三场景的训练/测试拆分均与基线逐项相同；标准 PLY 可读取，测试 PNG 数量分别为
42、8、30。相对本机基线，FPS 的 LPIPS 变化为 -0.0002、-0.0007、+0.0085，
并未呈现稳定收益。固定使用基线的 `counter` ReSplat PLY、相同采样种子与
20,000 个高斯中心时，FPS 所选相机的至少一个视锥覆盖率从 99.1% 降至 96.6%，
平均/最大角度间隙从 8.97°/21.02° 增至 9.57°/23.58°。这是视锥诊断，
不计遮挡；它支持继续研究方向与可见性覆盖，而非直接将相机中心 FPS 组合到 B。

## baseline(A) + short_budget7000(B)：单 B 短预算

A 仍为 `uniform16`；B 代码为 `baseline/utils_B/short_budget.py` 与
`baseline/module_b/train_from_ply_fastgs.py`（提交 `e37864f`），从相同官方
ReSplat 世界系 PLY 接续 7000 步，其余 FastGS 参数沿用基线，只保护最后一步
不在 opacity 重置后立即保存。三场景共用 7000 步，GPU 为 RTX 3090 24 GB，
HW 复用 PyCOLMAP 位姿。训练/测试拆分、16 个 context、权重及评测与本机基线
一致；标准 PLY 可读取，测试 PNG 分别为 42、8、30 张。各阶段原始报告在本机
`results/baseline_A_short_budget7000_B/`，下表由 `baseline/collect_results.py`
收集；经典 30k 本机参考仍待测。

| **group** | **scene** | **PSNR** | **SSIM** | **LPIPS** | **A(s)** | **ReSplat(s)** | **B(s)** | **总耗时** | **经典30k参考** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| dl3dv | 001dccbc… | 32.065 | 0.9405 | 0.1612 | 46.8681 | 30.61 | 56.58 | **134.058s** | — |
| hw | billboard | 22.636 | 0.7937 | 0.1681 | 8.4741 | 25.89 | 55.29 | **89.654s** | — |
| mipnerf360 | counter | 27.120 | 0.8725 | 0.2234 | 82.9770 | 34.57 | 67.48 | **185.027s** | — |

相对 10000 步本机基线，B 用时降低 21.5%、25.1%、15.1%，但 DL3DV 和
`counter` 的 PSNR 分别下降 0.778、0.634 dB，`counter` 的 LPIPS 上升 0.0163。
因此 7000 步未达到预设跨场景质量门槛，暂不与 FPS A 组合。

### billboard 短预算门槛曲线

以下是同一 `billboard` 场景、相同拆分与权重的单场景探针；其他两组未在
100–5000 步逐项测试，不能把这些行当成三场景结论。时间均为该行同一次运行的
B 阶段用时。`3000 原始` 的默认 opacity 重置恰好发生在最终保存前，导致
PSNR 5.598；修正版仅跳过末步重置，原始失败产物另存于
`results/baseline_A_short_budget3000_raw_B/`。

| B 步数 | PSNR | SSIM | LPIPS | B(s) |
|---:|---:|---:|---:|---:|
| 100 | 21.339 | 0.7127 | 0.2481 | 4.93 |
| 300 | 21.970 | 0.7466 | 0.2224 | 6.61 |
| 1000 | 22.290 | 0.7647 | 0.2055 | 11.55 |
| 3000 原始 | 5.598 | 0.1446 | 0.6329 | 29.90 |
| 3000 修正 | 22.597 | 0.7818 | 0.1841 | 26.57 |
| 5000 | 22.491 | 0.7876 | 0.1764 | 39.95 |
| 7000 | 22.636 | 0.7937 | 0.1681 | 55.29 |
| 10000 基线 | 22.571 | 0.7983 | 0.1597 | 73.84 |

后续每个 A/B 组合依次追加一个 `xx(A) + xx(B)` 小节：先说明 A/B 的算法、对应 `utils_A`/`utils_B` 文件和统一参数，再给同样的三场景表格与计时口径。服务器结果按 `results/xx_A_xx_B/<group>/<scene>/` 管理；`results/` 整体由 Git 忽略。更新表格前用 `baseline/collect_results.py` 从该目录生成可复制的 Markdown/JSON 汇总。
