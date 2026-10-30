# A/B 插件实验计划 v0

目标：保持官方 ReSplat 代码、权重及前向过程不变，只迭代模块 A（输入视角与预处理）和模块 B（从 ReSplat PLY 接续优化）。所有策略跨场景使用同一套参数，按 `md/main_table.md` 记录质量和时间。赛题要求 B 耗时严格小于本机经典 3DGS 的 `30000 × T_iter`；同时优先争取接近 10% 时间预算的满分区间。PSNR/SSIM/LPIPS 权重分别为 20/20/40，时间为 20。

当前入口：`baseline/run_pipeline.py --scene ... --output ... --module-a uniform16 --module-b fastgs`。A 策略放在 `baseline/utils_A/<名称>.py`，暴露 `select(train_indices, count)`；B 适配器放在 `baseline/utils_B/<名称>.py`，接受 `train`/`render` 动作。`uniform16` 与 `fastgs` 对应现有 baseline，输出实验目录 `results/baseline_A_baseline_B/`。新增策略时需要登记 CLI 选择、统一的输出目录命名和评测渲染器，且在相同输入、相同测试拆分下做消融。

| 优先级 | 待测模块 / 文件建议 | 实验与判据 |
|---|---|---|
| P0 | 桥接与计时回归（现有 `run_resplat.py`、`summarize.py`） | 核对 PLY 坐标/四元数/SH/opacity 约定，实测同分辨率、同视角数的经典 3DGS `T_iter`；按 A、ReSplat、B 分段记录时间与测试集指标。 |
| P1 | A：`utils_A/fps.py` | 在训练候选相机中心上做确定性最远点选取；与 `uniform16` 对比覆盖度、ReSplat 输出和最终测试指标。候选池与 B 的训练监督池保持不变。 |
| P1 | B：`utils_B/short_budget.py` | 先测 0/100/300/1000/3000 等统一迭代预算的质量–时间曲线；再依次试短程调度、禁用不合适的 opacity 硬重置、预算制 top-k 生长和 coarse-to-fine 分辨率。每项单独消融，注意分辨率切换时内参和梯度阈值同步缩放。 |
| P2 | A：`utils_A/coverage_local.py`、`utils_A/quality_filter.py` | 测边界覆盖加邻近共视、统一阈值模糊帧过滤；不能按场景手工选帧或调阈值。 |
| P2 | B：`utils_B/regularized.py` | 在已验证的短程训练器上试重要性剪枝、视图锥尺度约束、曝光补偿，重点观察留出集 LPIPS；曝光若用于训练，评测渲染也必须保持一致。 |
| P3 | B：`utils_B/pose_refine.py` 等 | 仅当前两档有收益且渲染器支持梯度时试位姿精化、长轴分裂、稀疏优化器及感知损失；确保导出相机/PLY 和评测一致。 |

测试顺序：先锁定同一批三组主场景和统一训练/测试划分，再逐个替换 A 或 B，之后测试 A/B 组合。每次记录 Git 提交、环境、权重校验和、参数、位姿来源、图像数/分辨率、阶段计时、PSNR/SSIM/LPIPS、标准 PLY 和渲染 PNG。大文件与原始 JSON 留在服务器端 `results/`；将可核实的汇总写入 `md/main_table.md`。上述新策略只是待测假设，尚未实现或验证。

## 本轮执行记录与判据

- [x] P0：将本机三场景基线复现的 A、ReSplat、B 阶段 JSON 和测试指标归档到
  `results/baseline_A_baseline_B/`；已核对同一次运行的分段计时、标准 PLY、
  留出视角渲染与根目录 README 的指标。HW 使用预建 PyCOLMAP 位姿。
- [ ] P0：在相同分辨率和训练视角口径下实测本机经典 3DGS 的 `T_iter`，再确认
  `time_B < 30000 × T_iter`。历史 4090 参考不能代替本机测量。
- [x] P1 / FPS A：`fps.py` 仅从训练相机中心确定性选择 16 张，起点为距训练相机
  均值最远的相机，之后重复选离已选集合最近距离最大的相机；并列时选训练顺序
  靠前者。假设空间覆盖改善可降低留出集 LPIPS。相对 `uniform16 + fastgs`
  保持相同 10000 步、权重、分辨率和拆分。预处理成本为一次相机中心计算和
  16 次距离扫描；若三场景中至少两组 LPIPS 无改善，或任一组 PSNR 下降超过
  0.5 dB，则不将 FPS 进入组合实验。
  三场景已完成：LPIPS 相对基线依次变化 -0.0002、-0.0007、+0.0085；
  `counter` 的固定点集视锥覆盖率从 99.1% 降到 96.6%，角度间隙增大。
  虽然前两组有微小改善，但总体没有稳定收益，本轮不进入 A/B 组合。
- [ ] P1 / 短预算 B：先以相同上游 PLY 和统一训练配置比较 100、300、1000、
  3000、10000 步；0 步需确认标准 PLY 直接渲染的相机和色彩口径后单列。
  主要比较 LPIPS 与 B 耗时；先找三场景中均保持 PSNR 下降小于 0.5 dB、
  LPIPS 增幅小于 0.01 的最短预算，再单独消融学习率调度和 opacity 重置。
  `billboard` 门槛试验显示 100/300/1000 步 LPIPS 均明显变差；原始 3000 步
  恰逢默认 opacity 重置并立即保存，PSNR 跌至 5.598。跳过末步重置后恢复到
  22.597 / 0.7818 / 0.1841，仍未满足 LPIPS 门槛。下一轮扩大到 5000、7000
  步寻找质量与时间的转折点；原始失败产物单独保存，不覆盖。7000 步的
  `billboard` 通过门槛，但 DL3DV 和 `counter` 的 PSNR 分别下降 0.778、
  0.634 dB，不能跨场景采用。
- [ ] P1 / `short_schedule` B：固定 7000 步和其余 FastGS 参数，只将位置学习率
  衰减终点从 30000 改为 7000；相对 `short_budget7000` 控制组隔离此参数。
  假设短程内更快到达低学习率可补回留出集 PSNR/LPIPS。先用 DL3DV 作为
  失败场景门槛；若 PSNR 与 LPIPS 同时改善，再跑另外两组。B 时间预计与
  7000 步相近；若仍不满足三场景 PSNR 下降 <0.5 dB、LPIPS 增幅 <0.01，
  不进入组合实验。

## 待实验模块扩充（2026-10-30 检索，均未在本项目验证）

- [ ] P1 / `utils_A/pose_coverage.py`：FPS 在 `counter` 上降低固定点集的视锥
  覆盖率，说明相机中心距离遗漏方向与可见性。以训练相机中心和前向方向构造
  跨场景归一化的姿态距离，贪心选 16 张；统一权重，不使用测试相机或测试图像。
  预计只增加相机数量乘以 16 的几何计算。对照 `uniform16` 与 `fps`，失败判据
  沿用三场景 LPIPS/PSNR，并核对同一点集视锥覆盖率。此候选是受
  [Chen 等，CVPR 2026，Coverage Optimization for Camera View Selection](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Coverage_Optimization_for_Camera_View_Selection_CVPR_2026_paper.html)
  启发的低成本近似；论文的方法依赖场景覆盖信息，我们的姿态代理并非其实现。
- [ ] P2 / `utils_A/frequency_quality.py`：用统一的归一化频域得分衡量候选帧
  细节，再与覆盖度共同选帧；需排查噪声、运动模糊和高频纹理导致的误选。
  每帧 FFT 增加 A 时间，先在三组固定拆分上离线计算，再决定是否进入完整流水线。
  来源：[Li 等，2024，Frequency-based View Selection in Gaussian Splatting Reconstruction](https://arxiv.org/abs/2409.16470)。
- [ ] P2 / 短程 B 调度：在已找到的有效迭代预算上，分别测试缩短位置学习率
  日程与跳过非末步 opacity 重置；一次只变一个参数，记录高斯数量、LPIPS 和
  B 时间。默认末步重置导致的 3000 步崩溃是本项目已观察到的实现问题。
  [Hahlbohm 等，CVPR 2026，Faster-GS](https://openaccess.thecvf.com/content/CVPR2026/html/Hahlbohm_Faster-GS_Analyzing_and_Improving_Gaussian_Splatting_Optimization_CVPR_2026_paper.html)
  可作为后续优化策略的外部参考，论文性能不作为本项目预期值。
- [ ] P3 / 信息量选帧：如轻量覆盖代理仍无效，再评估
  [FisherRF，ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02130.pdf)
  的 Fisher 信息思想；需要额外模型/导数，先量化 A 时间成本及与固定 ReSplat
  黑盒的兼容性。
