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

测试顺序：先锁定同一批三组主场景和统一训练/测试划分，再逐个替换 A 或 B，之后测试 A/B 组合。每次记录 Git 提交、环境、权重校验和、参数、位姿来源、图像数/分辨率、阶段计时、PSNR/SSIM/LPIPS、标准 PLY 和渲染 PNG。大文件与原始 JSON 留在服务器端 `results/`；将可核实的汇总写入 `md/main_table.md`。上表是初始模块队列；已完成和待测状态以下方清单为准。

## 本轮执行记录与判据

- [x] P0：将本机三场景基线复现的 A、ReSplat、B 阶段 JSON 和测试指标归档到
  `results/baseline_A_baseline_B/`；已核对同一次运行的分段计时、标准 PLY、
  留出视角渲染与根目录 README 的指标。HW 使用预建 PyCOLMAP 位姿。
- [x] P0：在相同分辨率和训练视角口径下完成本机经典 3DGS 30k 训练。
  DL3DV、HW、`counter` 的训练纯耗时分别为 738.76、1156.62、697.07 秒，
  `T_iter` 分别为 0.024625、0.038554、0.023236 秒。原 10000 步 B 的
  `time_B / (30000 × T_iter)` 为 9.75%、6.38%、11.40%，均低于 30k
  总预算，但 `counter` 超过 10% 满分时间档。原始参考报告见本机各场景
  `results/baseline_A_baseline_B/<group>/<scene>/reference_3dgs/module_b.json`。
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
- [x] P1 / `short_schedule` B：固定 7000 步和其余 FastGS 参数，只将位置学习率
  衰减终点从 30000 改为 7000；相对 `short_budget7000` 控制组隔离此参数。
  假设短程内更快到达低学习率可补回留出集 PSNR/LPIPS。先用 DL3DV 作为
  失败场景门槛；若 PSNR 与 LPIPS 同时改善，再跑另外两组。B 时间预计与
  7000 步相近；若仍不满足三场景 PSNR 下降 <0.5 dB、LPIPS 增幅 <0.01，
  不进入组合实验。三场景已完成：相对原 10000 步基线，PSNR 为
  +0.156、+0.042、-0.222 dB，LPIPS 为 -0.0102、-0.0003、+0.0014；
  B 时间占本机 30k 纯训练预算 7.92%、4.84%、9.00%。通过质量与时间门槛，
  保留为当前 B 候选，后续继续测试 5000 步等更短预算。
- [x] P1 / `short_schedule5000` B：保持相同训练/测试视角、末步重置保护与
  学习率终点等于迭代数，先测 DL3DV，再测 HW 和 `counter`。门槛沿用上项，
  对比 7000 步 B 的质量与耗时；三场景完成后才更新主表中的正式结论。
  实测三场景均通过质量门槛，B 用时 50.53、39.08、52.15 秒，占本机
  30k 纯训练预算 6.84%、3.38%、7.48%。DL3DV 比 7000 步更好，另两组
  质量略降；保留为更短预算候选。
- [x] P1 / `short_schedule3000` B：仅先做 DL3DV 门槛探针。3000 步正好
  碰到原始 opacity 重置节点，必须确认末步保护生效并读取实际 PLY/渲染；
  如果质量不能接近基线，不扩展到另两组。DL3DV 探针为
  33.551/0.9592/0.1277（PSNR/SSIM/LPIPS），因此扩展三场景；
  `counter` 为 26.715/0.8724/0.2235，相对原基线 PSNR -1.039 dB、
  LPIPS +0.0164，未通过跨场景门槛。所有场景 `split.json` 与基线相同，
  PLY 和测试 PNG 已核对。
- [x] P1 / `short_schedule1000` B：DL3DV 单场景探针为
  32.218/0.9488/0.1415，B 19.64 秒。PSNR 相对原基线 -0.625 dB，
  因此按门槛停止，不扩展到 HW 和 `counter`。
- [ ] P1 / 当前预算选择：5000 和 7000 步均跨场景通过质量门槛；
  5000 步更快且 DL3DV 更好，7000 步在 HW 与 `counter` 质量更好。
  两者 B 时间均处于同机经典 30k 的 10% 内。后续在官方评分细节明确时
  统一选一个预算，不能逐场景挑选。
- [x] P2 / `quality_filter` A：在统一 resize 后把训练候选图缩至
  256×256 灰度图，计算 Laplacian 方差，以同场景训练候选的中位数归一化。
  对 `uniform16` 原选帧中低于 0.8 倍中位数的帧，替换为序列位置最近的
  合格训练帧；阈值和替换规则跨场景固定，不使用测试图像。与原基线 B
  10000 步对照，三场景至少两组 LPIPS 改善且 PSNR 不下降超过 0.5 dB
  才考虑组合。离线审计显示该阈值将改变 DL3DV 5/16、HW 0/16、
  `counter` 3/16 张 context；该统计不能证明它们实际模糊，需完整评估。
  三场景已测：LPIPS Δ 为 +0.0009、+0.0001、-0.0011，仅 counter 改善，
  未通过至少两组改善条件；A 读图成本增加，不进入组合。
- [x] P2 / `coverage_local` A：从训练相机中心及前向方向选 8 个分散的
  边界锚帧，再为每个锚帧选 1 张姿态最近的邻帧，凑齐统一 16 张
  context。位置按训练相机范围归一化，姿态距离固定为 0.6×位置距离
  +0.4×方向差；邻近姿态只是共视代理，未使用 3D 点重叠或遮挡。
  只改 A，B 保持原 10000 步。对照 `uniform16` 与 FPS，
  按同样的三场景 LPIPS/PSNR 门槛及固定点集视锥覆盖诊断判断是否保留。
  三场景 LPIPS Δ 为 +0.0095、+0.0063、+0.0037，DL3DV PSNR Δ
  为 -0.697 dB，未通过。`counter` 固定点集覆盖率为 94.85%，均匀基线
  为 99.14%；锚帧数从 8 扫到 16 仍未恢复基线覆盖率，故不再跑该系列的
  完整组合。离线扫描在本机 `results/coverage_anchor_sweep/counter.json`。
- [x] P2 / `regularized7000` B：以通过门槛的 `short_schedule7000` 为
  控制组，只在训练结束后增加 FastGS 已有的十个训练相机多视图评分和
  `final_prune_fastgs(min_opacity=0.01)`；标准 PLY 在剪枝后保存，渲染
  不改变。先用 `counter` 检验剪枝数量、PLY/PNG 和 LPIPS；若 LPIPS
  不优于控制组或 PSNR 恶化超过 0.5 dB，即停止，不推广到其余场景。
  剪枝耗时计入 B。`counter` 实测从 299,985 个剪至 273,714 个高斯，
  PSNR/LPIPS 为 27.328/0.2086，低于控制组的 27.532/0.2085，B
  64.32 秒也比控制组 62.74 秒更慢，故按门槛停止。曝光和视图锥尺度是
  独立候选，不与此探针合并。
- [ ] P2 / `no_reset7000` B：以 `short_schedule7000` 为控制组，只跳过
  训练中途的全部 opacity 重置，位置学习率日程及其他参数保持一致。
  先测 `counter`，若 LPIPS 不改善或 PSNR 降幅超过 0.5 dB，则停止；
  若通过，再跑另外两组并检查同一 PLY、训练/测试拆分与渲染数量。

## 待实验模块扩充（2026-10-30 检索，均未在本项目验证）

- [ ] P1 / `utils_A/pose_coverage.py`：FPS 在 `counter` 上降低固定点集的视锥
  覆盖率，说明相机中心距离遗漏方向与可见性。以训练相机中心和前向方向构造
  跨场景归一化的姿态距离，贪心选 16 张；统一权重，不使用测试相机或测试图像。
  预计只增加相机数量乘以 16 的几何计算。对照 `uniform16` 与 `fps`，失败判据
  沿用三场景 LPIPS/PSNR，并核对同一点集视锥覆盖率。此候选是受
  [Chen 等，CVPR 2026，Coverage Optimization for Camera View Selection](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Coverage_Optimization_for_Camera_View_Selection_CVPR_2026_paper.html)
  启发的低成本近似；论文的方法依赖场景覆盖信息，我们的姿态代理并非其实现。
- [x] P2 / 频域质量离线审计：用统一的归一化频域得分衡量候选帧
  细节，再与覆盖度共同选帧；需排查噪声、运动模糊和高频纹理导致的误选。
  已在三组固定拆分上计算每张训练图 256×256 灰度 FFT 的高频能量比例，
  以同场景中位数的 0.8 倍审计原均匀 16 帧：DL3DV 5 张、HW 0 张、
  `counter` 3 张低于阈值，其中 DL3DV 5 张与 Laplacian 筛选完全相同，
  `counter` 2/3 张相同。原始统计见本机
  `results/frequency_quality_audit/audit.json`。频率指标仍可能把低纹理误当
  模糊；先看 `quality_filter` 的完整评估，再决定是否实现独立流水线。
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
