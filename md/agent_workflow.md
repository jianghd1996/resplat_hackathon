# A/B 插件实验工作流（供后续 agent 使用）

## 目标与边界

保持赛题指定的 ReSplat 代码、预训练权重及前向过程不变，只迭代输入前处理 A 和 PLY 接续优化 B，并测试有效的 A/B 组合。所有超参数必须跨场景统一，禁止人工逐场景调参。评分权重为 PSNR 20、SSIM 20、LPIPS 40、时间 20；模块 B 必须严格快于本机经典 3DGS 的 `30000 × T_iter`，用时不超过该预算 10% 可进入时间满分区间。端到端接近半分钟是期望目标，不能当作已达成的结果或硬门槛。输出需包含标准 3DGS PLY、测试视角 PNG、质量和时间报告。

## 模块化代码契约

| 位置 | 职责与接入方式 |
|---|---|
| `baseline/module_a/preprocess.py` | 共用三种数据格式的位姿读取/重建、图像规整、训练/测试拆分和 COLMAP 工作目录；通过 `--module-a` 选择策略。 |
| `baseline/utils_A/<名称>.py` | 一个文件一种 A 策略。策略提供 `select(train_indices, count, centers=None)`，返回训练序列中的位置索引；`centers` 为同序的训练相机中心。需在 `utils_A/__init__.py` 的 `STRATEGIES` 注册。 |
| `baseline/run_resplat.py` | 固定 ReSplat 黑盒调用及世界坐标系 PLY 导出；不得为 A/B 实验修改模型本体。 |
| `baseline/utils_B/<名称>.py` | 一个文件一种 B 训练与评测渲染适配器。当前 `fastgs.py` 调用原 FastGS 训练器/渲染器；新适配器提供同样的 `train`、`render` 命令入口，并在 `utils_B/__init__.py` 的 `BACKENDS` 注册。 |
| `baseline/run_pipeline.py` | 单场景一键运行，接收 `--scene`、`--output`、`--module-a`、`--module-b`，依次执行 A → 固定 ReSplat → B → 测试渲染 → 评估。 |
| `baseline/run_train.sh` / `run_eval.sh` | 按 `baseline/scenes.json` 跑三组主场景或指定组；同样接收 A/B 参数。 |

A 的现有插件接口只改变 context 选帧。若新增锐度或色彩预处理，先设计可跨三类数据复用的输入输出，再从共用预处理器调用策略，避免把实验参数写死在某个场景分支。B 必须从同一 ReSplat 世界系 PLY 开始，输出标准 PLY、测试 PNG、`module_b.json`、`metrics.json`。若训练改变曝光、位姿或分辨率，测试渲染也必须遵守对应的相机、色彩和内参约定。

## 保存模式

服务器端统一保存为 `results/<A名>_A_<B名>_B/<group>/<scene>/`。当前 `uniform16` 与 `fastgs` 在实验目录名中均映射为 `baseline`，因此为 `results/baseline_A_baseline_B/`；例如未来 `fps` A 与基线 B 为 `results/fps_A_baseline_B/`。三场景脚本自动按模块名生成目录；使用 `run_pipeline.py` 时，调用者将 `--output` 指向相同格式的位置。

每场景的 `work/` 有 `split.json`、`module_a.json`；`resplat/` 有前馈 PLY 和 `resplat.json`；`module_b/` 有训练 PLY、测试渲染 PNG、`module_b.json`、`metrics.json`。单场景入口另写 `model.ply` 和 `report.json`，三场景汇总写在组合目录的 `summary.{json,md}`。**整个 `results/` 被 Git 忽略**，原始 PLY、PNG、JSON、日志只在服务器保存。`tmp/` 供临时脚本/日志使用，不作为正式实验目录；旧 hw `sparse` 的备份放在与 `tmp` 同级的 `backup/`。GitHub 仅保存代码、实验设计和经核对的汇总表。

## 验证、迭代和记录

1. 在 `md/plan_v0.md` 给每个候选写明失真假设、统一参数、预计时间成本、对照组和失败判据。agent 可以联网检索论文、开源实现和新实验思路，扩充待测队列；记录标题、链接、版本/日期及适用前提。外部报告的提升只能作为待验证假设，不能填入本项目结果表；新方法仍须遵守 ReSplat 黑盒和跨场景统一参数的约束。
2. 先只改 A 或只改 B，与 `uniform16 + fastgs` 对照；单项有效后再做 A/B 组合。锁定相同数据、训练/测试拆分、官方权重、评估实现和场景无关参数。预建 `sparse/0` 与现场 PyCOLMAP 建图的时间口径分开标注。
3. 先检查接口、语法与小场景冒烟，再跑 DL3DV、hw、mipnerf360 三组主场景。核对 `split.json`、PLY 可被标准渲染器读取、测试 PNG 与 GT 数量一致。实测同分辨率/视角口径的经典 3DGS `T_iter`，验证 `time_B < 30000 × T_iter`。A、ReSplat、B 必须取自同一次运行，不拼接其他实验的时间。
4. 比较留出集 PSNR、SSIM、VGG-LPIPS 和时间；LPIPS 占 40 分，优先观察其变化，同时守住有效性条件。记录失败原因，下一轮优先试机制明确、跨场景有效且容易回退的候选。
5. 在服务器运行 `python baseline/collect_results.py --experiment <A名>_A_<B名>_B --format both`，取得可复制的 Markdown 和 JSON。核对代码提交、环境、权重校验和、位姿来源、图像/测试视角数、分辨率、统一参数和计时口径；收集器优先读取新阶段 JSON，旧 `report.json` 仅作回退。经典 30k 参考可从同机同场景的 `baseline_A_baseline_B` 目录读取，JSON 的 `reference_source` 标明来源；该参考不计入实验的 A/ReSplat/B 阶段时间。其他缺失指标填 `—`，不推算或借用别的运行数值。
6. 在 `md/main_table.md` 后追加 `xx(A) + xx(B)` 小节：先说明 A/B 设计、对应代码文件和统一参数，再用同样列顺序记录三主场景；每个本机实验数值旁标记该行减去同机 baseline 的 Δ，包含质量与 A/ReSplat/B/总耗时。注明硬件、位姿缓存状态、经典 30k 参考与报告来源。历史基线、当前服务器复现和新组合分开保留。将验证过的有益组合送入下一轮交叉实验。

正式命令、三种数据格式和服务器环境见 `README_ours.md`；候选与优先级见 `md/plan_v0.md`；对外汇总以 `md/main_table.md` 为准。
