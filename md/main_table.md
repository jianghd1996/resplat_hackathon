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

## 复现结果：baseline(A) + baseline(B)

A/B 的设计和代码与上表相同；这里仅放当前服务器重新运行并有报告佐证的数值。billboard 有 64 张有效图像、56 张训练图、16 张 context、8 张测试图，评测分辨率为 768×512。它使用已重建、已去畸变的 `sparse/0`，A 的时间**不含此前 PyCOLMAP 建图**，所以 243.4s 是缓存位姿口径。其本地报告位于 `tmp/billboard_verify.z28Bi5/billboard_report.json`；原始报告不上传仓库。

| **group** | **scene** | **PSNR** | **SSIM** | **LPIPS** | **A(s)** | **ReSplat(s)** | **B(s)** | **总耗时** | **经典30k参考** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| dl3dv | 001dccbc… | — | — | — | — | — | — | — | — |
| hw | billboard | 22.661 | 0.7985 | 0.1606 | 10.82 | 35.86 | 196.68 | **243.4s** | — |
| mipnerf360 | counter | — | — | — | — | — | — | — | — |

同批 hw 重建中的 `trashbin` 另有 10 个测试视角的评估：PSNR 27.799、SSIM 0.8911、LPIPS 0.1602；目前未收到其完整分段计时，因此暂不并入上面的三主场景表。

后续每个 A/B 组合依次追加一个 `xx(A) + xx(B)` 小节：先说明 A/B 的算法、对应 `utils_A`/`utils_B` 文件和统一参数，再给同样的三场景表格与计时口径。服务器结果按 `results/xx_A_xx_B/<group>/<scene>/` 管理；`results/` 整体由 Git 忽略。更新表格前用 `baseline/collect_results.py` 从该目录生成可复制的 Markdown/JSON 汇总。
