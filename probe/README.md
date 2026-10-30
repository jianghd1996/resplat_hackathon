# v0 机制诊断与 v1 定向探针

日期：2026-10-30。分析代码以 `8aab3e0` 的训练实现和已有实验为对象。
入口文档：[机制复盘](../md/rethink_v0.md)、[维度草案](../md/plan_v1_draft.md)、[下一轮计划](../md/plan_v1.md)。

本目录只读数据、已有实验与模型状态，不修改 A/B 训练实现，不启动完整训练。
全集指本机 `datasets/{DL3DV,hw,mipnerf360}` 的全部 13 个场景，不指公开数据集的全部场景。
历史质量结果仅覆盖三个主场景；其余场景只汇报探针统计。
原始结果保存在被 Git 忽略的 `results/rethink_v0/`，文档保存经核对的摘要。

在仓库根目录运行；本机已有环境和 VGG 权重缓存：

```bash
conda run -n resplat python probe/dataset_census.py
conda run -n resplat python probe/covisibility_swaps.py
conda run -n resplat python probe/context_geometry.py
conda run -n fastgs python probe/result_forensics.py
conda run -n fastgs python probe/training_mechanics.py
conda run -n resplat python probe/plot_geometry.py --out md/assets/rethink_v0_coverage.svg
```

第二条依赖第一条输出，最后的制图依赖前两条；其余可独立运行。`context_geometry.py` 和
`result_forensics.py` 需要本机保留的 v0 实验 PLY、划分、PNG 和 JSON。
`training_mechanics.py` 使用实际 FastGS 类和 LPIPS 初始化，CPU 上验证状态变化；
首次没有感知网络缓存时可能触发模型下载。它恢复随机状态仅用于本进程中的对照。

| 脚本 / 输出同名 JSON | 回答的问题 | 边界 |
|---|---|---|
| `dataset_census.py` | 全部 13 场景的位姿路径、距离、朝向、FOV、图像代理分布及三种 A 的轨迹覆盖 | 不做 SfM，不读测试图像计算质量代理；DL3DV 无轨迹，不当作覆盖为零 |
| `covisibility_swaps.py` | 保留均匀骨架、至多两次受约束交换，能否提高训练轨迹支持 | 7 场景有轨迹，6 场景回退；离线代理提升不是 PSNR/LPIPS 提升 |
| `context_geometry.py` | 三主场景的高斯中心视锥覆盖、多重覆盖、冗余与相机距离 | 固定 20k 点、seed=0；不考虑遮挡；混合点云只是参考点集敏感性检查 |
| `result_forensics.py` | 已有逐视图退化是否集中、对照输入是否相同、初始/最终尺度和透明度、残差空间分布 | 不产生新训练结果；PNG 残差受保存量化影响 |
| `training_mechanics.py` | LPIPS 初始化是否改 RNG、透明度限幅是否清空 Adam 状态、分裂横向尺度变化 | 状态与解析微实验，不能单独量化对最终质量的贡献 |
| `plot_geometry.py` | 将全集相机距离和轨迹覆盖画成可比较的静态SVG | 读取探针JSON，只绘制代理指标，不代表新训练成绩 |

几何距离均除以各场景训练相机包围盒对角线，不是米。轨迹覆盖分母为训练图像观测到的
point3D ID 并集；每视图支持率的 p10 反映覆盖较差的训练视角。已有 SfM 本身未重新按
训练集构建，因此这里是“仅读取训练观测”，不是宣称源 SfM 从未使用留出视角。

`dataset_census.py` 的 Laplacian 在原图直接缩放至 256² 后计算，与 `quality_filter`
先做管线规整再缩放的口径不同，不能用它替代历史实际替换数。

外部代码检索的原始副本及版本记录保存在本机 `results/rethink_v0/references/manifest.json`；
长期可追溯引用使用 `plan_v1.md` 中固定提交的 GitHub 链接。
