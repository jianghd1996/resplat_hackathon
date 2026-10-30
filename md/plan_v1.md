# plan_v1：用覆盖互补与收敛连续性驱动下一轮

2026-10-30。分析对象：本机v0结果及本页勾选的v1正式实验。本文同时保存实验计划、停止判据与已完成结论。
依据：[机制复盘](rethink_v0.md)、[维度草案与指标修订](plan_v1_draft.md)、[可复跑探针](../probe/README.md)。
结果记录继续遵守 [agent_workflow](agent_workflow.md)，v1完整训练结果记录在 [main_table_v1](main_table_v1.md)。

## 1. 目标、控制组与判据

固定ReSplat代码、权重和16视图输入；只迭代A/B。场景间统一参数，不按场景挑预算或手工选帧。
允许同一函数根据可测训练输入自适应：`context = A(input; θ)`，所有场景共用同一组θ和决策规则。
“统一参数”不要求所有输入采取相同动作；禁止的是按数据集名称/场景ID切换手调参数，或依据留出质量选择动作。
主质量控制为 `uniform16 + no_reset7000`；速度参照为 `uniform16 + no_reset5000`；原10000步baseline永久保留。
所有新结果报告相对原baseline和当前配对控制的两组Δ，避免一个候选“比旧基线好”却比现有最佳退步。

| 层次 | 执行规则 |
|---|---|
| 有效性 | 标准PLY、同一留出划分/渲染分辨率/指标实现；不改变测试相机去迎合输出；新B使用同一输入PLY哈希 |
| 硬时间条件 | 依仓库工作流，B严格低于同机同场景 `30000×T_iter`；另报端到端与经典管线相同口径时间，不混用纯训练和总时间 |
| 时间档 | B≤10%为仓库记录的满分档；跨10%不等于违反100%硬预算。候选有质量提升时仍记录Pareto取舍，不臆造档外分数 |
| 质量路线 | 优先LPIPS，兼看PSNR/SSIM。暂定相对当前控制每场景PSNR不低于−.2 dB、SSIM不低于−.002、LPIPS不高于+.001；至少两主场景LPIPS方向一致且超过已知运行漂移，才提升为通用质量候选 |
| 速度路线 | 同一质量护栏下，B或端到端至少缩短5%才列为优先速度候选；收益小于已知运行漂移时只保留观察，不称为提速 |
| 小效应处理 | 正式表固定FastGS seed=0，每个配置/场景只运行一次，seed不作为消融项。已观察到同seed CUDA路径并非位级确定，因此方向不一致或接近历史运行漂移的小效应不晋级；诊断运行不混入正式表 |
| 汇总 | 逐场景指标与Δ为主，三场景等权均值为辅；LPIPS相对改善先逐场景计算再平均；另报逐视图恶化数/中位数/最差视图 |

以上质量/5%门槛是**本轮工程筛选阈值，不是官方规则**。可保留超出护栏但有明确Pareto价值的探索结果，需单列权衡，不冒充通用提升。
仓库仅记录20/20/40/20权重，没有完整官方归一化和时间评分曲线；不能线性加权原始dB、SSIM、LPIPS和秒数生成“官方综合分”。
counter当前距10%边界仅1.567秒，优先关注稳定余量；若质量改善且仍在同档，不能因更小B没有额外时间分便忽略端到端价值。

## 2. 全集探针如何改变优先级

- 本地13场景全部纳入几何/图像统计；已有完整训练仅3场景。local在13/13场景平均相机距离变差、7/7有轨迹场景覆盖下降，因此停止该成对选帧家族的盲扫。
- 训练轨迹约束的两交换探针在7/7场景改善并集和支持p10，counter覆盖37.34%→41.22%、bicycle32.82%→37.42%。这是A的优先验证方向，但优化目标本身改善不等于质量泛化。
- no_reset仍在每次密集化清空opacity动量；7000步末次拓扑变化后没有优化更新。这是两个低成本、可分别反证的B方向。
- 三个输入PLY最短轴全部在1e-6下限，long-axis缩短无位移补偿的切向轴。分裂方案先做算子守恒/渲染探针，不直接进入全量训练。
- 感知网络初始化改变剪枝RNG，固定分辨率切换改变相机抽样栈。先消除混杂，才讨论loss或多尺度本身。

### 2.1 新方向：condition-guided A

用户提出用数据特点引导针对性预处理。这属于模块A：在固定ReSplat前，读取训练相机/训练图像的可用统计，
决定如何分配同样16个context。若条件控制的是B步数、loss或增密，则应另列为B，不能混成A单项收益。
优先做“有证据才修改均匀选帧”的可解释规则，暂不训练复杂路由网络：13场景输入统计、仅3场景历史质量，不足以支撑稳定的质量收益预测器。

以下描述性审计由 [condition_audit.py](../probe/condition_audit.py) 读取全集已有探针生成，没有拟合阈值或新增质量结果。
`角度缺口/FOV`为未选训练视图到最近context前向方向的夹角p95，除以水平FOV；仅作无量纲姿态代理，
忽略平移、roll、垂直FOV与遮挡，**不能解释为实际视野缺口比例**。

| 场景 | 水平FOV° | 均匀角度缺口/FOV | 均匀轨迹覆盖（%） | 两交换覆盖增益（百分点） |
|---|---:|---:|---:|---:|
| billboard | 55.0 | .089 | 83.33 | +.64 |
| trashbin | 54.9 | .254 | 65.75 | +.69 |
| bicycle | 56.0 | .679 | 32.82 | +4.60 |
| bonsai | 51.6 | .457 | 37.01 | +2.64 |
| counter | 52.0 | .339 | 37.34 | +3.88 |

同样约55°的FOV并没有同样的覆盖需求。当前Mip三个场景的交换覆盖增益更大，但这是代理目标的观察关联，
不能证明FOV/覆盖缺口导致了最终质量收益；trashbin覆盖比billboard低，交换增益却相近，也说明“缺口大”不等于“存在可改善的候选”。
六个DL3DV场景FOV约96°，角度缺口/FOV仍从.074到.199变化；缺轨迹应标为未知，不能推断为覆盖好或覆盖差。

| 可测condition | 机制用途 | 建议动作及限制 |
|---|---|---|
| 每相机内参/FOV、最近中心距离和角度缺口分布 | 同样角度间隔对不同FOV的意义不同；检测姿态空档 | 作为几何保护条件与候选诊断，不单独按“FOV<某值”路由；当前统计使用输入提供的共同内参，后续多内参场景须逐相机计算 |
| 当前均匀选择的轨迹覆盖、每视图支持p10 | 判断支持不足是否集中在少数训练视角 | 优先补弱支持视角，保持端点/分区；不能把稀疏轨迹覆盖当真实表面覆盖 |
| 候选相对uniform的实际新增支持与冗余 | 判断缺口是否可通过有限交换修补 | 候选增益不足或违反护栏时保留uniform；允许不同场景最终交换0/1/2张，但最大配额与规则共享 |
| 轨迹可用性、长度、每图观测数、图像网格占用 | 判断覆盖代理的可靠性，避免特征集中在局部纹理 | 轨迹缺失/不可靠则回退uniform；可靠性统计尚需补探针，不假定“有images.bin”就可靠 |
| 计算成本及代理增益稳定性 | 防止为很小的潜在收益支付额外前处理成本 | 记录检测、搜索、回退的实际A成本；已经搜索后才回退并不会省去搜索成本 |

第一版拟采用两级流程：先计算廉价位姿/内参条件，再在可用训练轨迹上生成A1候选；
用共享接受规则决定保持uniform还是采用候选。其优势假设是**避免无收益的修改并保留有支持的修改**，
不是凭condition凭空创造优于A1的新候选。需要区分“执行搜索前的门控”和“搜索后的接受决策”；只有前者才可能节省搜索时间。

不要现在根据7个覆盖值挑一个恰好分开HW/Mip的阈值。下一探针在训练轨迹ID的固定拆分上，
用一部分观测目标产生候选、另一部分检查支持增益，并用多次固定拆分观察稳定性；同时检查每图网格支持和轨迹长度偏置。
这些拆分仍只使用训练图像观测，相互相关且非独立场景验证，不能代替正式留出图像质量与场景级验证。
待代理可信度明确，预先冻结全局接受规则；若需要调阈值，仅在指定开发场景训练数据上做，保留后续压力场景不调参。

完整消融必须比较 `uniform`、`固定A1规则`、`condition-guided A2`，B全部固定no_reset7000。
若A2与A1总产生相同context，则无新增选择机制证据；若质量无收益且检测更慢，则不晋级。
FOV单条件只作为解释力的负对照，不能因为它近似识别数据集而宣布自适应有效。
条件失配、扰动内参/轨迹抽样后的决策稳定性也应记录，避免边界轻微变化导致大幅换帧。

## 3. GitHub实现对照与可迁移边界

检索/阅读日期2026-10-30。下列为官方仓库固定提交；版本用于追溯，不要求升级本机依赖。
外部论文/仓库的成绩不填写为本项目成绩。建议只迁移可隔离的机制，先满足现有渲染器与标准PLY契约。

| 维度 | 已读实现 | 可借鉴之处 | 本项目适配与优先级 |
|---|---|---|---|
| A / 共视统计、视角 | [hloc pairs_from_covisibility][hloc] `c13273b` | 用实际共享point3D观测构造共视关系 | 借观测结构，不照搬“共视最多的pair”作为16帧选择。当前counter缺points3D.bin，可从images.bin训练观测反向索引；P1 |
| A / 图像质量代理 | [OpenCV Laplacian][opencv] `9d40b84` | 二阶空间差分实现，说明它测边缘变化而非直接测模糊标签 | 用同图合成模糊/局部邻域验证代理，不引入额外图像网络；P2探针 |
| A/B / RNG与实验统计 | [PyTorch fork_rng][rng] `9933f27` | 保存并恢复CPU/设备RNG状态 | 隔离可选网络初始化，分离训练相机/剪枝/分裂随机源；不要求新版PyTorch；P0 |
| B / 调度、opacity状态、评分 | [FastGS模型][fastgs-model] 与 [评分函数][fastgs-score] `44e02a5` | 当前.8限幅/清空动量上游也存在，多视图评分单独消耗前向渲染 | 针对ReSplat初始化验证是否仍必要；先profile和状态连续性，不把继承行为误称本地bug；P0/P1 |
| B / 拓扑停止与恢复 | [3DGS训练循环][3dgs] `54c035f`、[gsplat策略][gsplat-default] `512d366` | 拓扑更新有明确截止点；gsplat支持重置后的暂停 | 将恢复期作为独立参数，保持总预算；不机械照搬长训练的比例；P1 |
| B / 剪枝与预算 | [LightGaussian prune_finetune][light] `6676b98` | 删除后仍训练恢复，而非只在保存前删点 | 借恢复机制；其重要性实现有渲染器依赖，不能把原版预算直接接入当前ReSplat点数；P2 |
| B / 分辨率与生长 | [DashGaussian train_dash][dash] `4e3b560` | 分辨率调度与增密率、点数调度联动 | 本地半→全仅改尺度并非对该方法的复现；先测单位/梯度，避免多因素同时移植；P2 |
| B / 梯度统计、split | [gsplat策略][gsplat-default] 与 [ops][gsplat-ops] | 显式按width/height/相机数换算屏幕梯度，参数和优化器状态随拓扑更新 | 先确认当前FastGS梯度定义，不能再盲乘分辨率；借鉴状态操作和归一化的检查方式；P1微探针/P2训练 |
| B / 抗混叠与尺度 | [Mip-Splatting模型][mip] `dda02ab` | 尺度过滤与opacity补偿是配套关系 | 涉及渲染语义；先验证导出与标准渲染一致性，不只加一个scale正则；P2探针/P3迁移 |
| B / 感知loss | [PerceptualSimilarity][lpips] `082bb24` | 归一化图像、特征距离和感知网络实现 | 对齐评估输入约定，隔离初始化RNG，保长宽比patch，测梯度贡献和新增成本；P2 |
| B / 位姿、曝光 | [gsplat simple_trainer][gsplat-trainer] `512d366` | 提供相机优化等训练接入参考 | 当前FastGS不能直接对相机求导。先测实际位姿/曝光失配，再做新后端梯度和世界系一致性验证；P3 |
| B / 内核和吞吐 | [Faster Gaussian Splatting][faster] `d23bb78` | 独立速度实现和消融组织方式 | 与本机FastGS并非同一后端；先列兼容性与正确性成本，不同时改算法与内核；P3 |

A的I/O优化以本仓库现有解码/resize实现为基准，不需要为了引用外部方案更换算法；外部存储/图像库的吞吐数字也不能代替本机冷运行profile。
共视图方案主要服务A，gsplat/Dash的梯度统计主要服务B；不能把后者依赖已训练场景的统计当成A的免费输入。

## 4. 待测模块队列

文件建议均相对`baseline/`，`probe/`文件则相对仓库根；**未创建的文件是建议名称**。
每项先实现独立开关/适配器，对照只改变该项；共用训练器可复用，但模块配置与输出标签要分开。

| 优先级 | 待测模块 / 文件建议 | 实验与判据 |
|---|---|---|
| P0 | `probe/repro_profile.py`；训练器完整配置/阶段记录 | 已确认同seed、同输入/context的CUDA训练不保证位级一致。正式实验仍统一seed=0且单次运行；只保留配置、哈希、相机序列与阶段计时诊断，不把重复或seed变成消融项，也不混入主表 |
| P0 | `probe/topology_transition.py` | 同一父模型在增密前后记录训练视图render、alpha、损失、点数、尺度、opacity状态；特别检查7000末步。诊断开销不并入正式质量/速度行。只用训练视图判断候选，正式留出评估按冻结协议进行 |
| P1-B1 | `utils_B/terminal_guard.py` | 7000步只跳过最后一次增密/剪枝事件（包含其.8限幅），其余更新次数与no_reset相同。看保存前渲染变化和最终三场景指标；若无明显瞬时/最终损失，不假定末次事件就是主因 |
| P1-B2 | `utils_B/topology_freeze.py` | 独立对照no_reset7000，固定允许增密至5600步、5601–7000只优化现有高斯；保持7000步LR、无周期reset。记录减少的评分时间及恢复曲线。B1若有收益，额外与B1比较以区分末步保护和较长恢复段 |
| P1-B3 | `utils_B/opacity_state.py` | 固定.8限幅与原增密序列，对未被限幅且拓扑映射中保留的opacity条目保留Adam一/二阶矩；实际限幅/新增条目按明确规则处理。先微测试状态映射，再与no_reset7000单项比较。不要同时取消.8限幅或改opacity阈值 |
| P1-A1 | `utils_A/covisibility_guarded.py` | 复用已跑的两交换规则：uniform骨架、16分区/端点、max_swaps=2、平均距离≤1.1×U、p10支持不降、最大间隙≤2×U；只用训练轨迹，无轨迹自动uniform。三主场景A→固定ReSplat→no_reset7000，与uniform配对；验证最终质量及新增A时间，不能只看轨迹目标 |
| P1-诊断 / A1后训练 | `probe/condition_reliability.py` → `utils_A/condition_guided.py` | 依据§2.1联合输入条件、轨迹可靠性和候选边际支持决定0/1/2次交换；共享规则，无数据集ID、无留出质量反馈。先补训练观测拆分/网格/轨迹长度诊断，再冻结接受条件。与uniform及固定A1三方比较，检查新增决策价值和A总成本；不能只证明它比uniform好 |
| P1-诊断 / P2-训练 | `probe/split_conservation.py` → `utils_B/surface_split.py` | 固定训练视图和父PLY，使用同一标准随机样本，量化现有split的均值/协方差/alpha覆盖/渲染跳变；针对薄片面内覆盖设计算子。未通过有限值、坐标、覆盖检查前不训练；不修改ReSplat本体 |
| P2 | `probe/resolution_units.py` → `utils_B/scale_aware_resolution.py` | 同一PLY和视图在半/全分辨率比较普通/绝对梯度分位数、半径、候选率；隔离相机序列。只有单位证据支持才设统一转换，再测2800/4200日程。对照同时含原coarse_to_fine与no_reset，不把阈值拍脑袋翻倍 |
| P2 | `utils_B/score_schedule.py` / `budget_growth_calibrated.py` | 仅若profile证明评分/生长是主要成本，先独立测评分/增密间隔100→200，记录恢复和收敛；若研究点数预算，先按候选类型归一化梯度并记录投影工作量。两种改动分开测试，不把它们捆成同一配置 |
| P2 | `probe/loss_alignment.py` → `utils_B/perceptual_patch.py` | 先用fork_rng隔离网络构造，固定训练抽样；比较保长宽比patch的LPIPS与原L1/SSIM梯度范数/夹角及成本，再决定完整loss试验。尚无证据时不提高权重、不扩大网络 |
| P2 | `probe/quality_calibration.py` / `probe/preprocess_profile.py` | 全集训练图像上做局部纹理/合成模糊校准，误删平滑图则停止；分解A解码/规整/写盘时间。缓存或并行输出须与原图像张量等价，冷/热时间分别报告，不将外移准备当加速 |
| P3 | `utils_B/pose_refine.py`、`sparse_optimizer.py`、曝光/滤波后端 | 只有直接残差证据或profile支持才开始；先通过相机梯度/优化器状态/标准PLY与测试render一致性。当前是依赖门槛未满足，不记录为质量失败 |

关于B2的5600：这是统一的20%恢复期假设，不是根据某个场景结果拟合的最优截止；若失败先分析曲线，
不立刻扫描十个截止点。B3也可能保留不良动量，预期收益只是待证假设。

关于A1接口：当前`select_context`只传中心、方向与图像路径，需要新增可选的训练轨迹数据入口，
在共用预处理器读取源模型的真实观测，再传给选择器；不要从规整后的空观测COLMAP文本获取轨迹。
保持旧策略参数兼容，返回排序且唯一的训练序列位置。DL3DV回退由“缺少可用轨迹”触发，不能写死数据集名来调参。
若轨迹和图像名无法匹配，必须显式记录回退原因；训练与测试集合不得混入选择候选。

## 5. 第一组 checklist：先获得可解释的单项结果

`[x]`只表示本轮确实完成的证据工作；`[ ]`为后续执行项。以下顺序不是要求同时启动全部实验。

- [x] 全集数据清点：13场景3009图像，位姿、训练划分、路径/FOV、图像代理、三种选帧几何与7场景轨迹覆盖。
- [x] 三主场景固定点样本与混合参考点云诊断，记录视锥不考虑遮挡的限制。
- [x] 已有44个完整评估条目核查；逐视图差值、相同输入哈希、PLY尺度/opacity与DL3DV残差分解。
- [x] CPU微探针复现opacity无值变化仍丢动量、LPIPS初始化改变剪枝RNG；解析核查末步增密/分裂缩放。
- [x] 全集受约束两交换探针：7场景实际交换，6场景按缺轨迹回退；不声称新PSNR/LPIPS结果。
- [x] 全集condition描述审计：FOV、角度缺口/FOV、空间距离p95、context比例、轨迹支持与已有交换增益；没有根据这些数值拟合路由阈值。
- [ ] A条件可靠性探针：补轨迹长度/图像网格支持、训练观测固定拆分及条件扰动；区分覆盖缺口、可实现增益与代理可信度。
- [x] P0配置口径：正式运行固定seed=0、单配置/场景一次；记录命令、提交、split、输入PLY哈希。诊断确认同seed路径并非位级确定，诊断重复未入主表。
- [x] P1-B1 terminal_guard：三主场景完成；只跳过末步事件有改善，但弱于长恢复段。
- [x] P1-B2 topology_freeze5600/7000：三主场景均改善，且最终点数并未减少，支持“固定拓扑恢复”机制；当前10%时间档首选。
- [x] P1-B3 opacity_state：CPU状态映射微测试和三主场景完成；质量改善伴随点数/时间明显增加，归为质量优先而非10%时间档。
- [x] P1-A1：接入训练轨迹接口，精确重现全集7场景各2次交换及6场景回退；三主场景正式运行完成。billboard LPIPS相对配对控制+0.0007、counter -0.0002，代理改善未稳定转化为质量。
- [x] A1后决策：暂不实现condition-guided A2。A2只能在A1候选中路由，现有两个真实换帧场景没有稳定收益；先等待能处理遮挡/内容分布的可靠性证据，不用FOV或数据集身份拟合阈值。
- [x] 组合：完成 topology_freeze + opacity_state；三场景质量最佳，但counter的B约为经典30k的10.56%，归为质量优先配置。
- [x] 晋级规则按当前工作原则修订：正式表保持固定seed0单次运行；小效应不因额外seed或重复中的最好值晋级。
- [x] split守恒探针：三个ReSplat输入各固定抽样2048个父高斯，最短轴均100%处于`1e-6`；当前split的切面线性质量比为0.78125，平均质心漂移0.87–0.90个父尺度，切向父坐标二阶矩约膨胀到1.37–1.41。long-axis仍把另一切向矩压到0.390625，与v0质量失败机制一致。
- [x] `surface_conserving`算子设计：两个子点在父高斯局部切平面作反对称单位椭圆位移，两个切向尺度除以`√2`、最短法向尺度保持不变；解析上切面质量=1、质心漂移=0、期望二阶矩=1。三场景正交切面alpha代理的积分比0.990–0.994、NRMSE 0.477–0.480，优于当前0.749–0.771和0.783–0.796；97²网格复核结论不变。
- [x] P2 surface_split正式实验：只替换split算子，其余固定为no_reset7000；三主场景LPIPS相对配对控制为−0.0030/−0.0002/−0.0012，PSNR为+0.610/−0.020/+0.192 dB。最终点数减少0.83%/1.36%/0.92%，B用时也减少0.78/0.23/2.74秒，说明收益不靠扩大模型；HW效应很小，仍需看组合是否稳定。
- [x] P2组合 surface_split + topology_freeze：相对topology_freeze，三场景PSNR为+0.060/+0.001/+0.061 dB，LPIPS为+0.0003/−0.0005/−0.0006；点数均略减，B占经典30k训练时间7.83%/4.94%/8.90%。等权均值略优并保住10%档，作为当前质量×速度候选。
- [x] split方向停止点：组合对topology_freeze的增量很小，且DL3DV LPIPS反向、24/42视图变差；不再扫描切向尺度/角度，也不继续叠加opacity形成三因素组合。保留可解释的两因素结果，转向新的独立机制探针。
- [ ] 几何压力验证：固定配置增加DL3DV `0b5635d53b3b…`、HW `trashbin`、Mip `bicycle`；补必要经典30k参考，禁止根据新场景逐个调参。
- [ ] 最终场景级验证：参数冻结后运行其余7个本地场景；先分别汇总三个数据集，再宏平均，避免DL3DV场景数较多主导结论。

P2/P3不自动排队运行。P1失败时，先根据已记录的轨迹和残差回答原因，再决定是否激活对应P2；
不为了“完成清单”补跑已经没有证据支持的组合。

## 6. 保存、收集和停止规则

探针代码保存在仓库`probe/`，输出放`results/rethink_v0/`（本轮）或`results/probe_v1/`（后续）。
完整实验继续使用 `results/<A名>_A_<B标签>_B/<group>/<scene>/`；B标签包含算法与步数，
例如 `baseline_A_terminal_guard7000_B`。所有正式配置固定seed=0且每场景单次运行，seed和重复号不进入模块名或表格。禁止覆盖已有结果。
只跑B复用A/PLY时明确标记，缺失A/ReSplat时间填“—”；正式端到端行必须来自同一次完整管线，不能拼接时间。

每个候选保存实际生效参数、Git提交/dirty标记、环境与权重校验和、split/context/输入PLY哈希、原始训练指标、
点数及状态轨迹、标准PLY/PNG、`module_b.json`、`metrics.json`。高频诊断文件额外标注instrumented，不参与正式时间排名。
现有`collect_results.py`继续收集单次结果；诊断重复只用于排查非确定性，不作为候选、不汇总进正式表。

主表新小节列：设计与单因素、固定参数、三主场景指标/Δ、A/ReSplat/B/总秒数、B/30k、
逐视图退化摘要、通过/停止原因；全集扩展另表，不能用缺失场景凑平均。原始results不提交Git。
若质量收益不超过噪声且时间更差，停止；若代理改善但质量不变，记录“代理未转化”而不继续盲增目标权重；
若只有一数据集获益，作为条件假设保留，不按场景定制参数冒充通用方案。

[hloc]: https://github.com/cvg/Hierarchical-Localization/blob/c13273bd0ecc2917a35910fd843712a1c6243193/hloc/pairs_from_covisibility.py
[opencv]: https://github.com/opencv/opencv/blob/9d40b841070eb9f2b7fe2935dea44f184672d224/modules/imgproc/src/deriv.cpp
[rng]: https://github.com/pytorch/pytorch/blob/9933f275f857870234004d951579fa40fd55c403/torch/random.py
[fastgs-model]: https://github.com/fastgs/FastGS/blob/44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f/scene/gaussian_model.py
[fastgs-score]: https://github.com/fastgs/FastGS/blob/44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f/utils/fast_utils.py
[3dgs]: https://github.com/graphdeco-inria/gaussian-splatting/blob/54c035f7834b564019656c3e3fcc3646292f727d/train.py
[gsplat-default]: https://github.com/nerfstudio-project/gsplat/blob/512d366b67073d77ca099ede742683c165dfc23b/gsplat/strategy/default.py
[gsplat-ops]: https://github.com/nerfstudio-project/gsplat/blob/512d366b67073d77ca099ede742683c165dfc23b/gsplat/strategy/ops.py
[light]: https://github.com/VITA-Group/LightGaussian/blob/6676b983e77baadd909effc56a6aaadafa964dcc/prune_finetune.py
[dash]: https://github.com/YouyuChen0207/DashGaussian/blob/4e3b5606e593b5e58b90a5c3ea8c421bedc308a1/train_dash.py
[mip]: https://github.com/autonomousvision/mip-splatting/blob/dda02ab5ecf45d6edb8c540d9bb65c7e451345a9/scene/gaussian_model.py
[lpips]: https://github.com/richzhang/PerceptualSimilarity/blob/082bb24f84c091ea94de2867d34c4544f68e0963/lpips/lpips.py
[gsplat-trainer]: https://github.com/nerfstudio-project/gsplat/blob/512d366b67073d77ca099ede742683c165dfc23b/examples/simple_trainer.py
[faster]: https://github.com/nerficg-project/faster-gaussian-splatting/blob/d23bb784d8f19b2432d847b80418d56f0494f599/README.md
