# Pretrain, Then Convert：项目详细说明（给接手的 agent）

本文件说明这个项目做了什么实验、为什么这样设计、怎么跑、结果在哪里、怎么复现和扩展。读完它，再看两个仓库的 README，
就可以独立复现论文里的任何数字、重建论文，或者在这个框架上加新实验。

- 论文：*Pretrain, Then Convert: Compute-Optimal Training of One-Step Image Generators*（ICML 2026 模板，正文 8 页，全文 32 页）
- 状态（2026-11-03）：**项目已完成**。全部 411 个训练任务、全部评测、论文都已完成；没有在跑的作业。
- 两个仓库（均为本地 git，未推送）：
  - 代码库 `pretrain_then_convert/`：模型、损失、训练、评测、全部实验的定义；
  - 写作库 `pretrain_then_convert_paper/`（本仓库）：论文 LaTeX、分析脚本、结果数据、Overleaf 包。

---

## 目录

1. 研究问题和结论
2. 文件地图
3. 方法背景（最少需要懂的）
4. 模型和训练设置
5. FLOP 计量
6. 实验设计（逐组说明）
7. 评测协议
8. 统计与分析方法
9. 结果速查：数字 → 表 / 图 → 脚本
10. 从零复现
11. 在此基础上做新实验
12. 算力和时间
13. 踩过的坑

---

## 1. 研究问题和结论

### 1.1 问题

一步生成器（1 NFE：一次网络前向就出图）有两条训练路线：

- **原生（native）**：从头到尾用一步目标训练。本文的一步目标是 improved MeanFlow（iMF）。
- **先预训练再转换（pretrain→convert）**：先用总预算的一部分 `f` 训练多步的 flow matching（FM）模型，再把它转换成一步模型，
  用剩下的 `1−f` 继续用 iMF 训练。`f = 0` 就是原生训练。

问题：**给定训练预算 C（FLOPs），f 该取多少？模型多大？为什么划算？** 之前的工作各自固定一个预训练长度和一个模型，
没人在固定总算力下系统测过。

### 1.2 结论（关键数字）

| 结论 | 数字 | 论文位置 |
|---|---|---|
| 转换在每个格子都赢原生 | 16 个两条路线都训练过的（尺寸, 预算）格子全部更好，FID 好 1.08–2.14 倍，15 个的区间不含 1 | 6.1 节，Fig. isoflop，Table 1 |
| 前沿上的优势 | 原生 / 转换 = 1.46、1.31、1.31、1.14（1e19 → 3e20） | Table 1 |
| 1e21 FLOPs 单次运行 | 转换 XL（f=0.6）FID 1.75，原生 XL 1.91（比 1.09，区间 [0.95, 1.25]）；FD-DINOv2 上转换好 1.20 倍 | 6.1 节，Fig. samples |
| 最优分配约 60% | 各预算 f* = 0.56、0.68、0.64、0.55；平均 0.60 [0.53, 0.72]；趋势 −0.01 [−0.16, +0.24] / 每 10 倍算力 | 6.2 节，Fig. fstar_stab |
| 最优点附近很平 | 最优尺寸下 f ∈ [0.25, 0.8] 都在最好值的 3–9% 内；原生差 1.3–1.7 倍，f=0.99 差 1.8–2.4 倍 | Fig. overview、landscape |
| 机制：样本交换 | 一个 iMF 步 = κ ≈ 13/6 + 8/D 个 FM 步（实测 2.34–2.87）；同样算力下转换模型多看 m(f) = 1 + f(κ−1) 倍样本 | 4 节，Eq. kappa / samples |
| FM 样本的价值 | 同样本数比较：f≈0.1、0.25、0.5 时 1.09、1.09、1.05，之后 0.95、0.91、0.72、0.34（f=0.65…0.99）；最优 f 处原生 / 转换 = 0.99 [0.94, 1.05] | 6.3 节，Table exchange / matched |
| 两成分模型 | log FID = 速度项 + 一步缺陷项；FM 样本在速度项里值 λ = 1.64 个一步样本，δ = 0.71；69 个转换的拟合误差低于种子噪声 | 6.3 节，附录 twocomp |
| 原生偏好更小的模型 | 原生最优尺寸比多步小 2.1–3.3 倍（预测 1.6–1.9 倍）；3e20 时原生 241M、转换 497M、FM 536M | Table nstar |
| shortcut 目标也成立 | 原生 / 最优转换 = 1.06、1.14、1.33（三个格子，三种子）；f* = 0.30–0.52 | 6.3 节，附录 shortcut |
| 一条定律描述两条一步路线 | 转换前沿 = 原生定律在 m̂·C 处；m̂ = 1.78 [1.59, 2.07]，和步成本推出的 1.67–1.99 一致；预测 1e21：原生 1.95（实测 1.91）、转换 1.83（实测 1.75） | 6.4 节，Fig. laws，Table joint |
| 外推更稳 | 用 1e19–1e20 拟合预测 3e20：联合定律误差 1% / 2%，单独原生定律 18% | Table heldout |
| 一步生成省生成算力 | 每张图生成算力 = 网络前向次数 × 前向 FLOPs + VAE 解码器 620 GFLOPs。1e21 转换 XL：0.97 TFLOPs、FID 1.75；同等或更好质量的最便宜多步图（XL@3e20，32 步 Heun，46 次前向，FID 1.53）16.6 TFLOPs = 17 倍（只算网络 46 倍）；1e21 多步 H 50 步 = 56 倍。4 步时所有多步模型 FID > 8，8 步 > 3.3 | 6.5 节，Fig. gencost，Table steps |
| 算上生成的总算力 | 训练 + 生成 K 张图：K = 1e7 时多步几乎处处最便宜；每个预算的转换模型在 K > 1.8–4.5e7 后成为达到其 FID 的最便宜路线（1e19 从第一张图起）；K = 1e9 时 FID 1.75 转换 2.0e21 FLOPs，多步贵 8.5 倍 | 6.5 节，Fig. gencost b/c |
| 一步税 | 最好 1-NFE / 最好 50 步 FID：2.21（1e19）→ 1.34（1e21） | 6.4 节 |
| 蒸馏对比 | 计入教师算力时 DMD2 学生比转换差 2.7–6.6 倍；FM 模型已有时，用 3e19 新算力转换得 2.38（XL）/ 3.20（L），DMD2 学生 2.80 / 3.26 | 6.6 节，附录 distill |

---

## 2. 文件地图

### 2.1 代码库 `pretrain_then_convert/`（50 个文件）

```
osr/
  models.py         imfDiT（官方 iMF PyTorch 推理代码的移植，参数名不变，可直接加载官方权重）；
                    add_v_head_from_u() / freeze_new_condition_embedders() 实现保函数转换
  losses.py         fm_loss、imf_loss（官方 JAX 目标逐行移植）、shortcut_loss
  jvp.py            手写的前向模式 JVP（可 torch.compile），iMF 训练用
  train.py          统一训练器：mainline / cooldown / convert / extend，所有触发点按实测累计 FLOPs
  train_dmd.py      DMD2 蒸馏（CC BY-NC-SA 4.0）
  data.py, ema.py   缓存 latent 的数据管线；power-function EMA（EDM2）
  flops.py          用 FlopCounterMode 实测每步 FLOPs
  sample.py         一步 / 多步采样器
  evaluate.py, extra_metrics.py   FID / IS；FD-DINOv2、ADM-FID、precision / recall
  experiments.py    ★ 论文里每一个训练任务的定义（组、run id、父节点、配置）
  paths.py          所有路径由环境变量决定（OSR_WORK、OSR_LATENTS、OSR_RUNS、OSR_EVAL、OSR_REF、OSR_VAE、OSR_DINO_REPO）
  flops_table.json  每个尺寸、每个目标的每样本训练 FLOPs
scripts/
  launch.py         列出 / 训练 / 评测 / 生成 SLURM 脚本
  encode_latents.py ImageNet → SD-VAE latent
  dino_ref_stats.py FD-DINOv2 参考统计、ADM 参考图的 Inception 特征
  eval_run.py       主评测（选 EMA、扫 guidance、FID-50k）
  eval_gridext.py   最优 guidance 落在网格边上时继续搜索
  eval_gridpolish.py 多步模型在 8 个共享候选点上统一打分
  eval_extra.py     FD-DINOv2、ADM-FID、precision / recall
  eval_fdsel.py     FD-DINOv2 用自己选的 guidance
  eval_prcurve.py   沿 guidance 扫描的 precision / recall
  eval_fid.py       任意 checkpoint 在指定 guidance 下的 FID（也用于官方 iMF 权重）
  fid_sample_noise.py FID-50k 的采样误差（对半分）
  sample_grid.py    同噪声的样本网格
  branch_diagnostics.py FM 分叉点的路径直线度和一步误差
  flops_table.py, bench_step.py, validate_pipeline.py, compact_finals.py, self_test.py
configs/examples/   几个训练配置样例
slurm/train.sbatch  launch.py 用的 SLURM 模板（需要自己加账号 / 分区 / QOS 行）
tests/              pytest（CPU 测试；shortcut 测试要 GPU）
```

### 2.2 写作库 `pretrain_then_convert_paper/`

```
paper/          LaTeX（= Overleaf 项目）：main.tex、refs.bib、ICML 样式文件、tables/（29 张）、figures/（17 个）
analysis/       30 个分析脚本 + run_all.sh；analysis/out/ 为中间结果（git 忽略，可重算）
data/           结果数据快照（见 2.3）
tools/          build.sh（编译 + 8 页检查）、check_pages.py、make_overleaf.py
docs/           本文件
build/main.pdf  编译好的论文（git 忽略）
pretrain_then_convert_overleaf.zip   Overleaf 包（git 忽略，tools/make_overleaf.py 生成）
```

### 2.3 数据快照 `data/`

| 路径 | 内容 | 来源 |
|---|---|---|
| `runs.csv` | 411 个训练任务：run_id、route、kind、size、seed、budget、f、lr、flops_end、parent（`父run:里程碑`） | 和 `osr/experiments.py` 逐项一致 |
| `eval/<run>/summary.json` | 选中的权重（raw / ema050 / ema100）、guidance 点、`fid50k`、`fid10k_sel`、`fid_2nfe_10k`、`fid_noguid_10k`；`grid1` 字段 = 扩展搜索前的值 | `eval_run.py` + `eval_gridext.py` + `eval_gridpolish.py` |
| `eval/<run>/extra.json` | `fd_dinov2`、`fid_adm`、`precision`、`recall`、`inception_score` | `eval_extra.py`（或 stage C 内） |
| `eval/<run>/fdsel.json` | FD-DINOv2 自选 guidance 的结果（36 个 run，清单 `analysis/fd_dinov2_runs.txt`） | `eval_fdsel.py` |
| `eval/<run>/prcurve.json` | guidance 扫描上的 P/R（12 个前沿模型，清单 `analysis/pr_curve_runs.txt`） | `eval_prcurve.py` |
| `eval/<run>/gridext.json` | 扩展搜索记录 | `eval_gridext.py` |
| `eval/official_imf/results.jsonl` | 官方 iMF 权重用我们管线测的 FID | `eval_fid.py --official` |
| `run_epochs.json` | 每个 run 的每样本 FLOPs 和训练结束时的 epoch 数 | 训练日志 |
| `flops_table.json`、`throughput.json` | 每样本 FLOPs；每卡吞吐 | `flops_table.py`、`bench_step.py` |
| `lit/verified_results*.json` | 已发表结果，每个数都带论文 arXiv 源码里的原文引用和位置 | 手工 + arXiv API 核对 |
| `samples/` | 样本图的无损原图 | `sample_grid.py` |
| `frozen/` | 需要训练好的 checkpoint 或作业记录才能算的结果：分叉点诊断、FID 采样噪声、GPU 小时数 | `branch_diagnostics.py`、`fid_sample_noise.py`、sacct |

---

## 3. 方法背景（最少需要懂的）

### 3.1 Flow matching（FM，多步）

- 路径 `z_t = (1−t)·x + t·ε`，`t ∈ [0,1]`，t=1 是噪声、t=0 是数据；目标速度 `ε − x`。
- 损失：`‖v_θ(z_t, t, y) − (ε − x)‖²`；t 取 logit-normal（μ=−0.4，σ=1.0）；10% 的样本类别换成空类（无条件）。
- 采样：50 步 Euler，classifier-free guidance（两次前向），guidance 只在区间 `[t_min, t_max]` 内生效。

### 3.2 MeanFlow / iMF（一步）

- 平均速度 `u(z_t, r, t) = 1/(t−r) ∫_r^t v ds`；一步就能从 `z_t` 跳到 `z_r`。
- MeanFlow 恒等式：`v = u + (t−r)·du/dt`，其中 `du/dt = ∂_z u·v + ∂_t u` 是一个 JVP（雅可比-向量积）。
- iMF 的网络 `u_θ(z_t, h, ω, y)`：`h = t − r`；guidance 尺度 ω（以 `1 − 1/ω` 输入）和区间作为输入；外加一个辅助 v-head（h=0 时的瞬时速度）。
- guidance 折进训练目标：`v_g = v_t + (1 − 1/ω')·sg[v_θ(z|y,ω') − v_θ(z|∅,1)]`。
- 复合预测 `V = u + (t−r)·sg[∂_z u·v_θ + ∂_h u]`，两个 head 都回归 `v_g`，用自适应权重 `ℓ / sg(ℓ + 0.01)`。
- (r, t) 是两个 logit-normal 抽样的 min / max，每个 batch 一半样本 r = t（此时就是带 guidance 目标的 FM）。
- 一步采样：`x̂ = ε − u_θ(ε, h=1, ω, y)`。
- 训练时 ω ∈ [1, 8]，幂律密度，`β_cfg` = 1（XS、S、B）或 2（M–H）；区间 `t_min ~ U[0, 0.5]`、`t_max ~ U[0.5, 1]`。

### 3.3 保函数转换（function-preserving conversion）

- FM 模型就是同一个 imfDiT，训练时固定 `h=0、ω=1、区间 [0,1]`，没有 v-head。
- 转换时：h、ω、t_min、t_max 这四个新条件嵌入器的最后一层 `W_2 ← 0`、`b_2 ← η(s_0)`（s_0 是 FM 训练时的取值），
  所以输出恒等于 FM 时的值；v-head 复制 u-head。
- 结果：转换后的网络和 FM 网络**输出完全相同**；一步采样等于 FM 模型走一步 Euler。嵌入器仍可训练（W_2 的梯度非零）。
- 优化器和 EMA 都重新开始。代码：`imfDiT.add_v_head_from_u()`、`imfDiT.freeze_new_condition_embedders()`。
- 消融：随机初始化新嵌入器和 v-head，FID 差 11%（M@3e19）和 7%（M@1e20）。

### 3.4 步成本 κ 和样本倍数 m(f)

- FM 一步每样本 `3F`（前向 F + 反向 2F）。
- iMF 一步每样本 ≈ `(2.5 + 1 + 3 + 24/D)·F`：无梯度的 guidance 前向（2.5F）+ JVP 切向（≈F）+ u 路径前后向（3F）+ 8 块 v-head 的前后向（24F/D）。
- 所以 `κ = c_iMF / c_FM ≈ 13/6 + 8/D`：D=12 时 2.83（实测 2.85–2.87），D=24 时 2.50（2.52），D=32 时 2.42（2.43），D=48 时 2.33（2.34–2.35）。
- 同样算力下转换模型看到的样本数是原生的 `m(f) = 1 + f(κ − 1)` 倍，最优 f 处 1.7–2.0。
- shortcut 模型一步只要 1.20 个 FM 步。

### 3.5 其它路线

- **Shortcut models**（Frans et al.）：h 当作步长 d；3/4 样本是 FM 目标（h=0），1/4 是自洽目标（2d 步 = 两个 d 步，用 EMA 目标网络，衰减 0.999），
  7 个层级等份；最小层级用 CFG 1.5；weight decay 0.1；一步采样 h=1 不加 guidance。
- **DMD2**：冻结的多步教师（冷却后的 FM 模型）→ 一步学生；在线 fake score（从教师初始化）+ 判别器头；每个生成器步 5 次 fake 更新；
  lr 3e-5；batch 128；学生从学生尺寸的 FM 模型（8.5e18 FLOPs 处）初始化。

---

## 4. 模型和训练设置

### 4.1 模型族（imfDiT，patch 2，32×32×4 latent，256 个 token）

| 尺寸 | 深度 | 宽度 | 头数 | 参数（推理路径） | 每个 run 的 GPU 数 | micro-batch（FM / iMF） |
|---|---|---|---|---|---|---|
| XS | 12 | 384 | 6 | 21M | 1–8 | 256 / 128 |
| S | 12 | 512 | 8 | 38M | 1–8 | 256 / 128 |
| B | 12 | 768 | 12 | 85M | 1–8 | 128 / 64 |
| M | 24 | 768 | 12 | 170M | 2–8 | 64 / 64 |
| L | 32 | 1024 | 16 | 403M | 4–8 | 64 / 32 |
| XL | 48 | 1024 | 16 | 604M | 8 | 32 / 32 |
| H | 48 | 1536 | 24 | 1.36B | 8 | 32 / 16 |

- B、M、L、XL 是官方 iMF 配置；XS、S、H 是我们补的。
- 结构：RMSNorm、RoPE、QK-norm、SwiGLU；类别、ω、区间、h 分别是 8、4、2×2、4 个上下文 token；**t 不是输入**（模型只看 h）。
- u-head 和 v-head 各 8 个 block。FM 模型没有 v-head。
- 移植正确性：官方 iMF-XL/2 权重用我们的管线测得 FID 1.73（论文 1.72），B/2 3.33（论文 3.39，官方代码 3.32）。

### 4.2 数据

- ImageNet-1k 训练集（1,281,167 张），ADM 中心裁剪到 256，SD-VAE（`sd-vae-ft-mse`）编码。
- 每张图存原图和水平翻转两份的后验均值和标准差（fp16，约 40 GB）；训练时每步从后验采样，随机翻转，按通道归一化（iMF 的常数）。
- 每个 epoch 的排列和翻转只由 (seed, epoch) 决定，所以断点续训是精确的。

### 4.3 训练超参数（所有路线）

| 项 | 取值 |
|---|---|
| 全局 batch | 256（所有尺寸、所有路线；DMD2 是 128） |
| 优化器 | AdamW，β=(0.9, 0.95)，无 weight decay（shortcut 用 0.1），无梯度裁剪 |
| 学习率 | 1e-4 恒定（WSD 的稳定段，iMF 原论文的值）；每个预算最后 15% 线性衰减到 0 |
| 主线 warmup | 线性，min(5000 步, 到第一个冷却起点步数的 10%) |
| 转换 schedule | warmup min(2000 步, 10%)，恒定，最后 15% 线性衰减 |
| 精度 | bf16 autocast，fp32 主权重，`torch.compile` |
| EMA | power-function EMA，σ_rel ∈ {0.05, 0.10}；评测时在 raw / EMA0.05 / EMA0.10 里选 |
| checkpoint | 每 20 分钟一个可续训的 checkpoint；收到 SIGTERM / SIGUSR1 时也保存，SLURM 下自动 requeue |

### 4.4 WSD 协议（这是让 IsoFLOP 网格算得起的关键）

```
FM 主线 (r1_<size>_s<seed>)  ──恒定 lr──────────────────────────────────▶
    │ f·C 处存 raw 权重（"weights" 里程碑）       │ 0.85·C 处存完整状态（"full" 里程碑）
    ▼                                             ▼
  转换 r3_<size>_<C>_f<f>_s<seed>              冷却 r1_<size>_<C>_cd_s<seed>
  (iMF，从 f·C 训到 C，最后 15% 衰减)           (lr 从 0.85·C 线性降到 0，得到多步模型)

原生 iMF 主线 (r2_<size>_s<seed>) ──恒定 lr──▶ 0.85·C 处分叉 ──▶ 冷却 r2_<size>_<C>_cd_s<seed>（原生一步模型）
```

- 每条路线、每个尺寸只训练**一条**恒定学习率的主线；每个预算的模型都是从主线分出的短分支。
- 所有触发点（里程碑、冷却起点、转换起点、终点）都按**实测累计 FLOPs**，不按步数。
- 一条主线训练到它所有分支需要的最远点（例如 FM 主线到 0.99·C_max）。

---

## 5. FLOP 计量

- 训练算力是**实测**的：`osr/flops.py` 用 `torch.utils.flop_counter.FlopCounterMode` 统计一个完整训练步的矩阵类运算
  （前向、无梯度 guidance 前向、JVP 切向、反向），训练器每步累加。逐元素运算不计（通用惯例）。
- JVP 按单遍计（等同于 JAX 的执行方式）。我们实际用的快速实现会重算一次 primal，真实执行 1.12 倍计数的 FLOPs，这部分不计。
- 注意力按 math 路径计，flash attention 的重算不计。
- FM 成本是 `6N·256` 经验公式的 1.11–1.21 倍（注意力和嵌入器）。
- 每样本 FLOPs（G = 1e9）：

| 尺寸 | FM | iMF | κ |
|---|---|---|---|
| XS | 39 G | 113 G | 2.87 |
| S | 68 G | 195 G | 2.86 |
| B | 149 G | 425 G | 2.85 |
| M | 298 G | 751 G | 2.52 |
| L | 697 G | 1694 G | 2.43 |
| XL | 1045 G | 2453 G | 2.35 |
| H | 2318 G | 5431 G | 2.34 |

- 「高效 iMF 实现」（JVP 只算 r≠t 的一半，区间调整的 guidance 前向只算 t 在区间外的样本）会让 κ 低 10–12%；
  按这个口径算，转换仍在每个预算都赢（Table efficient）。

---

## 6. 实验设计（逐组说明）

全部 411 个任务定义在 `osr/experiments.py`，按 11 个组组织（`python -m osr.experiments [组名]` 打印每组的任务数和 FLOPs）。
总计 1.31e22 FLOPs。

### 6.0 run id 命名规则

| 形式 | 含义 |
|---|---|
| `r1_<尺寸>_s<种子>` / `r2_<尺寸>_s<种子>` | FM / 原生 iMF 主线 |
| `r1_<尺寸>_<C>_cd_s<种子>` | 预算 C 的多步 FM 模型（冷却分支） |
| `r2_<尺寸>_<C>_cd_s<种子>` | 预算 C 的原生一步模型（冷却分支） |
| `r3_<尺寸>_<C>_f<f>_s<种子>` | 预算 C、分配 f 的转换模型 |
| `r1_<尺寸>_ext<C>_s0`、`*_rerun*_s0` | 延长 / 重跑 FM 稳定段，提供额外分叉点 |
| `*_lr<率>*`、`*_flr<率>*` | 学习率变体（flr：FM 预训练和转换都用这个率） |
| `*_s0st1` | 种子 0 的预训练，最后一段用种子 1 重跑 |
| `*_naive_s0` | 非保函数（随机）初始化的转换 |
| `r4_<学生><教师><C_T>_d<C_D>_s0`、`r4s_*` | DMD2 学生；超参扫描 |
| `t2_*`、`t3_*`、`g3fm_*` | 原生 / 转换 shortcut 模型；shortcut 种子 1、2 的 FM 主线 |
| `v1_*`、`v2_*` | 1e21 FLOPs 的运行（v1：原生 XL；v2：转换和多步 H） |
| `r3_<尺寸>_sunk<C>_d3e19_s0` | 已有 FM 模型 + 3e19 新算力的转换 |

### 6.1 `grid`：IsoFLOP 主网格（165 个任务：22 主线、51 冷却、92 转换）

- 目的：每条路线在每个预算上的 FID-尺寸曲线，找最优尺寸和前沿。
- 预算：1e19、3e19、1e20、3e20 FLOPs。
- 每个预算的尺寸窗口（让最优点落在窗口内部）：

| 预算 | 多步 FM（r1） | 原生（r2） | 转换（r3） |
|---|---|---|---|
| 1e19 | S B M L | S B M L | S B M L |
| 3e19 | B M L XL | B M L XL | B M L XL |
| 1e20 | B M L XL H | B M L XL H | B M L XL |
| 3e20 | M L XL H | B M L XL H | L XL H |

- 转换的 f 网格：{0.5, 0.8, 0.95, 0.99}。
- 1e19 和 3e19 有完整的第二个种子（独立的 FM 主线、原生主线、冷却和转换），用来估计种子噪声。

### 6.2 `split`：最优尺寸处补 f≈0.25 和 f≈0.65（22 个任务）

- 目的：f 网格在 0.5 附近太稀，补点以准确定位 f*。
- f=0.65：从最近的 full 里程碑延长 FM 稳定段到 0.65·C（`r1_<尺寸>_ext<C>_s0`）；1e19 时重跑 S、B 的 FM 主线到 6.5e18（`r1_*_rerun_s0`）。
- f≈0.25：用已有分叉点（如 3e19 用 1e19 网格的 f=0.8 点 = 8e18，即 f=0.27）。
- 覆盖：B、M@3e19，M、L@1e20，L、XL@3e20，S、B@1e19。

### 6.3 `edges`：网格边缘以外的尺寸（14 个任务）

- 网格窗口的下边缘：1e19 时转换和原生的最优尺寸都是 S（窗口里最小的尺寸），3e19 时原生的最优在窗口外。
  所以加了 XS@1e19（三条路线都做，转换 f ∈ {0.25, 0.5, 0.65, 0.8}）、原生 S@3e19（延长原生 S 主线）和 XS@3e19，以及原生 S@3e19 的第二个种子。
  结果：3e19 的原生前沿是 S（6.20），XS 更差（7.02），所以前沿已被包住。

### 6.4 `stage_seeds`：只换最后一段的种子（6 个任务）

- 同一个种子 0 的里程碑，冷却或转换段用种子 1 重跑：L、XL 的转换（f=0.5）和 FM 冷却、M、L 的原生冷却，都在 3e20。
- 结论：只换最后一段，FID 变化 0.1–1.0%（冷却）和 1.2%、2.5%（转换），远小于整条重跑的 1.9–5.5%。
  说明同一主线分出的分支之间的比较很准。

### 6.5 `lr`：预训练学习率（55 个任务）

- B、L@1e19（FM 和原生，lr 5e-5 / 2e-4 / 4e-4），B@3e19（2e-4），原生 M@1e20 和 3e20（2e-4 / 4e-4），
  原生 S 和转换最优点（S@1e19 f=0.5、M@3e19 f=0.65，FM 和转换都用同一率；2e-4 / 4e-4 / 8e-4）。
- 结论：1e19 时最好的率（4e-4）让原生 S 和转换 S 都好 15%；各自最好率下转换仍赢 1.51 倍（1e19）和 1.34 倍（3e19）（1e-4 时 1.50 和 1.28）；
  1e20 和 3e20 时原生最好率只好 2% 和 1%；8e-4 每一行都更差（最优在扫描范围内）。

### 6.6 `conversion_lr`：只调转换段的学习率（24 个任务）

- M@3e19 整条 f 曲线（5e-5 / 2e-4 / 4e-4），B@1e19 短转换，XL@3e20 最优附近（5e-5）。
- 结论：一半的率让最优分配好 8%（M@3e19）和 4.8%（XL@3e20），f* 从 0.63→0.69、0.55→0.54，仍在平坦区内。

### 6.7 `ablations`（6 个任务）

- f≈0.1：M@3e19（f=0.083，从 2.5e18 的 FM 状态）和 M@1e20（f=0.095）。只预训练 8–10% 就比原生好 14%。
- 初始化消融：M@3e19 和 M@1e20、f=0.65，随机初始化新嵌入器和 v-head（`init_mode: naive`）。差 11% 和 7%。

### 6.8 `runs_1e21`：1e21 FLOPs 的检验（7 个任务）

- 尺寸由网格上拟合的 FID(N, C) 曲面决定：XL 在两条一步路线最优的 1.2% 内，H 在多步最优的 0.6% 内。
- 任务：原生 XL（`v1_XL_main_s0` → `v1_XL_1e21_cd_s0`）；转换 XL f=0.6（XL 的 FM 主线从 cd_3e20 延长到 0.6e21，`v2_XL_ext_s0` → `v2_XL_1e21_f0.60_s0`）；
  H 的 FM 主线 `v2_H_fm_s0`（带 f=0.45…0.8 的分叉点）→ 转换 H f=0.6 和多步 H 冷却。
- 结果：转换 XL 1.75、原生 XL 1.91、转换 H 1.99（曲面预测 2.01）、多步 H 1.31。
- 每个 run 用 8 卡；两节点 DDP 在 batch 256 时扩展很差，所以都是单节点。

### 6.9 `distillation`：DMD2（43 个任务）

- 教师：L、XL 的 FM 冷却模型，1e20 和 3e20。学生：B、M，从学生尺寸的 FM 模型（cd_1e19 里程碑，8.5e18 FLOPs）初始化。
- 蒸馏预算：1e19、3e19、1e20（批 128 下 4.8k–77k 个生成器步）。
- 同尺寸学生：L←L@1e20、XL←XL@3e20，从教师本身初始化（selfinit）。
- 超参扫描 `r4s_*`：B 教师、2e18 FLOPs，单因素扫 lr、TTUR、GAN 权重；最后选 lr 3e-5。
- 算力记账：`C_total(k) = C_T/k + C_D`，k=1（计教师）、16（16 个学生分摊）、∞（教师沉没）；学生初始化的 8.5e18 计入 C_D。
- 结论：计教师时 B、M 学生比转换差 2.7–6.6 倍；同尺寸学生差 1.24–1.33 倍，分摊到 16 个学生时好于转换（0.75–0.76），教师沉没时 0.60–0.69。

### 6.10 `sunk`：FM 模型已有时（2 个任务）

- 把同尺寸 DMD2 学生用的教师（XL@3e20、L@1e20 的冷却模型，选中的 EMA 权重 `FINAL_SEL`）直接转换，新算力也是 3e19。
- 结果：转换 2.38（XL）、3.20（L）；DMD2 学生 2.80、3.26。

### 6.11 `shortcut`：第二种一步目标（67 个任务）

- 格子：B@1e19、M@1e19、M@3e19，三个种子。f：1e19 用 {0.25, 0.5, 0.8, 0.95, 0.99}，3e19 用 {0.27, 0.5, 0.65, 0.8, 0.95, 0.99}。
- 种子 0 从 iMF 网格的 FM 主线分叉（f=0.25 来自 `r1_B_rerun2_s0`、`r1_M_rerun_s0`）；种子 1、2 每个尺寸一条新的 FM 主线 `g3fm_*`，带所有分叉点；
  原生 shortcut 主线 `t2_*` 加冷却。
- 结论：预训练在每个格子都有用，但收益更小、f* 更低（0.30–0.52）：原生差 1.06、1.14、1.33 倍（iMF 是 1.5–2.5 倍）。
  M@3e19 的收益超过样本价格能解释的 4%，说明预训练还给了 shortcut 更好的速度场去自举。

### 6.12 训练依赖顺序

```
mainline ──▶ cooldown / extend ──▶（extend 的里程碑）──▶ convert
FM cooldown（教师）+ FM 里程碑（学生初始化）──▶ distill
FM cooldown 的 final_sel.pt ──▶ sunk convert
```

`osr.experiments.plan(groups)` 会自动把依赖的祖先任务加进来并排好顺序；`launch.py list` 显示每个任务的状态
（done / ready / started / waiting）。

---

## 7. 评测协议

### 7.1 主评测 `scripts/eval_run.py`（每个冷却、转换、蒸馏模型都做；主线和延长段不评测）

| 阶段 | 做什么 |
|---|---|
| A：选权重 | raw、EMA0.05、EMA0.10，各在 3（iMF）/ 2（FM）/ 3（DMD2）个 guidance 点上，5k 样本，seed 1 |
| B：选 guidance | iMF：ω ∈ {1, 1.5, 2, 3, 4, 5.5, 7, 8.5, 10.5} × 区间 {[0,1], [0.25,0.75], [0.4,0.65]}；FM：ω ∈ {1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4} × {[0,1], [0.2,0.85]}，50 步 Euler；DMD2：同 FM 的 ω（学生输入），全区间；shortcut：不选（h=1、无 guidance）。10k 样本，seed 1 |
| C：报告 | 50k 样本，seed 0，每类 50 张，在选中的权重和 guidance 上；所有模型同样的噪声和标签（共同随机数）；保存 Inception 和 DINOv2 特征 |

- 附带：一步模型在 C 点再测 2-NFE 和不加 guidance 的 10k FID。
- 输出：`$OSR_EVAL/<run>/{results.jsonl, summary.json, inception_*.npy, dinov2_*.npy, DONE}`。

### 7.2 扩展搜索 `eval_gridext.py` 和统一打分 `eval_gridpolish.py`

- 起因：B 阶段的最优点常落在网格边上（所有多步模型都选了较窄的区间；部分一步模型选了最大尺度或最窄区间）。
- 扩展搜索：对区间做坐标搜索（先下界、再上界；一步模型步长 0.05 且区间必须包含 t=0.5，多步模型步长 0.1），每个区间三个尺度，
  边上赢就继续往外走，最后在最好区间上加大尺度；5k 样本比较，最好的两个用 10k 复核；赢了就在新点重算 FID-50k。原值存在 `grid1`。
- 统一打分：贪心搜索在噪声下会让同一格子的两个种子落到不同的盆地（宽区间 + 低尺度 vs 窄区间 + 高尺度），所以所有多步模型在
  8 个共享候选点上用 10k 样本统一打分。
- 影响：多步前沿 FID 降了 6–13%（如 1e19 从 4.61 到 4.01）；一步前沿最多动 1%。论文里所有数字（含一步税）都用扩展后的值。

### 7.3 其它指标

| 指标 | 设置 | 脚本 |
|---|---|---|
| FID-50k（主指标） | torch-fidelity 的 Inception；参考统计 = JiT/iMF 评测用的 ImageNet-256 训练集统计 `jit_in256_stats.npz` | eval_run.py |
| ADM-FID、precision / recall | ADM 参考批 `VIRTUAL_imagenet256_labeled.npz`；P/R 用 k=3，Inception pool 特征 | eval_extra.py |
| FD-DINOv2 | ViT-L/14，224 px bicubic，CLS token；参考 = 10 万张 ImageNet 训练图（每类 100 张，dgm-eval 协议） | dino_ref_stats.py、eval_extra.py |
| FD-DINOv2（自选 guidance） | 对 FD-DINOv2 单独选 guidance | eval_fdsel.py |
| P/R 曲线 | 沿 guidance 扫描，10k 样本 | eval_prcurve.py |

- ADM-FID 和我们的 FID 排名几乎一致（Spearman 0.9997，237 个模型）。

### 7.4 少步数评测 `scripts/eval_steps.py`（生成成本，论文 6.5 节）

- 对象：多步路线每个预算的前沿模型（seed 0，选中的 EMA 权重 `final_sel.pt`）：r1_B_1e19、r1_M_3e19、r1_L_1e20、r1_XL_3e20、v2_H_1e21。
- 步数 1、2、4、8、16、32；Euler，偶数步再加 Heun（区间数减半，每区间 2 次前向）。
- 每个步数在 10k 样本（seed 1）上选 guidance：不加 guidance，以及 ω ∈ {1.5, 2, 3, 4, 5} × {全区间, 50 步选中的区间}；在最好的（求解器, guidance）上算 FID-50k（seed 0）。
- 前向次数精确计数：guidance 区间内每次评估算 2 次前向。50 步的点直接取 `summary.json`。
- 输出 `$OSR_EVAL/<run>/steps.json`（`select` = 10k 选择记录，`final` = 每个步数的 FID-50k）；可断点续跑。
- 2026-11-08 用 4×H200 一个 array 任务跑完（H 最慢，约 6 小时）。

---

## 8. 统计与分析方法

所有分析在写作库 `analysis/` 里，`analysis/run_all.sh` 按依赖顺序跑完（约 30 分钟，CPU）。

### 8.1 基本对象

- **格子（cell）**：(route, size, budget, f)，route ∈ {R1 多步, NAT 原生（f=0）, CONV 转换}。
  格子的值 = 各种子 log FID-50k 的平均。学习率变体（`_lr`、`_flr`）不进格子。
- **前沿**：每条路线每个预算上最好的格子（转换同时在尺寸和 f 上取最小）。
- 加载：`analysis/frontier.py`（`load`、`frontier`、`fit_floor`、`local_parabola_f`、`fstar_by_budget`）。

### 8.2 定律和最优点

- **带下限的幂律**：`FID*(C) = E + K·(C/1e20)^(−γ)`，在 log FID 上最小二乘，约束 0 ≤ E ≤ min FID，多个初值。
- **联合一步定律**：`FID*_nat(C) = L(C)`，`FID*_conv(C) = L(m̂·C)`，E、K、γ、m̂ 对 8 个前沿点（4 预算 × 2 路线）联合拟合（`joint_law.py`）。
  变体：m 固定为每个预算最优 f 的步成本值 m(f*)（3 参数，AICc 更优，预测 1e21 为 1.89 / 1.75）。论文报告自由 m̂ 版本。
- **单独拟合的问题**：每条路线 3 个参数拟合 4 个点，下限识别不了；31% 的 bootstrap 重采样里两条单独定律在 1e25 前交叉。联合定律去掉了这个自由度
  （约束的 F 检验 p=0.88）。
- **留一预算检验**（`paper_tables.py` 的 heldout）：每次去掉一个预算重拟合并预测它；3e20 一列是正向回测。
- **f\***：在 `w = log(1−f)` 上，过最好的 f 和两个邻居拟合抛物线，`f* = 1 − exp(−β/(2α))`；不凸或最好点在端点时取网格最优。
  对照估计量：0<f<0.95 所有点的二次拟合、网格最优（`noise_model.py` 的 `fstar_estimators`）。
- **N\***：在 log N 上过最好尺寸和两个邻居的抛物线顶点（最好尺寸在边上时是界）；`N* ∝ C^a` 的 a = log N* 对 log C 的斜率。
- **一步税**：`T(C) = 最好 1-NFE FID / 最好 50-NFE FID`。

### 8.3 种子噪声模型和 bootstrap（`analysis/noise.py`、`noise_model.py`）

- 33 个有两个种子的格子（1e19、3e19；另有 16 个晚转换格子）给出单次运行 log FID 的 SD：多步 0.019、原生 0.055、转换（f<0.95）0.043、晚转换（f≥0.95）0.144。
- 同一主线分出的分支共享大部分噪声：原生两个分支的种子差相关 0.85，同一 FM 主线的两个转换 0.84；多步没测到共享成分（设为 0）。
- 模型：`log FID(run) = μ(cell) + σ_g·(λ_g·z_主线 + sqrt(1−λ_g²)·ε_run)`；σ_g² = 种子差 d 的 d²/2 的平均，λ_g² = 共享主线的不同格子种子差的合并相关。
- 每个统计量（定律、f*、N*、各种比值、预测）在 2000 次重采样上重算，取百分位区间。重采样的随机种子固定（`noise_model.py` 用 20261023），
  所以区间可精确复现。
- 其它噪声：FID-50k 本身的采样误差 1.0%（对半分估计，`fid_sample_noise.py`）；只换最后一段种子 0.1–2.5%。
- f* 趋势的检测力：趋势超过每 10 倍算力 0.28 才有 80% 的概率测出来。
- 选择偏差检查（`robustness.py`）：8 个双种子格子里，种子 0 上最好的 f 在种子 1 上也是最好的有 7 个；用种子 0 选 f、在种子 1 上量，
  优势几何平均 1.612，种子 1 自选是 1.613。

### 8.4 同样本数比较（`matched_steps.py`）

- 每个转换和同尺寸的原生模型比：一个在相同 FLOPs 下，一个在相同样本数下。
- 原生模型的 log FID 对 log S 在各预算之间线性插值（XL 包含 1e21 的原生点），在转换模型的样本数 `S_conv = fC/c_FM + (1−f)C/c_iMF` 处取值。
- 超出原生最大预算的两个格子（L、H@3e20）需要短外推，标出并排除在中位数之外。
- 比值 `r(f) = FID_nat(N; S_conv) / FID_conv` = FM 样本相对一步样本的价值；r>1 表示花在 FM 上的样本比一步样本更有用。
- 稳健性：用最近三个预算的二次插值代替直线，每个 f 的中位数降 0.01–0.04，最优 f 处 0.99 → 0.96。

### 8.5 两成分模型（`two_component.py`）

- 原生：`g_N(S) = V_N(S) + D_N(S)`，一步缺陷 `D_N(S) = log(1 + (τ_N/S)^β)`，`τ_N = τ·(N/1e8)^(−δ)`。
- 转换：`log FID_conv = V_N(λ·S_FM + S_1) + D_N(S_1)`，FM 样本只进速度项，一步样本两项都进。4 个参数：λ、τ、β、δ。
- 拟合对象：等样本数落在本尺寸原生曲线内的转换（69 个），f<0.2 的点留出来做预测。期望 rms = sqrt(σ_conv² + σ_nat²) = 0.070（f<0.95）/ 0.155（f≥0.95）。
- 结果：λ = 1.64 [1.44, 1.89]，δ = 0.71 [0.38, 1.01]；rms 0.056 / 0.120；留一预算、留一尺寸都低于噪声；两个 f≈0.1 的点预测在区间内；
  推出的 f* = 0.50–0.73。

### 8.6 其它分析

| 分析 | 方法 | 脚本 |
|---|---|---|
| 原生最优尺寸预测 | `N*_nat(C) ≈ N*_FM(C/κ) = κ^(−a)·N*_FM(C)`，a=0.56，κ 在预测尺寸处取值，迭代到不动点 | figures.py（nstar_prediction）、paper_tables.py |
| 数据重复 | 每个格子的 epoch 数；曲面残差和 log epoch 的相关 −0.18 到 −0.23；前沿模型最多 312 epoch | repetition.py |
| 分叉点诊断 | FM 分叉点的 50 步 ODE 直线度 S 和一步跳的相对误差 G；f 从 0.5 到 0.99 最多变 11% | branch_points.py（数据来自 branch_diagnostics.py） |
| 蒸馏 | 学生 FID 和转换定律在相同总算力下比较，三种记账 | distillation.py |
| FD-DINOv2 版本 | 前沿、比值、一步税用 FD-DINOv2 重算；FD 自选 guidance 下原生 / 转换 = 1.29、1.52、1.34、1.20（1e19→3e20），1e21 为 1.20 | fd_dinov2.py、stage_seeds_fd.py |
| P/R | 前沿模型 P/R；原生在小预算上的缺陷主要是 recall 低 0.02–0.14 | precision_recall.py、pr_curves.py |
| 稳健性 | 回测、下限的 profile、交叉点、FD 比值的种子区间、同尺寸比较、选择偏差 | robustness.py |

---

## 9. 结果速查：数字 → 表 / 图 → 脚本

| 论文元素 | 内容 | 生成脚本 | 中间结果 |
|---|---|---|---|
| Fig. overview | 三条路线示意 + f 曲线 | diagrams.py | noise_model.json |
| Fig. related | 相关工作地图 | diagrams.py | — |
| Fig. method | WSD 协议、保函数转换、步成本 | diagrams.py | — |
| Fig. isoflop | 每个预算的 FID-尺寸曲线 | figures.py | suite_metrics.csv |
| Table 1（t1_col） | 每条路线每个预算的最好模型 + FD/P/R | paper_tables.py | — |
| Fig. samples（附录） | 1e21 三个模型同噪声样本 | sample_figs.py | data/samples |
| Fig. landscape（附录） | FID 在尺寸 × f 上的热图 | paper_figs.py | — |
| Fig. gencost、ta_steps | FID vs 每张图生成算力、vs 训练 + 生成总算力；少步数 FID 表 | generation_cost.py | generation_cost.json、data/eval/*/steps.json |
| Fig. mechanism | 样本交换、FM 样本价值、N*、shortcut | paper_figs.py | exchange_rate.json、two_component.json、shortcut.json |
| Fig. laws | 联合定律、优势、预测 vs 实测 | paper_figs.py | joint_law.json、noise_model.json、robustness.json |
| t2_fits | 单独定律参数 + a | tables.py | t2_fits.json |
| ta_joint、ta_heldout | 联合定律、留一预算 | paper_tables.py | joint_law.json |
| ta_twocomp | 两成分模型 | two_component.py | two_component.json |
| ta_noise、ta_fstar_nc、Fig. fstar_stab | 噪声、每个格子的 f*、f* 估计量 | noise_model.py、figures.py | noise_model.json |
| ta_matched、ta_exchange、Fig. matched | 同 FLOPs / 同样本数比较 | matched_steps.py、paper_tables.py | matched_steps.json、exchange_rate.json |
| ta_nstar、ta_efficient | 最优尺寸、高效 κ | paper_tables.py | nstar_pred.json、joint_law.json |
| ta_all | 每个网格模型的 FID | paper_tables.py | — |
| ta_lr、ta_ctune | 学习率 | tables.py、conversion_lr.py | lr_sweep.json、conversion_lr.json |
| ta_init、ta_sunk | 初始化消融、FM 已有 | paper_tables.py | — |
| ta_shortcut、Fig. shortcut | shortcut f 曲线 | shortcut.py | shortcut.json |
| t6_distillation、Fig. distillation | DMD2 | tables.py、distillation.py | distillation.json |
| Fig. branch_points | 分叉点诊断 | branch_points.py | branch_points.json |
| Fig. prcurve、Fig. precision_recall | P/R | pr_curves.py、precision_recall.py | — |
| t3_compare_short、t8_compare、ta_lit_sources | 和已发表结果比较 | lit_table.py | data/lit |
| ta_arch、ta_hparams、ta_eval、ta_compute | 复现性表格 | repro_tables.py（读代码库） | data/frozen/gpu_hours.json |
| t4_repro | 管线检查 | tables.py | — |
| 正文里散落的数字 | 多半在 `analysis/out/*.json`，用 grep 找 | — | — |

---

## 10. 从零复现

### 10.1 只重建论文（不需要 GPU）

```bash
cd pretrain_then_convert_paper
pip install numpy==2.5.3 scipy==1.18.1 matplotlib==3.11.2 pillow==12.3.0
# repro_tables.py 需要 import 代码库（会 import torch）：装代码库的 requirements，或设 PTC_CODE=代码库路径
bash analysis/run_all.sh          # 重算所有表和图（约 30 分钟）
tools/build.sh                    # 编译到 build/main.pdf，并检查正文正好 8 页
python tools/make_overleaf.py     # 生成 pretrain_then_convert_overleaf.zip
```

正确的标志：生成的 29 张表和 17 个图与仓库里提交的版本逐字节一致（PDF 除了 CreationDate），`check_pages.py` 输出 `PASS`。

### 10.2 从训练开始

```bash
cd pretrain_then_convert
pip install -r requirements.txt        # Python 3.12，PyTorch 2.14.1（CUDA 13）
export OSR_WORK=/big/disk/osr           # 或分别设 OSR_LATENTS / OSR_RUNS / OSR_EVAL / OSR_REF

# 1. 数据和参考统计
torchrun --nproc_per_node 8 scripts/encode_latents.py --root /path/to/imagenet/train
wget -P $OSR_WORK/ref_stats https://raw.githubusercontent.com/LTH14/JiT/refs/heads/main/fid_stats/jit_in256_stats.npz
wget -P $OSR_WORK/ref_stats https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
torchrun --nproc_per_node 2 scripts/dino_ref_stats.py

# 2. 检查
python -m pytest -q tests
OSR_WORK=/tmp/osr_self_test python scripts/self_test.py      # 一张卡，几分钟
python scripts/validate_pipeline.py                           # 设 OSR_OFFICIAL_IMF_B 还会测官方 iMF-B/2 的 FID

# 3. 训练（按组；slurm 只提交 ready 的任务，父任务完成后再跑一次）
python scripts/launch.py list grid
python scripts/launch.py slurm grid --submit      # 先在 slurm/train.sbatch 里加账号/分区/QOS
python scripts/launch.py train r1_M_s0 --gpus 2   # 或在本机直接训练一个

# 4. 评测（每个训练完的冷却 / 转换 / 蒸馏模型）
python scripts/launch.py slurm grid --eval --submit                  # = eval_run.py
torchrun --nproc_per_node 2 scripts/eval_gridext.py --run RUN_ID     # 最优点在网格边上时
torchrun --nproc_per_node 2 scripts/eval_gridpolish.py --run RUN_ID  # 多步模型
torchrun --nproc_per_node 2 scripts/eval_extra.py --run RUN_ID       # FD-DINOv2、ADM-FID、P/R
# 可选：eval_fdsel.py（fd_dinov2_runs.txt 里的 run）、eval_prcurve.py（pr_curve_runs.txt）

# 5. 只留选中的权重（省空间）
python scripts/compact_finals.py      # 写 final_sel.pt 并逐张量核对，之后 final.pt 可删
```

### 10.3 把新结果同步进写作库

```bash
rsync -a --include='*/' --include='summary.json' --include='extra.json' --include='fdsel.json' \
      --include='prcurve.json' --include='gridext.json' --include='steps.json' --include='results.jsonl' --exclude='*' \
      $OSR_EVAL/ pretrain_then_convert_paper/data/eval/
```

`data/runs.csv` 可以由 `osr/experiments.py` 重新生成（内容和现有文件完全一致，已验证）：

```python
import csv, sys; sys.path.insert(0, "pretrain_then_convert")
from osr import experiments as X
with open("pretrain_then_convert_paper/data/runs.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["run_id", "route", "kind", "size", "seed", "budget", "f", "lr", "flops_end", "parent"])
    for rid, r in X.all_runs().items():
        p = f"{r['parent'][0]}:{r['parent'][1]}" if r.get("parent") else ""
        w.writerow([rid, r["route"], r["kind"], r["size"], r["seed"], r.get("budget", ""), r.get("f", ""),
                    "" if r["lr"] is None else r["lr"], f"{r['flops_end']:.4e}", p])
```

`data/run_epochs.json` 来自每个 run 的 `metrics.jsonl`（最后一条带 `epochs` 的记录，以及 `start` 事件里的 `flops_per_sample`）。

---

## 11. 在此基础上做新实验

1. **定义任务**：在 `osr/experiments.py` 加一个构建函数 `_<组名>(add)`，在 `GROUPS` 和 `_BUILDERS` 里登记。遵守命名规则（6.0），
   用已有主线的里程碑当父节点（需要新分叉点就加 extend 任务，像 `split` 组那样）。同一个 run id 只属于第一个定义它的组。
2. **检查**：`python -m osr.experiments <组名>` 看任务数和 FLOPs；`python scripts/launch.py list <组名>` 看依赖是否满足。
3. **训练和评测**：同 10.2；一步模型最优 guidance 落在网格边上时一定要跑 `eval_gridext.py`，多步模型要跑 `eval_gridpolish.py`，
   否则和已有结果不可比。
4. **分析**：同步数据（10.3），在 `analysis/` 写新脚本（用 `common.py` 里的路径常量，输出写到 `analysis/out/`、`paper/tables/`、`paper/figures/`），
   加进 `analysis/run_all.sh` 的正确位置（读谁的输出就放在谁后面）。图用 `figures.py` 的 `F.save()`（按印刷尺寸保存、STIX 字体、统一配色）。
5. **注意保持可比性**：同样的 batch 256、lr 1e-4（除非实验本身是调 lr）、同样的 WSD 和 15% 衰减、同样的评测协议和共同随机数、按 FLOPs 触发。
   解析 run id 的正则要精确匹配（见 13 节的坑）。
6. **论文**：改完重跑 `run_all.sh` 和 `tools/build.sh`，确认 `check_pages.py` 仍是 `PASS`（正文正好 8 页，正文的图表都在 1–8 页）。
   大改之前先把 `build/main.pdf` 存一份旧版本。

---

## 12. 算力和时间

- 全部：约 17.5k GPU 小时训练 + 1.3k 评测（H100 80GB，最后一批在 H200；包括失败、被抢占和重复的作业）。
- 每卡吞吐（算法 FLOPs）：FM 214–331 TFLOP/s，iMF 173–292 TFLOP/s（`data/throughput.json`）。
- 时间估算：`小时 ≈ 这个 run 自己的 FLOPs / (GPU 数 × 每卡吞吐) / 3600`。例：
  - XL 的 FM 主线到 2.97e20（8 卡，285 TFLOP/s）≈ 36 小时；
  - 1e21 的原生 XL 主线到 8.5e20（8 卡，243 TFLOP/s）≈ 5 天；
  - B@1e19 的转换（f=0.5，自己 5e18，1 卡，228 TFLOP/s）≈ 6 小时。
- 一次主评测：2 卡，一般在 4 小时以内（`launch.py` 给评测作业 4 小时时限；多步模型 50 步采样更慢）。
- 存储：latent 40 GB；评测时 inception/dinov2 特征每个 run 几百 MB；checkpoint 压缩成 final_sel.pt 后全部约 110 GB。

---

## 13. 踩过的坑

**训练实现**
- PyTorch 的 forward-mode AD 不能 compile，eager 又慢又占内存（旧实现只有 52–100 TFLOP/s，L/XL 会 OOM）。改用 `osr/jvp.py` 手写切向规则，
  primal 和 tangent 拼 batch 共享一次 GEMM，再加 compile 的 flash 前后向，iMF 到 170–290 TFLOP/s。JVP 的数值对 forward-AD 校验到 1e-6（tests）。
- forward-AD 对每个线性层会多跑一次 `x @ 0` 的无用矩阵乘；FLOP 计数要避开（我们的计数用单遍参考实现）。
- 续训 / 分叉时把整个 checkpoint 加载到 GPU，H 尺寸多占 27 GB 导致 OOM 循环；改为先加载到 CPU。
- XL 学生 + XL fake score + XL 教师放 4 卡 OOM，要 8 卡。
- 两节点 DDP 在 batch 256 下扩展很差：所有 run 单节点。
- SciPy 1.18 去掉了 `sqrtm(disp=False)`；Fréchet 距离改用 eigh。

**评测**
- 最优 guidance 落在网格边上很常见（尤其多步模型），不做扩展搜索会严重高估多步 FID（6–13%），一步税被夸大。
- 贪心坐标搜索在噪声下会让同一格子的不同种子落到不同盆地：多步模型要在共享候选点上统一打分。
- FD-DINOv2 在 FID 选出的 guidance 上测会有偏，所以另做了 FD 自选 guidance 的评测。

**分析**
- 解析 run id 的正则必须精确：曾有宽松的匹配把 `_flr` 学习率变体当成普通转换混进 f 曲线。
- 对加噪的格子取最小值（前沿）会让 bootstrap 的前沿系统性偏低；区间解读时要注意，联合定律和留一检验更可靠。
- 单独的带下限定律 3 参数配 4 个点，下限识别不了，外推会交叉；用联合定律。
- matplotlib constrained layout 在第二次绘制时才收敛：`F.save()` 先往内存里画一次 PNG 再存 PDF，否则版面会变。

**论文版面**
- `figure*` 会出现在源码位置的下一页；正文正好 8 页靠把次要图表挪到附录、填满第 2、7、8 页的空隙。
- 只看「Conclusion」从哪页开始会漏掉溢出到第 9 页的几行：用 `tools/check_pages.py`（`\markendmain` 把正文结束位置写进 .aux）。

**作业管理**
- 不要在 bash 脚本运行时修改它（bash 按字节偏移读脚本）；要改就复制成新文件。
- 不要用 `pkill -f <模式>`，如果模式也出现在当前命令行里会杀掉自己的 shell；按 PID 杀。
- squeue 偶尔返回空：自动提交器要在 squeue 出错或缺少「最近还在写 metrics.jsonl 的 run」时跳过这一轮，否则会重复提交在跑的任务。
  被 SIGTERM 的重复作业会先存一次 latest.pt。
- 长时间的监控或自动提交不要放在交互会话里，终端断开就没了；用脱离终端的后台进程或 SLURM 作业。
