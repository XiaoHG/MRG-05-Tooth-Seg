# 组会报告大纲（审核稿）

- **报告题目**：OralDINO：多源口腔影像的 DINOv3 域内预训练与下游验证
- **副标题**：组会进度汇报
- **日期**：2026-11-11
- **汇报人**：【待填】
- **建议时长**：20 分钟汇报 + 10 分钟讨论
- **建议页数**：20 页正文 + 3–5 页 backup
- **状态**：大纲已审核；PPT 已生成（20 页）

> 本目录只用于组会报告，不修改项目原有内容。
> 所有数字的来源见同一目录下的 `SOURCES.md`；汇报前请按最新训练日志刷新。
>
> **审核后调整**：最终 PPT 不包含论文 B / 单牙 token 内容；创新点改为
> 「数据层 / 预训练层 / 评估层」，原 Slide 8–9 已删除；新增论文 A 摘要页
> （第 20 页），最终 21 页。其余页与数字口径不变。

---

## 0. 汇报定位（一句话）

本报告回答三个问题：**为什么做 OralDINO；怎么做（创新点 / 架构 / 实验）；现在做到哪、下一步怎么走。**

主线叙事：

```text
检测基线已收口（8 模型 + 人工复核）
        ↓
P0 域内预训练失败并完成归因（loss 下降 ≠ 下游提升）
        ↓
改用官方 DINOv3 recipe + 小 batch softmax centering
        ↓
Stage 0 诊断通过，Stage 1 训练中（44%）
        ↓
下一步：续训 → 导出 backbone → 下游 probe → 论文 A
```

---

## 1. 页面总览

| 页 | 标题 | 类型 | 建议时长 | 核心信息 |
| ---: | --- | --- | ---: | --- |
| 1 | 封面 | 固定 | 0.5 min | 题目、姓名、日期、组会 |
| 2 | 汇报提纲 | 固定 | 0.5 min | 创新点 / 进度 / 计划三部分 |
| 3 | 研究背景：口腔筛查的可及性与标注瓶颈 | 背景 | 1.5 min | 为什么要做域内预训练 |
| 4 | 相关工作与空白 | 背景 | 1.5 min | DINOv3、DinoDental、AlphaDent、COde、TLNM、Caries-DETR |
| 5 | 研究问题与总体技术路线 | 背景 | 1 min | RQ1–RQ3 + 一张路线图 |
| 6 | 创新点总览 | 创新点 | 1 min | 3 个创新点，主推 1+2 |
| 7 | 创新点一：OralDINO 域内预训练 | 创新点 | 2 min | 多源数据 + 官方 DINOv3 + 小 batch 适配 |
| 8 | 创新点二：单牙 token + 牙间拓扑建模 | 创新点 | 2 min | 结构感知，抑制误检 / 服务 FDI |
| 9 | 创新点三：氟斑牙病灶级分级 | 创新点 | 1 min | 下游验证 / 论文 B |
| 10 | 模型架构总图 | 架构 | 1.5 min | 预训练 + 三条下游路径 |
| 11 | 预训练关键设计 | 架构 | 1.5 min | softmax centering、crops、loss、prototype |
| 12 | 数据与评估协议 | 实验设计 | 1.5 min | 42,616 唯一图、154 张氟斑牙、828 固定 test |
| 13 | 实验矩阵 | 实验设计 | 1.5 min | E1–E10 与状态 |
| 14 | 进度一：检测基线 8 模型 | 进度 | 2 min | before/after 全部提升；RT-DETRv2-L 最优 |
| 15 | 进度二：P0 失败复盘 | 进度 | 2 min | QWK 0.315 vs base 0.718；原因与教训 |
| 16 | 进度三：官方 DINOv3 Stage 0/1 | 进度 | 2 min | 诊断通过；Stage 1 44%；loss 健康 |
| 17 | 当前问题与风险 | 问题 | 1.5 min | SIGKILL、GPU 争用、K/N、数据质量 |
| 18 | 下一步计划 | 计划 | 1.5 min | 本周 / 两周 / 一个月 |
| 19 | 论文撰写安排 | 论文 | 1.5 min | 论文 A 结构、时间线；论文 B 暂缓 |
| 20 | 总结与讨论 | 固定 | 1 min | 3 点结论 + 3 个请教问题 |
| B1 | Backup：Stage 1 完整 loss 表 | backup | — | 按 500 步统计 |
| B2 | Backup：数据审计明细 | backup | — | 来源、重复、分辨率 |
| B3 | Backup：关键配置 | backup | — | Stage 0/1/2 参数 |
| B4 | Backup：检测完整指标 | backup | — | 8 模型 before/after |
| B5 | Backup：P0 特征几何 | backup | — | pair cosine、稳定性、权重漂移 |

---

## 2. 逐页大纲

### Slide 1｜封面

- 题目：OralDINO：多源口腔影像的 DINOv3 域内预训练与下游验证
- 副标题：组会进度汇报
- 汇报人 / 导师 / 日期 / 单位
- 建议配图：口腔照片 + DINOv3 架构的简洁拼图（可选）

### Slide 2｜汇报提纲

1. **创新点与实验设计**：研究问题、创新点、模型架构、实验矩阵
2. **实验进度**：检测基线、P0 复盘、官方 DINOv3 Stage 0/1
3. **下一步计划**：续训、下游评估、论文撰写

讲稿提示：先给结论——“检测基线已收口，预训练路线已完成纠偏，当前 Stage 1 训练中”。

### Slide 3｜研究背景：口腔筛查的可及性与标注瓶颈

- 口腔疾病（龋病、氟斑牙、牙周病等）筛查高度依赖专业医生，基层与偏远地区可及性不足。
- 智能手机口内照片是低成本入口；但真实场景存在高光、金属、遮挡、视角变化。
- 下游任务链：**牙齿检测 → 实例分割 / FDI 牙位 → 龋病 / 氟斑牙分级**。
- 标注瓶颈：氟斑牙 5 级目前仅 **154 张**；检测框标注成本高，跨数据集标注标准不一致。
- 自监督预训练可利用无标注口内图像，但通用 DINOv3 与口腔影像存在明显域鸿沟。

建议图表：口腔筛查场景图 + “检测→分割→FDI→分级”任务链示意。

### Slide 4｜相关工作与空白

- **DINOv3**：自监督 ViT，具备高质量 dense token；自然图像 LVD-1689M 预训练。
- **DinoDental**：系统评估自然图像 DINOv3 在牙科任务上的迁移，但**不做牙科域内预训练**。
- **AlphaDent**：高分辨率口内照片实例分割；**COde**：5 万张口内照片 + 诊疗文本；**OMNI**：4,166 张多视角错颌图像。
- **TLNM / Mask R-CNN**：牙齿检测 + FDI 编号；**Caries-DETR**：结构感知查询初始化。
- 空白：
  1. 多源口内数据整合后做 DINOv3 域内预训练的公开工作很少；
  2. 现有检测器逐框独立预测，未显式建模牙弓/邻牙拓扑；
  3. 氟斑牙多为整图分级，缺少病灶级方法。

建议图表：一张“工作对比矩阵”（任务 × 是否域内预训练 × 是否结构建模），突出空白。

### Slide 5｜研究问题与总体技术路线

- **RQ1**：多源口内图像能否通过 DINOv3 continue-pretraining 得到可迁移的 OralDINO backbone？
- **RQ2**：单牙实例 token + 牙间拓扑建模能否降低困难子集误检/漏检，并服务 FDI？
- **RQ3**：OralDINO 能否在氟斑牙分级与牙齿检测上相对 base DINOv3 取得提升？
- 技术路线：

```text
多源公开口内数据
   → 数据审计 / SHA256 去重
   → OralDINO（DINOv3 ViT-B/16 域内预训练）
   → 下游：冻结 probe / LoRA / 全量微调 / 检测 A3 / 单牙 token
```

建议图表：横向三阶段路线图（数据 → 预训练 → 下游）。

### Slide 6｜创新点总览

| 创新点 | 定位 | 一句话 |
| --- | --- | --- |
| 创新点一 | 主线 | 多源口内数据 + 官方 DINOv3 域内预训练，构建 OralDINO |
| 创新点二 | 方法 | 单牙实例 token + 牙弓位置编码 + 牙间 Transformer，建模结构先验 |
| 创新点三 | 下游 / 论文 B | 先分割单牙再合成，做病灶级氟斑牙分级 |

- 取舍：主推 **1 + 2**；创新点 3 作为下游验证，不在一篇论文里全部叠加。
- 工程落地（手机 App / 云服务 / 牙齿向量库）放在“应用前景”，不作为学术创新点。

### Slide 7｜创新点一：OralDINO 域内预训练

- **问题**：自然图像 DINOv3 与口内照片存在域鸿沟；牙科数据分散、单库体量有限。
- **方法**：
  - 整合 AlphaDent、COde、OMNI 等多源数据，SHA256 去重后 **42,616 张唯一图**；
  - 以官方 DINOv3（iBOT + KoLeo，ViT-B/16）为基座做 continue-pretraining；
  - 小 batch 下 Sinkhorn 会退化为均匀目标，改用 **softmax + center**；
  - 两阶段：224/96 主训练 → 512/128 高分辨率适应。
- **差异**：DinoDental 只评估自然 DINOv3 的直接迁移；本工作做域内预训练并验证下游。
- **验证**：冻结线性 probe（QWK）、LoRA/全量微调、检测 A3。

讲稿提示：这里要讲清“为什么不是直接拿 base DINOv3 用”——P0 的教训在 Slide 15。

### Slide 8｜创新点二：单牙实例 token + 牙间拓扑建模

- **问题**：逐框独立检测导致
  - 白色高光、金属固定器、口唇被误检为牙齿；
  - 门牙等明显区域漏检；
  - 牙位之间缺少“相邻有序”的约束。
- **方法**：
  1. 检测器输出每颗牙 ROI；
  2. DINOv3 编码器把每颗牙编码为**单个实例 token**（不是像素 patch）；
  3. 加入牙弓坐标/左右镜像位置编码；
  4. 上层 Transformer 建模牙 ↔ 牙邻接与顺序；
  5. 输出置信度校正、FDI 编号、后续病灶分级。
- **差异**：TLNM/Mask R-CNN 以逐实例分类为主；Caries-DETR 侧重查询初始化；本点显式建模牙弓拓扑。
- **验证**：困难子集每图 FP/FN、FDI 32 类准确率。

建议图表：单牙 ROI → token 序列 → 牙间注意力的示意图。

### Slide 9｜创新点三：氟斑牙病灶级分级（下游 / 论文 B）

- **问题**：现有方法多为“整图 → 5 级”，轻度病例边界细微，整图噪声大。
- **方法**：单牙分割/裁剪 → 拼接合成去背景 → 病灶块级检测 + 邻牙上下文 → 严重程度分级。
- **差异**：FusionDentNet、MLTrMR 等以整图分级为主；本点强调单牙病灶定位 + 邻牙校正。
- **定位**：作为论文 A 的下游验证任务；若数据补足，再独立发展为论文 B 的架构创新。
- **前置条件**：氟斑牙样本量、像素级/病灶级标注。

### Slide 10｜模型架构总图

一张图讲清三条下游路径：

```text
                  ┌─ 冻结线性 probe ──→ 氟斑牙 5 级
多源口内图 → OralDINO ─┼─ LoRA / 全量微调 ─→ 氟斑牙 / 口腔疾病分类
                  └─ 检测 A3（RT-DETR/YOLO backbone）→ 牙齿检测
                            └─ 单牙 token + 牙间 Transformer → FDI / 分级
```

- 预训练 backbone：ViT-B/16，768 维，4 storage tokens，RoPE。
- 下游统一用冻结/LoRA/全量三种适配策略对比。
- 检测 A3 作为论文 A 的主下游之一；氟斑牙作为第二下游。

建议图表：架构总图（本页是全篇最重要的一张图）。

### Slide 11｜预训练关键设计

- Student / teacher EMA；2 个 global crop（224）+ 6 个 local crop（96）。
- 损失：DINO CLS + iBOT patch + KoLeo；Gram 暂时关闭（EMA 版本近似空转）。
- **softmax + center**：batch < 512 时 Sinkhorn 的 teacher 目标退化为均匀分布，
  `loss ≈ ln K`；改用 DINOv2 风格 softmax+center 后 Stage 0 明显下降。
- prototype：**32768**（K/N ≈ 0.77，存在记忆风险，需 probe 验证）。
- Stage 1：batch 320，120 epochs，16,080 steps，峰值 LR 1.21e-4，实测 21.6 GB。
- Stage 2：512/128，3,216 steps，低 LR 适应。

建议图表：预训练 pipeline 图 + 一张 loss 曲线（Stage 0）。

### Slide 12｜数据与评估协议

- **预训练数据**：55,559 个文件 → **42,618 个唯一 SHA256**；12,941 个重复（主要在 COde）。
  - AlphaDent 1,455（唯一 1,441）；COde 49,938（37,020）；OMNI 4,166（4,157）。
  - 2 张截断 JPEG 已删除；97.3% 的图短边 < 384，中位 448×298。
- **氟斑牙数据**：154 张，5 级：normal 16 / doubtful 35 / mild 50 / medium 36 / severe 17。
- **检测数据**：OMNI_COCO before/after，固定 test **828 张**，train/val 2,481/857。
- **评估协议**：
  - 分类：冻结 CLS + 标准化 + Logistic Regression，5 折 × 5 次重复，报告 Acc / Macro-F1 / QWK；
  - 适配：LoRA / 全量微调；
  - 检测：mAP50 / mAP50-95，固定 test，不参与调参。

建议图表：数据来源堆叠柱状图 + 氟斑牙类别分布。

### Slide 13｜实验矩阵

| 编号 | 实验 | 数据 / 模型 | 指标 | 状态 |
| --- | --- | --- | --- | --- |
| E1 | 检测基线 8 模型 | OMNI before/after, 828 test | mAP50 / mAP50-95 | ✅ 完成 |
| E2 | P0 42k 预训练 | 42k, batch 8, 512 | QWK / 特征几何 | ❌ 失败，已归因 |
| E3 | Stage 0 诊断 | 224/96, batch 96, 2,220 步 | DINO/iBOT loss | ✅ 通过 |
| E4 | Stage 1 主训练 | 224/96, batch 320, 16,080 步 | loss / probe | 🔄 44% |
| E5 | Stage 2 高分辨率 | 512/128, batch 24, 3,216 步 | loss / probe | ⏳ 待做 |
| E6 | 冻结线性 probe | 7k/10k/16k checkpoints | QWK / F1 / Acc | ⏳ 待做 |
| E7 | LoRA / 全量微调 | OralDINO vs base | QWK / F1 | ⏳ 待做 |
| E8 | 检测 A3 | RT-DETR/YOLO + backbone | mAP50 / mAP50-95 | ⏳ 待做 |
| E9 | 消融 | K / batch / KoLeo / 分辨率 / Gram | QWK / mAP | ⏳ 待做 |
| E10 | 单牙 token + 牙间 Transformer | 检测 ROI | FDI / FP/FN | ⏳ 未来 |

讲稿提示：强调“先做 E6，再决定 E7/E8 的投入”，避免在无效 backbone 上做大量下游。

### Slide 14｜进度一：检测基线 8 模型（已完成）

固定 828 张 test，after 标签下：

| 模型 | mAP50 | mAP50-95 |
| --- | ---: | ---: |
| RT-DETRv2-L | 0.8749 | **0.7137** |
| Faster R-CNN | 0.8250 | 0.6583 |
| RT-DETR-L | 0.9845 | 0.6372 |
| YOLO26m | **0.9867** | 0.6333 |
| YOLO11n | 0.9855 | 0.6324 |
| YOLOv8m | 0.9857 | 0.6324 |
| YOLO11s | 0.9852 | 0.6320 |
| RetinaNet | 0.7191 | 0.5715 |

- 人工复核后，**8 个模型 mAP50-95 全部提升**，Δ +0.022 ~ +0.044（相对 +4.0% ~ +6.6%）。
- 收益集中在严格定位质量（mAP50-95），不是粗检出（mAP50 已饱和）。
- RT-DETRv2-L 对标注质量最敏感：mAP50-95 +0.044，Recall +0.062。
- 结论：检测基线可复现、排名稳定；后续 A3 用 RT-DETRv2-L / YOLO26m 作为主基线。

建议图表：after 指标横向柱状图 + before/after Δ 表。

### Slide 15｜进度二：P0 失败复盘（重要负结果）

P0：42k、batch 8、512、KoLeo 0.1、Sinkhorn。

| 指标 | base DINOv3 | P0 10k | P0 final |
| --- | ---: | ---: | ---: |
| 氟斑牙 QWK | **0.718** | 0.315 | 0.143 |
| 公开 6 类 acc | **0.820** | 0.613 | 0.574 |
| pair cosine | 0.83 | 0.039 | 0.030 |
| grayscale 余弦 | 0.959 | 0.301 | 0.262 |

- 现象：**DINO loss 从 9.4 降到 ~1.0，下游 QWK 却从 0.315 掉到 0.143**。
- 原因：
  1. 目标错配：DINO 优化的是裁剪/增强一致性，不是下游类别边界；
  2. KoLeo 0.1 过强，特征近正交（pair cosine 0.03）；
  3. 对灰度/颜色/模糊极不稳定；
  4. 过训练，最佳 checkpoint 在 5k–10k；
  5. 窄域小数据全参数继续预训练，灾难性遗忘。
- 教训：**loss 不能用于选 checkpoint，必须用下游 probe 早停**；这也是改用官方 recipe + softmax 的直接原因。

讲稿提示：这页建议保留。负结果把“为什么现在的方案更合理”讲清楚了，也体现问题定位能力。

### Slide 16｜进度三：官方 DINOv3 Stage 0/1（进行中）

**Stage 0 诊断（已完成）**

- 224/96，batch 96，2,220 步，23:35，7.9 GB；
- DINO loss：10.397（均匀基线）→ **9.68**；iBOT 5.00；KoLeo -0.28；
- 结论：softmax centering 生效，可以进入 Stage 1。

**Stage 1 主训练（进行中，44%）**

- 7,080 / 16,080 步；batch 320；21.6 GB；2.6 s/it（GPU 争用，探针为 1.31 s/it）。
- 最新：DINO local/global **0.78 / 0.69**，iBOT **2.48**，KoLeo -0.25；
- 梯度健康：backbone 1.83，DINO head 0.21，iBOT head 0.15（仍在上升）。
- 16:21:41 被 SIGKILL（exitcode -9）；ckpt/6999 完整，可从 6,999 续训。
- 判断：训练曲线健康，低 loss 是否等于好特征仍需 frozen probe 验证。

建议图表：DINO/iBOT/KoLeo 三条曲线 + 梯度曲线；标注 Stage 0 / Stage 1 / 中断点。

### Slide 17｜当前问题与风险

| 问题 | 现状 | 影响 | 处理 |
| --- | --- | --- | --- |
| SIGKILL | 16:21:41，exitcode -9 | 训练中断 | 查 dmesg/cgroup；从 6999 续训 |
| GPU 争用 | 探针 1.31 s/it → 实际 2.6 s/it | ETA 翻倍 | 协调 GPU；监控 nvidia-smi |
| K/N = 0.77 | 32768 prototypes / 42,616 图 | 头部记忆风险 | 7k/10k/16k probe；必要时降 K |
| 数据重复 | 12,941 个重复文件 | epoch 边界偏 | 已建去重视图 |
| 分辨率低 | 97.3% 短边 < 384 | 512 阶段多为上采样 | 224 主训练 + 512 短适应 |
| DCP → HF | 官方 checkpoint 不能直接 probe | 下游评估受阻 | 补导出脚本 |
| 数据合规 | 多源公开数据 license 不同 | 发表/开源风险 | 逐数据集核对 license |

### Slide 18｜下一步计划

| 时间 | 任务 | 交付物 |
| --- | --- | --- |
| 本周 | 排查 SIGKILL；从 6,999 续训 Stage 1 | 续训日志；loss 曲线 |
| 本周 | 写 DCP → HF backbone 导出脚本 | 可 probe 的 backbone |
| 下周 | 完成 Stage 1；跑 Stage 2 | 16k 最终 checkpoint |
| 下周 | 7k/10k/16k 冻结 linear probe | QWK/F1 表；选最佳 checkpoint |
| 2 周内 | LoRA / 全量微调对比 base | 下游适配表 |
| 2–4 周 | 检测 A3（RT-DETRv2-L / YOLO26m） | mAP 对比 |
| 4 周内 | 消融：K、batch、KoLeo、分辨率、Gram | 消融表 |
| 1–2 月 | 论文 A 初稿 | 完整 draft |
| 后续 | 论文 B（氟斑牙架构） | 待 A 的 backbone 与数据 |

**关键 gate**：如果 7k/10k/16k probe 都不优于 base DINOv3，暂停下游大投入，先分析数据/配方，而不是继续堆训练。

### Slide 19｜论文撰写安排

**论文 A（主线，低风险）**

- 暂定题目：*OralDINO: Domain-Adaptive DINOv3 Pretraining for Intraoral Images*
- 目标会议/期刊：【待确认，建议 MICCAI / ISBI / Medical Image Analysis 之一】
- 章节：
  1. Introduction：域鸿沟、数据分散、标注瓶颈；
  2. Related Work：DINOv3、牙科基础模型、牙齿检测；
  3. Method：数据审计与去重、continue-pretraining、小 batch softmax centering、下游适配；
  4. Experiments：linear probe、LoRA/full、检测 A3、消融；
  5. Discussion：负结果（P0）与适用边界；
  6. Conclusion。
- 图表计划：框架图、数据组成、loss 曲线、probe 表、检测表、消融表、可视化。
- 时间线（建议）：11 月完成预训练与 probe；12 月下游与消融；1 月成稿；2 月投稿。

**论文 B（暂缓，高风险）**

- 题目方向：氟斑牙病灶级分级架构（ATD + LGRF）。
- 前置：论文 A 的 backbone + 氟斑牙数据补足 + 与 MLTrMR/FusionDentNet 对齐的评估协议。
- 不在一篇论文里叠加全部模块与全部下游任务。

讲稿提示：主动说明“先 A 后 B”的取舍，避免被问“为什么不做端到端大系统”。

### Slide 20｜总结与讨论

**三点结论**

1. 检测基线已收口：8 模型 + 828 固定 test + 人工复核，结论可复现。
2. P0 失败已归因：loss 下降不等于下游提升；官方 recipe + softmax centering 是目前更合理的路线。
3. Stage 0 通过、Stage 1 训练健康（44%）；下一步用 probe 而不是 loss 选 checkpoint。

**三个请教问题**

1. 论文 A 的目标会议/期刊与时间线是否合适？
2. 下游优先级：先做氟斑牙分级，还是先做检测 A3？
3. 氟斑牙数据是否继续扩量/补病灶级标注，还是先用现有 154 张做方法验证？

---

## 3. 待确认事项（审核时请重点看）

- [ok] 汇报时长：20 分钟是否合适？是否需要压到 15 分钟？
- [待定] 目标会议/期刊：MICCAI / ISBI / MedIA / 其他？
- [氟斑牙分级，也要做其他牙齿疾病检测] 论文 A 的主下游：检测 A3 vs 氟斑牙分级，哪个作为第一主任务？
- [显示] 是否在组会展示 P0 失败细节？（建议展示）
- [不包含] 是否包含论文 B / 单牙 token 的内容？还是只作为“未来工作”一页？
- [显示] 是否展示具体数据来源与 license？
- [不需要] 汇报人姓名、导师姓名、单位、日期。
- [不需要] 是否需要英文版 slides / 英文术语表。
- [不需要] 是否需要我把大纲扩成逐页讲稿（speaker notes）。

---

## 4. 图表清单（生成 PPT 前准备）

| 图 | 内容 | 数据来源 |
| --- | --- | --- |
| Fig 1 | 口腔筛查任务链 / 研究动机 | 自绘 |
| Fig 2 | 相关工作对比矩阵 | Slide 4 |
| Fig 3 | 总体技术路线 | Slide 5 |
| Fig 4 | 预训练 pipeline（crops + losses） | Slide 11 |
| Fig 5 | 下游架构总图 | Slide 10 |
| Fig 6 | 数据来源与类别分布 | `lab_train/output/data-audit/summary.md` |
| Fig 7 | Stage 0/1 loss + 梯度曲线 | `lab_train/output/official-dental-vitb-224/training_metrics.json` |
| Fig 8 | 检测 8 模型 before/after | `output/val-test/comparison.md` |
| Fig 9 | P0 vs base QWK / pair cosine | `output/pretrain/WHY_WORSE_THAN_BASE.md` |
| Fig 10 | 下一步时间线（Gantt） | Slide 18 |
| Fig 11 | 风险矩阵 | Slide 17 |

---

## 5. 页数与时间建议

- **15 分钟版**：删掉 Slide 9（创新点三）、Slide 13（实验矩阵）、Slide 17（风险）合并到 Slide 16；
  保留 1–8、10–12、14–16、18–20，共 16 页。
- **20 分钟版**：按本大纲 20 页。
- **30 分钟版**：增加 backup B1–B5，并展开 Slide 7/8 的方法细节。

---

## 6. 审核后下一步

1. 你确认大纲、页数、目标会议、是否保留 P0/论文 B；
2. 我按确认后的结构生成 PPT 内容（每页标题 + bullets + 图表占位 + speaker notes）；
3. 再根据你的意见做一版视觉稿（配色、字体、图表）。
