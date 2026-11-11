# MRG-05-Tooth-Seg 项目现状分析（2026-11-11）

> 本文档汇总当前数据、检测基线、DINOv3 域内预训练、Stage 1 checkpoint 评估与下一步决策。
> 所有数字来自项目现有日志、probe 输出与文档；口径以文中标注为准。
> 本文档只做分析，不修改任何训练代码或数据。

---

## 0. 一句话结论

**Stage 1 把 P0 的灾难性退化救了回来，但 5k/10k/15k/16k 全部低于 base DINOv3；特征过度分散在 5k 就已经发生，早停假设被否定。下一步不是继续训练，而是先用 LoRA/全量微调判断 backbone 是否还有价值，再决定改配方还是换路线。**

---

## 1. 项目定位与阶段

- 长期目标：口腔牙齿检测 / 分割 / FDI / 氟斑牙分级，最终服务口腔疾病筛查。
- 当前主线：论文 A —— DINOv3 口内图像域内预训练（OralDINO）+ 下游有效性评估。
- 阶段：A0（数据/基线）和 A1（数据整合）已完成；A2（预训练）Stage 1 完成；A3（下游有效性）已完成 frozen probe，LoRA/全量、检测 A3 未做；A4（撰写）未开始。

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| A0 | 冻结数据/评估协议、复现下游基线 | 完成 |
| A1 | 多源牙科数据整合、审计、去重 | 完成 |
| A2 | DINOv3 域内预训练 | Stage 0/1 完成；Stage 2 未跑 |
| A3 | 下游有效性评估 | frozen probe 完成；LoRA/全量、检测 A3 未做 |
| A4 | 论文撰写与投稿 | 未开始 |

---

## 2. 数据现状

### 2.1 预训练数据

| 项 | 值 |
| --- | ---: |
| 扫描文件 | 55,559 |
| 唯一 SHA256 | 42,618 |
| 重复文件 | 12,941（9,721 组） |
| 严格解码失败 | 2（AlphaDent 截断 JPEG，已删除） |
| 与评测集哈希重叠 | 0 |
| 中位分辨率 | 448×298 |
| 短边 < 384 | 54,057 / 55,559 ≈ 97.3% |

来源：

| 来源 | 文件 | 唯一 |
| --- | ---: | ---: |
| AlphaDent | 1,455 | 1,441 |
| COde-Dataset | 49,938 | 37,020 |
| omni_coco | 4,166 | 4,157 |

问题：

- **COde 占 87%，且分辨率低**；AlphaDent 只有约 1.4k 高分辨率图。
- 97.3% 的图短边 < 384，512 训练大多属于上采样。
- 重复率高，已用 `build_dedup_view.py` 建 42,616 张去重视图。

### 2.2 下游评估数据

| 用途 | 数据 | 规模 |
| --- | --- | ---: |
| 氟斑牙 4 级 | `dataset/dental_fluorosis/images` | 200（normal/mild/moderate/severe 各 50） |
| 口腔疾病 6 类 | `nsr51324/Oral_Diseases` 平衡子集 | 720（6 类各 120） |
| 牙齿检测 | OMNI_COCO before/after | 828 固定 test |

> `dataset/positive` 是旧的 141 张 5 目录版本，已不作为主评估集；P0 的 QWK 0.718/0.315 是旧口径。

---

## 3. 检测基线（已完成）

固定 828 张 test，after 标签：

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

- 人工复核后，8 个模型 mAP50-95 全部提升，Δ +0.0224 ~ +0.0440；
- 收益集中在严格定位质量，mAP50 已饱和；
- RT-DETRv2-L 对标注质量最敏感：mAP50-95 +0.0440，Recall +0.0618；
- 结论：检测基线可复现，后续检测 A3 用 RT-DETRv2-L / YOLO26m。

---

## 4. 预训练历程

### 4.1 P0（失败，已归因）

配置：42k、batch 8、512、KoLeo 0.1、Sinkhorn。

| 指标 | base DINOv3 | P0 10k | P0 final |
| --- | ---: | ---: | ---: |
| 氟斑牙 QWK | **0.718** | 0.315 | 0.143 |
| 公开 6 类 acc | **0.820** | 0.613 | 0.574 |
| pair cosine | 0.83 | 0.039 | 0.030 |
| grayscale 余弦 | 0.959 | 0.301 | 0.262 |

原因：

- Sinkhorn 在 batch 8 下目标退化为均匀分布；
- KoLeo 0.1 过强，特征近正交；
- 对灰度/颜色/模糊极不稳定；
- 过训练 + 窄域小数据全参数继续预训练 → 灾难性遗忘。

### 4.2 Stage 0 诊断（通过）

- 224/96，batch 96，2,220 步，23:35，7.9 GB；
- DINO loss：10.40 → **9.68**；iBOT 5.00；KoLeo -0.28；
- 结论：softmax+center 生效。

### 4.3 Stage 1 主训练（完成）

| 项 | 值 |
| --- | ---: |
| 分辨率 | 224/96 |
| batch | 320 |
| epochs | 120 |
| 总步数 | 16,080 |
| 峰值 LR | 1.21e-4 |
| 显存峰值 | 21,596 MB |
| 最终 checkpoint | `ckpt/15999` |
| 最终 DINO local/global | 0.345 / 0.263 |
| 最终 iBOT | 2.352 |
| 最终 KoLeo | -0.252 |

运行情况：

- 从 11-10 11:36 到 11-11 12:04，多次中断/续训（0 → 7000 → 8500 → 15500）；
- 大部分区间约 2.6 s/it，最后 GPU 空出来后约 1.3–1.5 s/it；
- wall-clock 约 24.5 小时，包含中断和 overnight gap；
- Stage 2（512/128）未跑。

---

## 5. Stage 1 checkpoint 评估

### 5.1 评估协议

- 特征：冻结 backbone，取 CLS token，224 输入，ImageNet 归一化；
- 分类器：StandardScaler + LogisticRegression（C=0.1，class_weight='balanced'）；
- 5 折分层交叉验证 × 5 次重复（25 个 fold）；
- 指标：Accuracy / Macro-F1 / QWK；
- 对照：base DINOv3 official weights，用完全相同协议；
- checkpoint：5k（4999_keep）、10k（9999_keep）、15k（14999_keep）、16k（15999）。

### 5.2 氟斑牙 4 级（200 张）

| checkpoint | teacher Acc | teacher F1 | teacher QWK | student QWK |
| --- | ---: | ---: | ---: | ---: |
| base | **0.799** | **0.798** | **0.916** | — |
| 5k | 0.744 | 0.740 | 0.847 | 0.844 |
| 10k | 0.767 | 0.764 | 0.872 | 0.869 |
| 15k | 0.780 | 0.778 | 0.886 | 0.885 |
| 16k | 0.779 | 0.777 | 0.884 | 0.889 |

- 趋势：**5k < 10k < 15k ≈ 16k < base**；
- 5k 是四个 checkpoint 里最差的；
- 15k/16k 是当前最佳，但仍比 base 低约 0.03 QWK。

### 5.3 口腔疾病 6 类（720 张）

| checkpoint | teacher Acc | teacher F1 | teacher QWK | student QWK |
| --- | ---: | ---: | ---: | ---: |
| base | **0.828** | **0.827** | **0.848** | — |
| 5k | 0.795 | 0.795 | 0.808 | 0.805 |
| 10k | 0.799 | 0.799 | 0.819 | 0.815 |
| 15k | 0.799 | 0.799 | 0.820 | 0.822 |
| 16k | 0.799 | 0.798 | 0.820 | 0.822 |

- 趋势：**5k < 10k ≈ 15k ≈ 16k < base**；
- 10k 以后基本平台，仍比 base 低约 0.028 QWK。

### 5.4 配对检验

| 对比 | ΔQWK | p |
| --- | ---: | ---: |
| base − 5k（氟斑牙） | -0.0689 | 8.5e-6 |
| base − 5k（口腔疾病） | -0.0399 | 1.0e-6 |
| base − 10k（氟斑牙） | -0.0441 | 1.7e-4 |
| base − 10k（口腔疾病） | -0.0285 | 6.4e-5 |
| base − 15k（氟斑牙） | -0.0309 | 2.6e-4 |
| base − 15k（口腔疾病） | -0.0273 | 3.2e-4 |
| base − 16k（氟斑牙） | -0.0324 | 2.6e-4 |
| base − 16k（口腔疾病） | -0.0280 | 3.7e-4 |

**全部显著低于 base（p < 0.001 左右）。**

### 5.5 特征几何

| 数据集 | 模型 | pair cosine | participation ratio |
| --- | --- | ---: | ---: |
| 氟斑牙 200 | base | 0.823 | 94.4 |
| 氟斑牙 200 | 5k | 0.156 / 0.147 | 148.6 / 149.0 |
| 氟斑牙 200 | 10k | 0.139 / 0.133 | 145.7 / 146.1 |
| 氟斑牙 200 | 15k | 0.142 / 0.142 | 144.4 / 144.1 |
| 氟斑牙 200 | 16k | 0.142 / 0.142 | 144.4 / 144.1 |
| 口腔疾病 720 | base | 0.628 | 191.1 |
| 口腔疾病 720 | 5k | 0.074 / 0.074 | 347.7 / 348.8 |
| 口腔疾病 720 | 10k | 0.068 / 0.066 | 341.0 / 341.1 |
| 口腔疾病 720 | 15k | 0.070 / 0.070 | 338.7 / 338.3 |
| 口腔疾病 720 | 16k | 0.070 / 0.069 | 338.7 / 338.1 |

---

## 6. 关键发现与诊断

### 6.1 P0 → Stage 1 的恢复是真实的

- 公开 6 类 acc：P0 0.613 → Stage 1 0.799；base 0.828；
- 说明 softmax+center、batch 320、K=32768、Gram off 这些修改有效；
- 但不同 probe 协议下绝对值不完全可比，方向是明确的。

### 6.2 早停假设被否定

原假设：P0 最佳在 5k–10k，Stage 1 也可能“越训越差”，早停可救。

实际：

- 氟斑牙：5k 0.847 → 10k 0.872 → 15k 0.886 → 16k 0.884；
- 口腔疾病：5k 0.808 → 10k 0.819 → 15k/16k 0.820；
- **5k 最差，15k 最好；训练越久越好，但一直没追上 base。**

结论：问题不是“训练太久”，早停不能关闭差距。

### 6.3 过度分散在 5k 就已经发生

- 氟斑牙 pair cosine：base 0.823 → 5k 0.156 → 16k 0.142；
- 口腔疾病：base 0.628 → 5k 0.074 → 16k 0.069。

说明：

- backbone 在 5k 就已经离开 base 的语义几何；
- 之后几何基本不变，但下游仍从 5k 到 15k 缓慢改善；
- 所以 **pair cosine 是“漂移”的强信号，但不是唯一解释**；信息没有完全丢失，只是线性可分性变差。

### 6.4 为什么仍然低于 base

可能原因（按优先级）：

1. **全参数 SSL 目标与下游目标错配**：DINO/iBOT 优化的是裁剪一致性，不是疾病类别边界；
2. **K/N = 0.77 偏高**：32768 prototypes / 42,616 图，head 容易记忆；
3. **KoLeo 0.05 仍在推远特征**：当前已经过度分散；
4. **数据源偏置**：COde 87%、低分辨率；与氟斑牙高分辨率临床图域不匹配；
5. **全参数漂移**：从 5k 就发生，说明不是训练长度问题，而是 recipe/数据本身；
6. **base 非常强**：base DINOv3 在 200 张上 QWK 0.916，本身已经很高，提升空间小。

---

## 7. 当前未验证 / 不确定

| 项 | 状态 | 影响 |
| --- | --- | --- |
| LoRA / 全量微调 | 未做 | frozen probe 只是下界，可能低估 backbone |
| 512 特征 / Stage 2 | 未做 | 高分辨率是否加分未知 |
| 检测 A3 | 未做 | 检测任务排序可能与分类 probe 不同 |
| 完整 11,653 张口内疾病集 | 只用 720 子集 | 外部结论的稳定性 |
| 氟斑牙 4 级只有 200 张 | 方差较大 | 需配对检验/多种子 |
| 数据源质量/多样性 | 未系统消融 | 可能是漂移的根因 |

---

## 8. 下一步决策树

### Gate 1：LoRA / 全量微调能否救回来？（最高优先级）

在 `dental_fluorosis` 200 张和口腔疾病 720 张上：

| 实验 | 目的 |
| --- | --- |
| base + LoRA | 参考 |
| **15k/16k + LoRA** | 关键：是否追上 base |
| base + 全量微调 | 参考 |
| **15k/16k + 全量微调** | 信息是否还在 |

判据：

- trained + LoRA ≥ base + LoRA：backbone 还有价值，frozen probe 太严格；论文 A 可继续；
- trained + LoRA < base + LoRA：预训练伤害了特征；改配方或换路线。

建议：LoRA rank 4–8、alpha 8–16、q/k/v/out、lr 1e-3、10–20 epochs、早停、多种子；200 张上慎用全量微调。

### Gate 2：改一轮配方（如果 LoRA 也不行）

优先做“抑制漂移”的改动：

1. **base 特征蒸馏**：CLS/patch 特征与 base 的 cosine/MSE，权重 0.1–0.5；
2. **冻结 Gram anchor**：`ema_teacher: false` + `ckpt=base` + `rep_update: false`；
3. **降 K**：32768 → 8192/16384；
4. **降 KoLeo**：0.05 → 0.01–0.02；
5. **降 LR / 短 schedule**：5e-5–8e-5，30–60 epochs，5k–10k 步就 probe；
6. **数据平衡/质量**：COde 降权、AlphaDent 高分辨率加权、过滤低质量图。

Gate：5k–10k 步 probe；pair cosine 是否回到 0.3–0.5，QWK 是否接近/超过 base。

### Gate 3：换路线

- base DINOv3 + LoRA/线性头做下游；
- 检测 A3 仍可做，但故事从“域内预训练提升”改为“base backbone + 适配/检测”；
- 论文 A 改为“系统评估/负结果分析”，或把主结果改为适配/检测；
- 论文 B 继续暂缓。

---

## 9. 建议时间表

| 时间 | 任务 | 交付物 |
| --- | --- | --- |
| 本周 | 15k/16k + LoRA / 全量（两任务） | 适配对比表 |
| 本周 | 512 特征 probe（可选，成本低） | 分辨率结论 |
| 下周 | Gate 1 决策；若失败，启动一轮新配方 | 新 checkpoint |
| 2 周内 | 新配方 probe；若有效，Stage 2 / 检测 A3 | mAP / QWK |
| 3–4 周 | 消融、论文初稿 | draft |

---

## 10. 论文层面

- 当前摘要已经写明“5k/10k/15k/16k 均未超过 base”，这是诚实的；
- 如果 LoRA/全量有正向信号：论文 A 维持“域内预训练 + 适配”的故事；
- 如果没有：
  1. 不要硬写“性能显著提升”；
  2. 可以选择“负结果 + 失败机制分析”的写法（有价值，但风险高）；
  3. 或把论文 A 改成“口腔基础模型适配/检测”的方法论文；
  4. 论文 B 继续暂缓。

---

## 11. 风险

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| frozen probe 低于 base | 论文 A 主结果不成立 | 先做 LoRA/全量 |
| 5k 就漂移 | 早停不能救 | 改配方（蒸馏/降 K/降 KoLeo） |
| 200 张方差大 | probe 结论不稳 | 配对检验、多种子 |
| 数据源偏置 | 漂移根因 | 来源平衡/质量过滤 |
| GPU 中断/争用 | 训练周期不可控 | checkpoint/keep_every，协调 GPU |
| 时间线 | 论文 A 延期 | 设定 Gate 和时间盒 |

---

## 12. 复现与文件

### 评估脚本

```text
lab_train/pretrain/eval_stage1_checkpoint.py   # positive 旧集（4-exclude/4-merge/5-class）
lab_train/pretrain/eval_dental_fluorosis.py    # dental_fluorosis 4 级 200 张
lab_train/pretrain/eval_stage1_disease.py      # 口腔疾病 6 类 720 张
```

### 结果目录

```text
lab_train/output/eval-4999-dental-fluorosis/
lab_train/output/eval-9999-dental-fluorosis/
lab_train/output/eval-14999-dental-fluorosis/
lab_train/output/eval-15999-dental-fluorosis/
lab_train/output/eval-4999-disease/
lab_train/output/eval-9999-disease/
lab_train/output/eval-14999-disease/
lab_train/output/eval-15999-disease/
```

每个目录包含 `probe_summary.md`、`probe_summary.csv`、`probe_details.csv`、`features.npz`。

### 训练配置/输出

```text
lab_train/output/official-dental-vitb-224/config.yaml
lab_train/output/official-dental-vitb-224/ckpt/{4999_keep,9999_keep,14999_keep,15999}
lab_train/output/official-dental-vitb-224/logs/log.txt
lab_train/output/official-dental-vitb-224/training_metrics.json
```

### 相关提交

```text
db9ad2c  feat: add frozen linear probe for official DINOv3 checkpoints
483d433  feat: add 6-class oral disease probe
0e689ac  feat: add 4-level dental fluorosis probe
63edddf  fix: use checkpoint name in probe summary title
```

---

## 13. 需要导师决策的问题

1. 论文 A 是否接受“负结果/系统评估”的写法？
2. 是否允许再花 1–2 周做一轮配方迭代（base 蒸馏/降 K/降 KoLeo/短训练/数据平衡）？
3. 如果 base + LoRA 效果最好，是否可以把它作为论文主方法？
4. 下游优先级：氟斑牙分级，还是检测 A3？

---

## 附录：关键数字速查

| 指标 | 值 |
| --- | ---: |
| 预训练唯一图 | 42,618 |
| 氟斑牙 4 级 | 200（50/class） |
| 口腔疾病 6 类 | 720（120/class） |
| 检测 test | 828 |
| base 氟斑牙 QWK | 0.916 |
| 5k / 10k / 15k / 16k 氟斑牙 QWK | 0.847 / 0.872 / 0.886 / 0.884 |
| base 口腔疾病 QWK | 0.848 |
| 5k / 10k / 15k / 16k 口腔疾病 QWK | 0.808 / 0.819 / 0.820 / 0.820 |
| 氟斑牙 pair cosine（base → 5k → 16k） | 0.823 → 0.156 → 0.142 |
| 口腔疾病 pair cosine（base → 5k → 16k） | 0.628 → 0.074 → 0.070 |
| P0 公开 6 类 acc | 0.613 |
| Stage 1 公开 6 类 acc | 0.799 |
| base 公开 6 类 acc | 0.828 |
