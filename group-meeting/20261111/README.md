# 组会报告（2026-11-11）

本目录只保存本次组会报告的大纲、生成脚本与 PPT，不修改项目原有内容。

## 状态

- [x] 大纲（`OUTLINE.md`）——已审核
- [x] 数字来源与口径（`SOURCES.md`）
- [x] 审核意见回填
- [x] 完整版 PPT（`OralDINO_group_meeting_20261111.pptx`，21 页）
- [x] 10 分钟版 PPT（`OralDINO_group_meeting_20261111_10min.pptx`，12 页；含 DINOv3 架构页）
- [x] 手动版备份（`OralDINO_group_meeting_20261111_10min_manual_backup.pptx`）
- [ ] 视觉微调 / 现场排练（按需）

## 当前结论（2026-11-11）

- Stage 1 已完成 16,080 步；最终 checkpoint `ckpt/15999`，`ckpt/14999_keep` 也已评估。
- 冻结线性 probe（CLS @224，5 折 × 5 次），氟斑牙 QWK：
  - base **0.916** → 5k **0.847** → 10k **0.872** → 15k **0.886** → 16k **0.884**。
- 口腔疾病 6 类 QWK：
  - base **0.848** → 5k **0.808** → 10k **0.819** → 15k/16k **0.820**。
- 5k/10k/15k/16k 全部低于 base（p<0.001）；**5k 最差，15k 最好** → 早停假设否定。
- 特征过度分散在 **5k 已发生**：pair cosine 0.156（氟斑牙）/0.074（口腔疾病），之后基本不变。
- 下一步：base 蒸馏/冻结 Gram anchor、降 K/KoLeo/LR、数据平衡；LoRA/全量验证。

## 审核结论（已回填）

| 项 | 结论 |
| --- | --- |
| 汇报时长 | 10 分钟（10 分钟版）；20 分钟（完整版） |
| 目标会议/期刊 | 待定（MICCAI / ISBI / MedIA 候选） |
| 论文主下游 | 氟斑牙分级，同时做其他牙齿疾病检测 |
| P0 失败复盘 | 保留、展示 |
| 论文 B / 单牙 token | 不包含 |
| 数据来源与 license | 展示 |
| 汇报人姓名/单位/日期 | 不需要 |
| 英文版 slides | 不需要 |
| speaker notes | 不需要 |
| 氟斑牙任务定义 | 4 级：normal / mild / moderate / severe |
| 氟斑牙样本量口径 | 200 张（dental_fluorosis，4 级各 50；P0 的 141 张为旧口径） |
| 检测基线展示 | before/after 同时展示 |
| Stage 1 进度 | 已完成 16,080 步；15999/14999 已做 frozen probe |

## 报告结构

### 完整版（21 页）

1. 创新点安排、实验设计、模型架构、论文撰写安排（Slide 3–13、19–20）
2. 实验进度报告（Slide 14–16：检测基线 / P0 复盘 / Stage 1 + probe）
3. 问题、下一步与论文安排（Slide 17–19）
4. 论文摘要与总结（Slide 20–21）

### 10 分钟版（12 页，已整理）

1. 封面
2. 背景、域鸿沟与标注瓶颈（含 RQ1–RQ3）
3. 三个创新点 + A0–A4 路线 + 讨论点（保留手动补充）
4. **对 DINOv3 的理解（含架构图）**
5. 多源数据与预训练设计（含 Stage 0/1/2 参数）
6. 模型架构与下游评估（含 A3 术语说明）
7. 数据、评估协议与实验矩阵
8. 检测基线（8 模型 before/after）
9. Stage 1 完成 + 5k/10k/15k/16k probe 结果
10. 问题、下一步与论文安排
11. 论文摘要（保留）
12. 总结与讨论

> 手动删除的 P0 复盘页保留删除状态；其“loss ≠ 下游”的结论仍保留在总结页。
> P0 的完整内容仍在完整版第 15 页。

## 文件

| 文件 | 用途 |
| --- | --- |
| `OUTLINE.md` | 逐页大纲、讲稿提示、待确认事项、图表清单 |
| `SOURCES.md` | 所有数字的来源与口径，便于审核核对 |
| `OralDINO_group_meeting_20261111.pptx` | 完整版 PPT（21 页，16:9） |
| `OralDINO_group_meeting_20261111_10min.pptx` | 10 分钟版 PPT（12 页，16:9，含 DINOv3 架构页） |
| `OralDINO_group_meeting_20261111_10min_manual_backup.pptx` | 手动编辑版备份（11 页） |
| `build_ppt.py` | 完整版生成脚本 |
| `build_ppt_10min.py` | 10 分钟版生成脚本 |
| `README.md` | 本目录说明与审核状态 |

## 重新生成 PPT

```bash
# 完整版（21 页）
/tmp/pptx-env/bin/python group-meeting/20261111/build_ppt.py

# 10 分钟版（12 页）
/tmp/pptx-env/bin/python group-meeting/20261111/build_ppt_10min.py
```

依赖：`python-pptx`（本机临时 venv 在 `/tmp/pptx-env`）。
如果换机器，先 `python3 -m venv /tmp/pptx-env && /tmp/pptx-env/bin/pip install python-pptx`。

## 汇报前检查

1. 打开 10 分钟版，确认中文字体（PingFang SC / Arial）在本机正常；
2. 第 9 页的 probe 表应与你最新的评估结果一致；如果补了更多 checkpoint，更新表格；
3. 第 10 页的下一步应与当前结论一致（配方迭代/LoRA）；
4. 确认目标会议/期刊，更新第 10 页的论文安排。
