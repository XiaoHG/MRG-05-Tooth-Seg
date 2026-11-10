# 组会报告数字来源与口径说明

本文件用于审核时核对大纲中的数字。所有数字均来自项目现有文件，本目录不修改原始内容。

> 数据截止：2026-11-10。汇报前请按最新训练日志刷新 Stage 1 进度。

---

## 1. 检测基线（8 模型，828 固定 test）

来源：

- `output/val-test/results.csv`（after 标签）
- `output/val-test-before/results.csv`（before 标签）
- `output/val-test/comparison.md`（before/after 对比、Δ、排名）

口径：

- 固定 `test` split = 828 张，不参与调参；
- `before` = 人工复核前标签，`after` = 复核后标签；
- 8 个模型：YOLO11n、YOLO11s、YOLOv8m、YOLO26m、RT-DETR-L、Faster R-CNN、RetinaNet、RT-DETRv2-L；
- RF-DETR-M 因原生 `validate` 不支持 `--split`，未纳入固定 test 对比。

大纲使用的关键数字：

| 指标 | 值 | 来源 |
| --- | ---: | --- |
| RT-DETRv2-L mAP50-95（after） | 0.7137 | `output/val-test/results.csv` |
| RT-DETRv2-L mAP50（after） | 0.8749 | 同上 |
| YOLO26m mAP50（after） | 0.9867 | 同上 |
| before → after 的 mAP50-95 提升 | +0.0224 ~ +0.0440 | `output/val-test/comparison.md` |

---

## 2. P0 失败复盘

来源：

- `output/pretrain/WHY_WORSE_THAN_BASE.md`（机制分析、特征几何、稳定性）
- `lab_train/output/pretrain/oraldino-vitb16-512-42k-p0/EVALUATION.md`（P0 评估）
- `lab_train/README.md` 第 706 行附近（P0 实测结论）

口径：

- P0 = 42k、batch 8、512、KoLeo 0.1、Sinkhorn；
- 下游 = 氟斑牙 5 级（141 张）+ 公开口腔疾病 6 分类（720 张）；
- 冻结 CLS + 线性 probe。

大纲使用的关键数字：

| 指标 | 值 | 来源 |
| --- | ---: | --- |
| base DINOv3 氟斑牙 QWK | 0.718 | `WHY_WORSE_THAN_BASE.md` |
| P0 teacher 最佳 QWK（10k） | 0.315 | 同上 / `lab_train/README.md` |
| P0 student 最佳 QWK（5k） | 0.381 | 同上 |
| base 公开 6 类 acc | 0.820 | 同上 |
| P0 公开 6 类 acc | 0.613 | 同上 |
| P0 10k pair cosine | 0.039 | 同上 |
| P0 10k grayscale 余弦 | 0.301 | 同上 |

> 注意：`lab_train/output/pretrain/oraldino-vitb16-512/EVALUATION.md` 里还有一张
> 141 张 5 折 probe 表（base 0.718、oral512-8k 0.702、final 0.657 等），那是另一组
> checkpoint/配方评估；组会主表用 `WHY_WORSE_THAN_BASE.md` 的 0.315 口径。

---

## 3. 预训练数据审计

来源：

- `lab_train/output/data-audit/summary.md`
- `lab_train/output/data-audit/image_audit.csv`
- `lab_train/output/data-audit/summary.json`

关键数字：

| 指标 | 值 |
| --- | ---: |
| 扫描文件 | 55,559 |
| 唯一 SHA256 | 42,618 |
| 重复文件 | 12,941（9,721 组） |
| AlphaDent | 1,455（唯一 1,441） |
| COde-Dataset | 49,938（唯一 37,020） |
| omni_coco | 4,166（唯一 4,157） |
| 严格解码失败 | 2（AlphaDent 截断 JPEG） |
| 灰度/低饱和 | 3 |
| 与评测集哈希重叠 | 0 |
| 中位分辨率 | 448×298 |
| 短边 < 384 | 54,057 / 55,559 ≈ 97.3% |

---

## 4. 氟斑牙数据

来源：`dataset/positive/`

| 类别 | 张数 |
| --- | ---: |
| normal | 16 |
| doubtful | 35 |
| mild | 50 |
| medium | 36 |
| severe | 17 |
| **合计** | **154** |

> P0 评估时使用 141 张（doubtful 22）；当前目录已增至 154 张（doubtful 35）。
> 组会里如果引用 P0 的 QWK，需要注明“141 张评估口径”。

---

## 5. Stage 0 诊断

来源：

- `lab_train/output/official-dental-diag-224/logs/log.txt`
- `lab_train/output/official-dental-diag-224/training_metrics.json`

关键数字：

| 指标 | 值 |
| --- | ---: |
| 步数 | 2,220 |
| batch | 96 |
| 总耗时 | 23:35（0.638 s/it） |
| 显存峰值 | 7,923 MB |
| DINO loss（均匀基线） | ln(32768) = 10.397 |
| DINO loss（结束） | 9.68 |
| iBOT loss（结束） | 5.00 |
| KoLeo（结束） | -0.28 |

---

## 6. Stage 1 当前进度

来源：

- `lab_train/output/official-dental-vitb-224/logs/log.txt`
- `lab_train/output/official-dental-vitb-224/training_metrics.json`
- `lab_train/output/official-dental-vitb-224/stdout.log`
- `lab_train/output/official-dental-vitb-224/config.yaml`

关键数字（截至被 SIGKILL 前的最后一版日志）：

| 指标 | 值 |
| --- | ---: |
| 进度 | 7,080 / 16,080（44%） |
| batch | 320 |
| 显存峰值 | 21,594 MB |
| 稳态 step time | ~2.6 s/it（探针为 1.31 s/it） |
| DINO local / global（7,080） | 0.78 / 0.69 |
| iBOT（7,080） | 2.48 |
| KoLeo（7,080） | -0.25 |
| backbone grad norm | 1.83 |
| DINO head grad norm | 0.21 |
| iBOT head grad norm | 0.15 |
| 最后正常日志 | 2026-11-10 16:21:30 |
| SIGKILL | 2026-11-10 16:21:41（exitcode -9） |
| 最新完整 checkpoint | `ckpt/6999`（16:17:52 保存） |

> 如果汇报前已经续训，请按最新 `logs/log.txt` 和 `training_metrics.json` 更新本表。

---

## 7. 版本与论文安排

来源：

- `draft/README.md`（创新点、论文 A/B、版本迭代安排、验收标准）
- `versions/v13.md`（v13 收口记录）
- `lab_train/README.md` 第八节（官方 DINOv3 渐进式训练）
- `lab_train/OFFLINE_DINOV3.md`（离线安装、三阶段命令）

版本规划：

| 版本 | 内容 | 状态 |
| --- | --- | --- |
| v13 | 多模型对比收尾 + 冻结数据/下游基线 | 已完成 |
| v14 | 多源公开牙科数据整合 | 已完成 |
| v15 | OralDINO 域内预训练 | 进行中 |
| v16 | 下游有效性评估（LoRA/冻结/微调） | 未开始 |
| v17 | 论文 A 撰写与投稿 | 未开始 |

论文 A/B 的结构与验收线见 `draft/README.md` 的「论文 A / 论文 B」与「版本迭代安排」章节。

---

## 8. 审核时需要确认的口径

- [不用] P0 的 QWK 用 0.315（teacher 10k）还是 0.381（student 5k）作为主数字？
- [141] 氟斑牙样本量写 141（P0 评估口径）还是 154（当前目录口径）？
- [同时展示] 检测基线是写 after 的 8 个模型，还是同时展示 before/after 全表？
- [还没有] Stage 1 进度是否已更新（续训后）？
- [待定] 目标会议/期刊是否确定？
