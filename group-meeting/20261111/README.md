# 组会报告（2026-11-11）

本目录只保存本次组会报告的大纲、生成脚本与 PPT，不修改项目原有内容。

## 状态

- [x] 大纲（`OUTLINE.md`）——已审核
- [x] 数字来源与口径（`SOURCES.md`）
- [x] 审核意见回填
- [x] PPT 生成（`OralDINO_group_meeting_20261111.pptx`，20 页）
- [ ] 视觉微调 / 现场排练（按需）

## 审核结论（已回填）

| 项 | 结论 |
| --- | --- |
| 汇报时长 | 20 分钟 |
| 目标会议/期刊 | 待定（MICCAI / ISBI / MedIA 候选） |
| 论文 A 主下游 | 氟斑牙分级，同时做其他牙齿疾病检测 |
| P0 失败复盘 | 保留、展示 |
| 论文 B / 单牙 token | 不包含 |
| 数据来源与 license | 展示 |
| 汇报人姓名/单位/日期 | 不需要 |
| 英文版 slides | 不需要 |
| speaker notes | 不需要 |
| 氟斑牙样本量口径 | 141 张（P0 评估口径） |
| 检测基线展示 | before/after 同时展示 |
| Stage 1 进度 | 尚未续训；按 7,080/16,080 汇报 |

## 报告结构

1. 创新点安排、实验设计、模型架构、论文撰写安排（Slide 3–13、19）
2. 实验进度报告（Slide 14–16）
3. 下一步计划（Slide 17–18）

## 文件

| 文件 | 用途 |
| --- | --- |
| `OUTLINE.md` | 逐页大纲、讲稿提示、待确认事项、图表清单 |
| `SOURCES.md` | 所有数字的来源与口径，便于审核核对 |
| `OralDINO_group_meeting_20261111.pptx` | 最终 PPT（20 页，16:9） |
| `build_ppt.py` | PPT 生成脚本；改完内容后重新运行即可 |
| `README.md` | 本目录说明与审核状态 |

## 重新生成 PPT

```bash
/tmp/pptx-env/bin/python group-meeting/20261111/build_ppt.py
```

依赖：`python-pptx`（本机临时 venv 在 `/tmp/pptx-env`）。
如果换机器，先 `python3 -m venv /tmp/pptx-env && /tmp/pptx-env/bin/pip install python-pptx`。

## 汇报前检查

1. 打开 PPT，确认中文字体（PingFang SC / Arial）在本机正常；
2. 如果 Stage 1 已续训，更新 Slide 16 的进度、loss 与 checkpoint 数字；
3. 确认 Slide 17 的 SIGKILL 结论（dmesg / cgroup）是否已有结果；
4. 确认目标会议/期刊，更新 Slide 19。
