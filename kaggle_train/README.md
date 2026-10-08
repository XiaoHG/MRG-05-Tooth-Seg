# Kaggle 训练项目

本目录是 Kaggle 训练启动器，不是原项目的第二份实现。Kaggle 运行时会从 GitHub 拉取指定版本的原项目，安装其 `[train]` 依赖，然后调用原项目的 `cli/main.py train`。这样 Kaggle 使用的代码版本由 GitHub 的 branch/tag/commit 控制，避免手工复制代码后与原项目漂移。

当前原项目实际是 YOLO11 单类别 `tooth` 边界框检测，输出检测框，不是像素级分割；也不输出 FDI 牙位或氟斑牙临床分级。Kaggle 训练结果只能在带人工真值的验证集上评估，不能把无标签测试图像当作定量验证集。

## 1. 准备 GitHub 仓库

1. 将原项目推送到 GitHub。推荐使用公开仓库；如果是私有仓库，需要在 Kaggle 中安全地提供访问凭据，不能把 Token 写入 Notebook、脚本或 Dataset。
2. 确认 GitHub 目标版本包含 `cli/main.py`、`src/toothseg/` 和 `pyproject.toml`，并且可以在本地运行：

```bash
python cli/main.py validate --data dataset
```

3. 记下仓库 URL 和要训练的 branch、tag 或 commit。默认 ref 是 `main`，正式实验建议使用不可变 commit 或 tag，并将它记录到实验备注中。

## 2. 准备并上传数据集

数据集应作为 Kaggle Dataset 上传，而不是直接把大量图片塞进 Notebook。Kaggle Dataset 创建后会挂载到 `/kaggle/input/<dataset-slug>/`，目录名以 Kaggle 页面实际显示的 slug 为准。

### 推荐：上传已经划分好的 prepared 数据集

将以下目录直接放在上传包的根目录：

```text
tooth-dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml
  classes.txt                 # 可选
  split_manifest.json         # 推荐保留
  dataset_manifest.json       # 推荐保留
```

其中每个图片和标签应按 stem 一一对应，例如 `images/train/case001.jpg` 对应 `labels/train/case001.txt`。标签是单类别 YOLO 检测格式：

```text
class_id x_center y_center width height
```

当前类别 ID 应为 `0`，坐标归一化到 `[0, 1]`，宽高必须为正数。`data.yaml` 中的 `path` 应能在 Kaggle 挂载路径下工作。最稳妥的方式是使用项目 `prepare` 生成的 `data.yaml`，并在上传前检查其中的路径是否为相对路径或当前数据集路径。

### 备选：上传 raw 数据集

如果希望在 Kaggle 中按固定随机种子重新划分数据，可以上传原始数据源目录：

```text
tooth-raw-dataset/
  source_a/
    images/
    labels/
    classes.txt
  source_b/
    images/
    labels/
    classes.txt
```

每个 source 需要符合原项目支持的 raw YOLO 数据源结构。启动器会调用原项目的：

```bash
python cli/main.py prepare --data <raw-dir> --output /kaggle/working/toothseg-output/dataset --val-ratio 0.2 --seed 42
```

推荐优先上传 prepared 数据集：它能固定 train/val 划分，便于不同实验比较。raw 模式只会生成 Kaggle 工作目录中的副本，不会修改 Kaggle Dataset 原文件。

如果数据集采用 `train/`、`val/` 和 `annotations/train/`、`annotations/val/` 结构，也可以直接使用 `--dataset-mode prepared`。启动器会根据 `--dataset` 传入的路径自动识别该布局，在工作目录创建链接视图后训练，不要求把目录改名为 `images/` 和 `labels/`。

## 3. Kaggle 页面配置
wo 
1. 新建 Kaggle Notebook，选择 GPU 加速器。YOLO11 训练通常应使用 GPU；CPU 只适合冒烟测试。
2. 在 Notebook 的 **Add Input** 中添加上传的数据集，记下实际挂载路径。
3. 在 Notebook 设置中打开 **Internet**，因为启动器需要 `git clone` GitHub，并从 pip 安装依赖；如果 Internet 被禁用，运行会在 clone 或安装阶段失败。
4. 将本目录中的 `train.py` 上传为 Notebook 文件，或将 `kaggle_train` 作为一个小型 Kaggle Dataset 添加后引用它。
5. 不要把训练输出写入 `/kaggle/input`。该目录是只读输入；脚本默认写入 `/kaggle/working/toothseg-output`。

## 4. 最小训练命令

在 Kaggle Notebook 单元格运行：

```python
!python /kaggle/working/train.py \
  --repo-url https://github.com/YOUR_ACCOUNT/MRG-05-Tooth-Seg.git \
  --repo-ref main \
  --dataset /kaggle/input/tooth-dataset \
  --dataset-mode prepared \
  --epochs 50 \
  --imgsz 1024 \
  --batch -1 \
  --device 0 \
  --name tooth-detect-kaggle-v1
```

如果 `train.py` 位于一个 Dataset 中，将路径替换为该 Dataset 的实际路径，例如 `/kaggle/input/toothseg-kaggle-train/train.py`。如果是直接上传到 Notebook 的文件，Kaggle 常见位置是 `/kaggle/working/train.py`。

建议第一次先运行 1 个 epoch 验证路径、依赖和 GPU：

```python
!python /kaggle/working/train.py \
  --repo-url https://github.com/YOUR_ACCOUNT/MRG-05-Tooth-Seg.git \
  --repo-ref main \
  --dataset /kaggle/input/tooth-dataset \
  --dataset-mode prepared \
  --epochs 1 \
  --imgsz 640 \
  --batch 1 \
  --device 0 \
  --name smoke-test
```

## 5. 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--repo-url` | 无 | GitHub 仓库 URL；也可用环境变量 `TOOTHSEG_REPO_URL` |
| `--repo-ref` | `main` | branch、tag 或 commit。建议正式实验使用 tag/commit |
| `--dataset` | 无 | Kaggle Dataset 的挂载目录 |
| `--dataset-mode` | `auto` | `prepared` 使用现成 `data.yaml`；`raw` 调用项目 `prepare`；`auto` 自动判断 |
| `--epochs` | `50` | 训练轮数 |
| `--imgsz` | `1024` | 输入尺寸；显存不足时可降到 `640` |
| `--batch` | `-1` | `-1` 由 Ultralytics 自动选择；OOM 时改成 `1`、`2` 等 |
| `--device` | 空 | Kaggle GPU 通常填写 `0` |
| `--no-amp` | 关闭 | 关闭混合精度；只有遇到 AMP/CUDA 兼容问题时使用 |
| `--weights` | 空 | 可传入可访问的 `.pt` 初始权重。未提供时使用 `yolo11n.pt` |
| `--val-ratio` | `0.2` | raw 模式的验证集比例 |
| `--seed` | `42` | raw 模式的数据划分随机种子 |
| `--name` | `tooth-detect-kaggle` | 输出实验目录名，建议每次实验唯一 |

## 6. 输出和下载

训练结束后，重点文件位于：

```text
/kaggle/working/toothseg-output/
  dataset/                         # raw 模式生成
  train/<experiment-name>/
    weights/best.pt
    weights/last.pt
    args.yaml
    results.csv
    results.png
```

在 Kaggle 右侧 **Output** 中保存版本，或在 Notebook 末尾打包下载：

```python
!zip -r /kaggle/working/toothseg-training-output.zip /kaggle/working/toothseg-output/train
```

不要只下载 `best.pt`；同时保存 `args.yaml`、`results.csv`、验证指标、GitHub commit、数据集版本、训练参数和日志，保证实验可复现。模型权重应通过 `val` 在固定人工标注验证集上比较：

```python
!python /kaggle/working/toothseg-repo/cli/main.py val \
  --weights /kaggle/working/toothseg-output/train/tooth-detect-kaggle-v1/weights/best.pt \
  --data-yaml /kaggle/input/tooth-dataset/data.yaml
```

## 7. 继续训练与恢复训练

`--weights` 表示用已有 `best.pt` 初始化一次新的训练实验，不是恢复原实验状态：

```python
!python /kaggle/working/train.py \
  --repo-url https://github.com/YOUR_ACCOUNT/MRG-05-Tooth-Seg.git \
  --repo-ref <commit-or-tag> \
  --dataset /kaggle/input/tooth-dataset \
  --dataset-mode prepared \
  --weights /kaggle/input/previous-run/best.pt \
  --epochs 50 \
  --device 0 \
  --name tooth-detect-kaggle-v2
```

当前项目 CLI 暴露的是 `best.pt` 初始化训练接口，没有提供 `train --resume` 参数。若需要从中断位置恢复，应在同一 Kaggle 运行中使用 Ultralytics 的原生恢复机制，或补充项目 CLI 后再使用；不要把 `best.pt` 和 `last.pt` 的语义混用。

## 8. 常见问题

**`git clone` 失败**：确认 Kaggle Internet 已打开，URL 可公开访问，且 `--repo-ref` 存在。私有仓库不要把 Token 放到命令历史或公开 Notebook 中。

**`No module named ultralytics`**：启动器会执行 `pip install -e .[train]`。检查 clone 的 GitHub 版本确实包含 `pyproject.toml`，并重新运行安装步骤；Kaggle 的 pip 安装日志中应出现 `ultralytics`。

**找不到 `data.yaml` 或图片**：先在 Notebook 中执行 `!find /kaggle/input/tooth-dataset -maxdepth 3 -type f | head`，确认 `--dataset` 指向数据集根目录，而不是多包了一层目录。prepared 模式必须同时存在 `images/train`、`images/val`、`labels/train`、`labels/val` 和 `data.yaml`。

**CUDA out of memory**：降低 `--imgsz`，显式设置较小的 `--batch`，必要时使用 `--no-amp` 仅用于排查。Kaggle GPU 型号不同，不能保证 `1024/-1` 在每次运行中都适合。

**训练结果覆盖**：每次运行使用不同的 `--name`。脚本会清理并重新 clone `/kaggle/working/toothseg-repo`，但不会清理输入 Dataset，也不会覆盖 `/kaggle/input`。

**验证指标异常**：检查 train/val 是否有数据泄漏、标签是否与图片 stem 匹配、类别 ID 是否为 `0`，并确认比较的是同一个固定人工标注验证集。无标签 test 数据只能用于定性推理，不能计算 mAP。

## 9. 一次完整实验的记录清单

- GitHub 仓库 URL、commit/tag 和 `pyproject.toml` 版本。
- Kaggle Dataset slug、Dataset version、prepared/raw 模式及数据清单哈希。
- Kaggle GPU 型号、Ultralytics/PyTorch 版本。
- `epochs`、`imgsz`、`batch`、`device`、AMP 设置、初始权重。
- `best.pt`、`last.pt`、`args.yaml`、`results.csv` 和验证输出。
- 固定人工验证集上的 precision、recall、mAP50、mAP50-95，以及典型误检和漏检案例。
