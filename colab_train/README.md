# Colab 训练启动器

`train.py` 是 Google Colab 训练入口，功能与 `kaggle_train/train.py` 对齐，但默认使用 `/content` 路径，并支持把数据集和输出目录放在挂载的 Google Drive 中。此目录是独立的 Colab 版本，不会修改 Kaggle 启动器。

当前项目是单类别 `tooth` 目标检测，不是像素级分割，也不输出 FDI 牙位或临床分级结果。

## Colab 准备

建议在 Notebook 中先执行：

```python
from google.colab import drive
drive.mount('/content/drive')
```

数据可以放在 `/content` 或 `/content/drive/MyDrive/`。已准备数据应包含以下任一结构：

```text
dataset/
  images/train/
  images/val/
  labels/train/
  labels/val/
```

或 OMNI 风格：

```text
dataset/
  train/
  val/
  labels/train/
  labels/val/
```

第二种结构会被复制为标准 `images/`、`labels/` 视图，避免 Ultralytics 解析路径时丢失图像与标签对应关系。原始输入不会被修改。

## 运行方式

先从仓库下载启动器：

```python
!wget -q -O /content/train.py \\
  https://raw.githubusercontent.com/XiaoHG/MRG-05-Tooth-Seg/main/colab_train/train.py
```

建议先进行 1 epoch 冒烟测试：

```python
!python /content/train.py \\
  --repo-url https://github.com/XiaoHG/MRG-05-Tooth-Seg.git \\
  --repo-ref main \\
  --dataset /content/drive/MyDrive/tooth-dataset/after \\
  --dataset-mode prepared \\
  --model yolo11n.pt \\
  --epochs 1 \\
  --imgsz 640 \\
  --batch 1 \\
  --device 0 \\
  --name smoke-test
```

确认数据路径、GPU 和标签无误后，再启动正式训练：

```python
!python /content/train.py \\
  --repo-url https://github.com/XiaoHG/MRG-05-Tooth-Seg.git \\
  --repo-ref main \\
  --dataset /content/drive/MyDrive/tooth-dataset/after \\
  --dataset-mode prepared \\
  --model yolo11n.pt \\
  --epochs 50 \\
  --imgsz 1024 \\
  --batch -1 \\
  --device 0 \\
  --output-dir /content/drive/MyDrive/toothseg-output \\
  --name omni-coco-after-yolo11n
```

RT-DETR-L 或已有权重也可以选择：

```python
!python /content/train.py \\
  --repo-url https://github.com/XiaoHG/MRG-05-Tooth-Seg.git \\
  --dataset /content/drive/MyDrive/tooth-dataset/after \\
  --dataset-mode prepared \\
  --model rtdetr-l.pt \\
  --epochs 50 \\
  --imgsz 1024 \\
  --batch -1 \\
  --device 0 \\
  --name omni-coco-rtdetr-l-after
```

使用已有权重初始化训练时增加 `--weights /path/to/checkpoint.pt`。这表示用该权重初始化新训练，不等同于从中断处恢复；中断恢复应使用 Ultralytics 的 `resume` 机制。

## 输出

默认输出位于：

```text
/content/toothseg-output/
  train/<experiment-name>/
    .runtime/dataset.yaml
    weights/best.pt
    weights/last.pt
    args.yaml
    results.csv
```

如果使用 `--output-dir /content/drive/MyDrive/toothseg-output`，训练结果会直接写入 Google Drive。建议保存整个实验目录，而不仅是 `best.pt`，同时保留数据版本、Git commit、参数和验证结果。

## 参数

| 参数 | 说明 |
| --- | --- |
| `--repo-url` | Git 仓库地址，必须可从 Colab 访问 |
| `--repo-ref` | branch、tag 或 commit；正式实验建议使用固定 commit |
| `--dataset` | `/content` 或 Google Drive 中的数据根目录 |
| `--dataset-mode` | `prepared` 使用已有 train/val；`raw` 由项目 CLI 准备；`auto` 自动判断 |
| `--model` | Ultralytics 模型名或配置，如 `yolo11n.pt`、`yolo11s.pt`、`rtdetr-l.pt` |
| `--weights` | 可选的初始 `.pt` 权重 |
| `--epochs` | 训练轮数 |
| `--imgsz` | 输入尺寸；显存不足时可改为 640 |
| `--batch` | `-1` 自动选择，显存不足时可改为 1 或 2 |
| `--device` | Colab GPU 通常使用 `0`；省略则使用 Ultralytics 默认选择 |
| `--output-dir` | 输出位置，可放在 `/content` 或 Google Drive |
| `--name` | 实验名称，每次实验应使用不同名称 |

## 注意事项

- 在 Colab 菜单中确认已选择 GPU 运行时。
- `/content` 是临时运行空间；需要长期保存的结果应写入 Google Drive 或在会话结束前下载。
- Colab 需要联网以 clone 仓库和安装训练依赖。
- `test` 集应在模型和参数确定后作为独立评估集，不要在调参阶段反复使用。
