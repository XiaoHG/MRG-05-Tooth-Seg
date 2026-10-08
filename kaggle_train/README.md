# Kaggle 训练启动器

`train.py` 是 Kaggle 训练入口。它会从 GitHub 拉取指定版本的项目代码，安装训练依赖，然后调用项目的 `cli/main.py train`。

当前项目是 YOLO11 单类别 `tooth` 检测，不是像素级分割，也不输出 FDI 牙位或临床分级结果。

## 数据集结构

启动器支持两种 prepared 数据集结构。

标准 YOLO 结构：

```text
dataset/
  images/train/
  images/val/
  labels/train/
  labels/val/
  data.yaml                 # 可选
```

当前 OMNI 数据集结构：

```text
before/
  train/                    # 训练图像
  val/                      # 验证图像
  test/                     # 可选，无标签图像
  annotations/train/        # 训练标签
  annotations/val/          # 验证标签
```

通过 `--dataset` 传入数据集根目录即可。启动器会自动识别上述结构。对于第二种结构，脚本会将图像和标签复制到：

```text
/kaggle/working/toothseg-output/dataset-view/
  images/train/
  images/val/
  labels/train/
  labels/val/
```

复制是必要的，因为 Ultralytics 会解析符号链接并根据 `images/` 推导 `labels/` 路径。原始 Kaggle Dataset 只读，不会被修改。

## Kaggle Notebook 用法

建议直接从 GitHub 下载启动器，不需要把 `train.py` 上传到 Kaggle Dataset：

```python
!wget -q -O /kaggle/working/train.py \
  https://raw.githubusercontent.com/XiaoHG/MRG-05-Tooth-Seg/f71045f/kaggle_train/train.py
```

先运行 1 个 epoch 验证路径、标签和 GPU：

```python
!python /kaggle/working/train.py \
  --repo-url https://github.com/XiaoHG/MRG-05-Tooth-Seg.git \
  --repo-ref main \
  --dataset /kaggle/input/datasets/hgxiao/omni-coco-before-after/before \
  --dataset-mode prepared \
  --epochs 1 \
  --imgsz 640 \
  --batch 1 \
  --device 0 \
  --name smoke-test
```

冒烟测试成功后运行正式训练：

```python
!python /kaggle/working/train.py \
  --repo-url https://github.com/XiaoHG/MRG-05-Tooth-Seg.git \
  --repo-ref main \
  --dataset /kaggle/input/datasets/hgxiao/omni-coco-before-after/before \
  --dataset-mode prepared \
  --epochs 50 \
  --imgsz 1024 \
  --batch -1 \
  --device 0 \
  --name omni-coco-before
```

运行日志中应看到训练数据位于：

```text
/kaggle/working/toothseg-output/dataset-view/images/train
```

如果日志仍显示直接读取 `/kaggle/input/.../before/train`，说明 Notebook 使用了旧版启动器，需要重新执行上面的 `wget`。

## 参数说明

| 参数 | 说明 |
| --- | --- |
| `--repo-url` | GitHub 仓库地址 |
| `--repo-ref` | branch、tag 或 commit；正式实验建议使用 commit |
| `--dataset` | Kaggle Dataset 中传入的实际数据根目录 |
| `--dataset-mode prepared` | 使用已有 train/val 划分，不重新划分数据 |
| `--epochs` | 训练轮数 |
| `--imgsz` | 输入尺寸；显存不足时可改为 640 |
| `--batch` | `-1` 自动选择，显存不足时可改为 1 或 2 |
| `--device` | Kaggle GPU 通常使用 `0` |
| `--weights` | 可选的初始 `.pt` 权重；不传则使用 YOLO11n 预训练权重 |
| `--name` | 输出实验名称，每次实验建议使用不同名称 |

## 输出位置

```text
/kaggle/working/toothseg-output/
  dataset-view/
  dataset.yaml
  train/<experiment-name>/
    weights/best.pt
    weights/last.pt
    args.yaml
    results.csv
```

训练完成后，请从 Kaggle Output 下载整个实验目录，而不只是 `best.pt`，同时保留数据集版本、Git commit、训练参数和验证结果。

## 注意事项

- Kaggle Notebook 需要开启 Internet，脚本要从 GitHub clone 项目并安装 `ultralytics`。
- `/kaggle/input` 是只读目录，训练输出写入 `/kaggle/working`。
- `test` 图像没有标签，不能用于计算 mAP、precision 或 recall。
- `best.pt` 用于初始化新的训练实验；中断恢复应使用 Ultralytics 的 `last.pt` 恢复机制，不能混淆两者语义。
