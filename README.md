# MRG-05-Tooth-Seg

面向口腔图像的单类别牙齿检测项目。当前基线使用 YOLO11 和 YOLO 检测框标注，输出牙齿边界框、可视化结果、检测 JSON 以及单颗牙齿裁剪图。当前未实现像素级分割、FDI 牙位识别或氟斑牙分级。

## 当前数据集

`dataset/raw/` 已删除。当前数据集直接由以下目录组成：

```text
dataset/
  images/
    train/       # 2481 张有标注图像
    val/         # 857 张有标注图像
    test/        # 828 张无标注推理图像
  labels/
    train/       # 与 images/train 按文件 stem 对应
    val/         # 与 images/val 按文件 stem 对应
  classes.txt
  data.yaml
  notes.json
  split_manifest.json
  dataset_manifest.json
```

数据集共包含 3338 对图像和标签、48923 个检测框，类别为单一的 `tooth`。标签使用 YOLO 检测格式：

```text
class x_center y_center width height
```

类别 ID 为 `0`，坐标归一化到 `[0, 1]`。`images/test` 没有标签，不参与训练或定量验证。

## 安装

```bash
pip install -e .
pip install -e .[train]
```

## 数据校验

```bash
python cli/main.py validate --data dataset
```

校验不会修改图像、标签或数据集清单。当前数据集已经完成划分，不需要执行 `prepare`；该命令用于从外部原始数据源重新构建数据集，而本项目当前不包含 `dataset/raw/`。

## 训练

```bash
python cli/main.py train --data-yaml dataset/data.yaml --epochs 50 --imgsz 1024 --batch -1 --project output/train --name tooth-detect
```

如需指定 GPU，可添加 `--device 0`；显存不足时可降低 `--imgsz` 或使用较小的 `--batch`。

## 验证

```bash
python cli/main.py val \
  --weights output/train/tooth-detect/weights/best.pt \
  --data-yaml dataset/data.yaml
```

验证指标只适用于带标签的 `val` 数据集，不应使用无标签测试图像推断模型性能。

## 推理

```bash
python cli/main.py predict \
  --weights output/train/tooth-detect/weights/best.pt \
  --image dataset/images/test/example.jpg \
  --output output/predict
```

```bash
python cli/main.py predict-dir \
  --weights output/train/tooth-detect/weights/best.pt \
  --image-dir dataset/images/test \
  --output output/predict
```

```bash
python cli/main.py predict-visualize \
  --weights output/train/tooth-detect/weights/best.pt \
  --image-dir dataset/images/test \
  --output output/predict \
  --conf 0.25
```

`predict` 和 `predict-dir` 会生成原图、检测框叠加图、检测 JSON 以及单颗牙齿裁剪图；`predict-visualize` 主要生成聚合后的可视化图和 JSON。

## 测试

```bash
python -m pytest -q
```

## 注意事项

- 当前任务是单类别牙齿检测，不是像素级牙齿分割。
- 检测结果不包含 FDI 牙位编号，也不构成氟斑牙分级或临床诊断。
- 更新清单时不得修改 `dataset/images` 和 `dataset/labels` 中的图像与标签文件。
