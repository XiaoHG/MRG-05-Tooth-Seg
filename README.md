# MRG-05-Tooth-Seg

面向口腔图像的牙齿检测工作区。当前 v5 是一个可复现的单类别牙齿检测流程：使用 YOLO11 根据五列 YOLO 标注检测牙齿边界框。目前尚未生成像素级分割掩码，也不输出 FDI 编号。

## 项目状态

- 原始标注数据保存在 `dataset/raw/` 中，保持不变。
- `dataset/images/train`、`dataset/images/val`、`dataset/labels/train` 和 `dataset/labels/val` 由原始数据生成。
- `dataset/images/test` 用于存放用户添加的、没有标注的预测图像。
- 项目支持训练、验证、推理和测试。
- v5 数据集准备和代码测试已经完成。
- v6 可以从预测 JSON 文件中导出伪标签候选数据，供人工审核。
- 已成功完成 1 个 epoch 的 CPU 冒烟运行；最终的 50 epoch 模型尚未验收。

## 数据集

`dataset` 的目录结构如下：

```text
dataset/
  raw/
    fluorosis/
    OMNI_COCO-seg-train/
  images/
    train/
    val/
    test/        # 仅存放没有标注的预测图像
  labels/
    train/
    val/
  classes.txt
  data.yaml
  split_manifest.json
  dataset_manifest.json
```

原始数据是不可变的输入。`prepare` 命令只重新生成训练集和验证集的图像与标签，不会修改 `dataset/raw/` 和 `dataset/images/test`。

数据格式：

- 单一类别：`tooth`
- YOLO 检测标签：`class x_center y_center width height`
- 使用固定随机种子将带标注的原始数据划分为训练集和验证集。
- `images/test` 不包含标签，不参与训练和定量验证。

## 安装

```bash
pip install -e .
pip install -e .[train]
```

## 构建数据集

验证生成的训练集和验证集：

```bash
python cli/main.py validate --data dataset
```

从 `dataset/raw/` 下的所有兼容数据源重新构建训练集和验证集：

```bash
python cli/main.py prepare --data dataset/raw --output dataset --val-ratio 0.2 --seed 42
```

该命令会生成 `data.yaml`、`split_manifest.json` 和 `dataset_manifest.json`。其中，`dataset_manifest.json` 记录数据源以及生成文件的分配情况。

## 训练

```bash
python cli/main.py train --data-yaml dataset/data.yaml --epochs 50 --imgsz 1024 --batch -1 --project output/train --name tooth-detect-v5
```

输出文件：

- `output/train/tooth-detect-v5/weights/best.pt`
- `output/train/tooth-detect-v5/weights/last.pt`

`--batch -1` 允许 Ultralytics 自动选择批量大小。需要指定 CUDA 设备时，可以添加 `--device 0` 选择第一个 CUDA 设备。如果旧型号 GPU 出现 AMP 相关的 CUDA 错误，可以使用 `--no-amp`。

在 Windows Conda 环境中，NumPy/MKL 与 PyTorch 可能加载重复的 Intel OpenMP 运行时。项目在导入 NumPy 前设置 `KMP_DUPLICATE_LIB_OK=TRUE`，以允许进程启动。这是兼容性处理，并不保证性能；生产训练应使用只包含一个 OpenMP 运行时的环境。

对于 5 GB 显存的 GPU，如果自动批量设置失败，可以使用较保守的配置：

```bash
python cli/main.py train --data-yaml dataset/data.yaml --epochs 50 --imgsz 640 --batch 1 --no-amp --device 0 --project output/train --name tooth-detect-v5
```

当前已验证的冒烟权重是 `output/train/tooth-detect-v5-cpu-smoke/weights/best.pt`。该权重仅经过 1 个 CPU epoch 的训练，适合验证流程，不适合用于性能结论。

## v7 增量训练

v7 面向加入审核数据后的重复训练轮次，尤其是 v6 产生的已批准伪标签。推荐做法是使用上一轮验证集表现最好的模型初始化新的实验，并在同一份固定的人工标注验证集上重新评估。

训练命令默认使用 `yolo11n.pt`。提供 `--weights` 后，可以使用指定模型初始化新一轮训练，例如已有的 `best.pt`：

```bash
python cli/main.py train --weights output/train/tooth-detect-v5-6/weights/best.pt --data-yaml dataset/data.yaml --epochs 50 --project output/train --name tooth-detect-v7
```

v7 的预期语义如下：

- 在训练完成并加入新的审核数据后，使用上一轮的 `best.pt` 进行初始化。这是一个新的微调实验，应使用新的输出名称，例如 `tooth-detect-v7`。
- 只有在训练因中断而需要继续时，才使用 `last.pt`。恢复模式会还原之前的训练状态，与使用 `best.pt` 初始化新实验不同。
- 只有在明确需要独立基线或完全重新开始时，才使用 `yolo11n.pt`。

初始化和恢复是两种不同的操作。使用现有 `best.pt` 初始化训练是受支持的：

```bash
python cli/main.py train --weights output/train/<previous-experiment>/weights/best.pt --data-yaml dataset/data.yaml --project output/train --name tooth-detect-v7
```

预期的恢复接口是：

```text
train --resume output/train/<unfinished-experiment>/weights/last.pt
```

CLI 当前尚不支持 `--resume`；该参数预留用于从 `last.pt` 恢复被中断的训练。不要覆盖之前的实验目录，应保留每一轮的权重、参数、数据集清单和验证结果。

每一轮增量训练都应使用固定的验证集清单重新构建数据集，使已批准的伪标签只进入训练集。应将精确率、召回率、mAP50、mAP50-95，以及具有代表性的误检和漏检牙齿案例与上一轮已接受模型进行比较。只有当固定人工标注验证集达到项目阈值时，才可以将新的 `best.pt` 作为下一轮初始化权重。如果性能下降，应回退到上一轮权重和数据集清单，而不是继续累积未经验证的伪标签。

## 验证

```bash
python cli/main.py val --weights output/train/tooth-detect-v5/weights/best.pt --data-yaml dataset/data.yaml
```

验证指标只覆盖带标注的 `val` 数据集。无标注的测试图像无法计算 mAP、精确率或召回率。

## 推理

预测单张图像：

```bash
python cli/main.py predict --weights output/train/tooth-detect-v5/weights/best.pt --image dataset/images/test/example.jpg --output output/predict
```

预测一个目录，包括无标注的 `images/test` 目录：

```bash
python cli/main.py predict-dir --weights output/train/tooth-detect-v5/weights/best.pt --image-dir dataset/images/test --output output/predict
```

每张图像都会写入 `output/predict/<image-name>/`：

- `<image-name>_original.png`：原始图像
- `<image-name>_overlay.png`：带检测框的完整图像
- `<image-name>_detections.json`：检测框和置信度
- `<image-name>_<index>.png`：单颗牙齿裁剪图

## v8 批量递归推理

v8 完善了目录批量推理能力。使用 `predict-dir` 时，可以指定一个输入根目录和一个输出目录，程序会递归遍历输入目录及其所有子目录中的图像文件，并逐张执行牙齿检测。

```bash
python cli/main.py predict-dir --weights output/train/tooth-detect-v7/weights/best.pt --image-dir dataset/images/test --output output/predict-v8
```

处理规则：

- 支持 `.jpg`、`.jpeg`、`.png`、`.bmp` 和 `.webp` 文件，扩展名不区分大小写。
- 输入目录下的所有子目录都会被递归扫描；非图片文件会被跳过。
- 输出目录由 `--output` 指定，不会修改输入图像。
- 所有预测结果都直接保存到指定输出目录下，每张图像使用原有的 `<image-name>/` 子目录和文件命名方式。
- 输入目录不存在或递归后没有支持的图像时，命令会报告错误并停止。

例如，输入目录如下：

```text
input-images/
  case-a/tooth.png
  case-b/molar.png
  overview.jpg
```

输出目录如下：

```text
output/predict-v8/
  tooth/tooth_original.png
  tooth/tooth_overlay.png
  tooth/tooth_detections.json
  molar/molar_original.png
  molar/molar_overlay.png
  molar/molar_detections.json
  overview/overview_original.png
  overview/overview_overlay.png
  overview/overview_detections.json
```

每个 `*_detections.json` 除了检测框和置信度外，还记录 `source_image` 字段，用于标识图像相对于输入根目录的路径。例如：

```json
{
  "image": "tooth",
  "source_image": "case-a/tooth.png",
  "detections": []
}
```

由于输出沿用原有的按文件名保存方式，如果不同子目录中存在相同 stem 的图片，后处理的结果可能覆盖先处理的结果；需要避免同名输入，或在调用前自行完成重命名。

v8 只扩展批量推理的数据流和输出组织方式，不改变当前单类别 `tooth` 检测模型，也不实现像素级分割、FDI 牙位识别或氟斑牙分级。

## v6 伪标签

导出置信度不低于 `0.9` 的检测结果和完整源图像，生成可供审核的候选数据集：

```bash
python cli/main.py pseudo-label-export --predict output/predict-v8 --output dataset/raw/pseudo-label-v8-candidates
```

该命令会从每个预测结果目录中读取对应的 `<image-name>_original.png` 原图，不再需要额外提供 `--images` 路径；随后将原图、YOLO 标签、源文件哈希、检测来源信息以及 `review_status: pending` 记录写入候选目录。命令不会修改 `dataset/raw/`。审核候选数据后，将选中记录的状态设置为 `review_status: approved`，然后发布：

```bash
python cli/main.py pseudo-label-publish --candidates dataset/raw/pseudo-label-v6-candidates --raw dataset/raw
```

在冻结现有人工标注验证集的情况下重新构建训练数据。已批准的伪标签只会分配到训练集：

```bash
python cli/main.py prepare --data dataset/raw --output dataset --fixed-val-manifest dataset/dataset_manifest.json --seed 42
```

v6 候选数据不是测试集，不得作为精确率、召回率或 mAP 的真实标注依据。

## 测试

```bash
python -m pytest -q
```

## 注意事项

- 当前标签用于训练检测框，而不是像素级分割。
- 像素级分割需要每颗牙齿的多边形或掩码标签，以及分割模型。
- 当前权重不是 FDI 编号模型。
- 该流程是科研基线，不是临床诊断系统。
