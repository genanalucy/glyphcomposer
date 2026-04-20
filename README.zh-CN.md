# GlyphGen

[English README](README.md)

建议的仓库名：`glyphcomposer`

GlyphGen 是一个面向毕业设计的“结构化字形生成”研究原型项目。
它的目标是把两个部件和一个二元结构标签，转换成一张灰度单字图像。

这个仓库围绕四条主线组织：

1. 从字体与拆字表自动构建监督数据集。
2. 训练条件 U-Net 基线模型。
3. 在同一条件接口下训练 VAE 和潜空间扩散模型。
4. 提供评测、命令行推理和 Gradio 演示界面。

## 任务定义

本项目解决的是“受结构约束的字形生成”问题，而不是开放域的文生图。

- 输入：`component_a`、`component_b`、`structure`，以及可选的 `style_id`
- 输出：一张 `256x256` 的灰度单字图像
- 支持两种输入模式：
  - `text_component`：部件以 Unicode 文本输入，并通过字体渲染成图像
  - `image_component`：部件直接以灰度图输入，主要用于未登录字符或 OOV 扩展

默认支持的结构标签如下：

- `left_right`
- `top_bottom`
- `full_surround`
- `surround_from_above`
- `surround_from_below`
- `surround_from_left`
- `surround_from_upper_left`
- `surround_from_upper_right`
- `surround_from_lower_left`

## 仓库结构

```text
configs/                  示例 YAML 配置文件
scripts/                  数据构建、训练、评测、推理、演示脚本
src/glyphgen/             核心源码包
assets/                   示例拆字表与 OOV 模板
tests/                    轻量级 smoke tests
```

## 快速开始

1. 创建虚拟环境。
2. 安装 `requirements.txt` 中的依赖。
3. 准备字体文件目录和拆字 CSV。
4. 构建数据集清单与图像资源。
5. 训练基线模型。
6. 按需继续训练 VAE、扩散模型、结构探针和 OOV LoRA 适配模型。

示例命令：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/build_dataset.py \
  --decomposition-csv assets/decompositions_sample.csv \
  --font-path /System/Library/Fonts/Hiragino\ Sans\ GB.ttc \
  --output-dir data/generated/sample_dataset

python scripts/train_baseline.py \
  --config configs/baseline.yaml \
  --manifest data/generated/sample_dataset/manifests/train_random.jsonl \
  --val-manifest data/generated/sample_dataset/manifests/val_random.jsonl

python scripts/infer.py \
  --checkpoint checkpoints/baseline/baseline_last.pt \
  --component-a 犭 \
  --component-b 句 \
  --structure left_right \
  --output outputs/dog.png
```

主训练流程：

```bash
python scripts/train_vae.py \
  --config configs/vae.yaml \
  --manifest data/generated/sample_dataset/manifests/train_random.jsonl \
  --val-manifest data/generated/sample_dataset/manifests/val_random.jsonl

python scripts/train_diffusion.py \
  --config configs/diffusion.yaml \
  --manifest data/generated/sample_dataset/manifests/train_random.jsonl \
  --val-manifest data/generated/sample_dataset/manifests/val_random.jsonl \
  --vae-checkpoint checkpoints/vae/vae_best.pt

python scripts/train_structure_probe.py \
  --config configs/probe.yaml \
  --manifest data/generated/sample_dataset/manifests/train_random.jsonl \
  --val-manifest data/generated/sample_dataset/manifests/val_random.jsonl
```

## 数据格式

每条 manifest 样本固定包含：

- `target_char`
- `font_id`
- `glyph_image`
- `structure`
- `component_a`
- `component_b`
- `component_a_image`
- `component_b_image`
- `style_id`
- `split`

V1 版本只保留“恰好包含两个叶子部件和一个二元结构标签”的样本。

如果需要处理未登录字符或 OOV 样本，请使用图像条件数据准备脚本：

```bash
python scripts/prepare_oov_dataset.py \
  --csv assets/oov_image_samples_template.csv \
  --output-dir data/generated/oov_dataset
```

## 说明

- 仓库本身不附带字体文件。
- 示例拆字 CSV 很小，只用于验证整条数据处理链是否可运行。
- OOV 模板需要你手动提供目标字图和对应的部件图。
- 最终输出的字图必须由模型生成；规则程序只用于构造训练数据、标签和布局热图。
