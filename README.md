# HSTU: Hierarchical Sequential Transduction Unit for Generative Recommendation

PyTorch implementation of the HSTU model from Meta's paper *"Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations"* (ICML 2024), evaluated on the MovieLens-1M dataset.

---

## Model Architecture

HSTU is a Transformer-based sequential recommendation model with the following key designs:

- **SiLU Attention**: Replaces softmax with SiLU activation for attention scores, combined with relative position and time biases (RAB)
- **Single UVQK Projection**: A single linear layer projects input into U, V, Q, K components, reducing parameter count
- **Gated Output**: Attention output is gated by U component after LayerNorm: `Y = f2(Norm(AV) * U)`
- **RelativeBucketedTimeAndPositionBasedBias (RAB)**: Log-scale bucketed relative position and time-difference embeddings added to attention logits
- **SampledSoftmaxLoss**: Dynamic negative sampling (128 negatives per positive) with temperature-scaled softmax, matching Meta's training recipe

### Default Hyperparameters

| Parameter | Value |
|---|---|
| `embed_dim` | 256 |
| `q_dim` / `v_dim` | 32 |
| `num_heads` | 4 |
| `num_layers` | 6 |
| `seq_len` | 256 |
| `dropout` | 0.2 |
| `neg_num` | 128 |
| `temperature` | 0.05 |
| `lr` | 1e-3 |
| `weight_decay` | 1e-2 |
| `batch_size` | 64 |

---

## Project Structure

```
HSTU/
├── net.py          # Model: PositionalEncoding, HSTU layer, Net (multi-layer HSTU)
├── data.py         # Data pipeline: MovieLens-1M loading, splitting, DataLoader
├── train.py        # Evaluation: HR@10 and NDCG@10 with history exclusion
├── main.py         # Training loop: AdamW + CosineAnnealing + SampledSoftmaxLoss
└── ml-1m/          # MovieLens-1M dataset (ratings.dat, movies.dat, users.dat)
```

---

## Quick Start

### Requirements

```bash
pip install torch numpy
```

### Download Data

Download [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/) and place `ratings.dat`, `movies.dat`, `users.dat` into `ml-1m/`.

### Train

```bash
python main.py
```

- Automatically selects the GPU with the most free memory
- Loads `best_model.pth` if available (resume training)
- Evaluates HR@10 and NDCG@10 every 10 epochs
- Saves best model to `best_model.pth`

---

## Results on MovieLens-1M

| Metric | HR | NDCG |
|---|---|---|
| @10 | 0.3296 | 0.1912 |
| @50 | 0.5778 | 0.2467 |
| @100 | 0.6755 | 0.2626 |
| @200 | 0.7654 | 0.2752 |

Training logs and evaluation metrics are saved to `train_log.txt` and `test_log.txt`.

---

## Reference

- [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152) (ICML 2024)
- [Meta's official implementation](https://github.com/facebookresearch/generative-recommenders)

---

# HSTU：层级序列转导单元生成式推荐

基于 Meta 论文 *"Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations"* (ICML 2024) 的 PyTorch 实现，在 MovieLens-1M 数据集上进行评估。

---

## 模型架构

HSTU 是一种基于 Transformer 的序列推荐模型，具有以下核心设计：

- **SiLU 注意力**：用 SiLU 激活函数替代 softmax，结合相对位置和时间偏置（RAB）
- **单一 UVQK 投影**：一个线性层将输入投影为 U、V、Q、K 四个分量，减少参数量
- **门控输出**：注意力输出经 LayerNorm 后与 U 分量逐元素相乘实现门控：`Y = f2(Norm(AV) * U)`
- **相对分桶位置时间偏置（RAB）**：对相对位置差和时间差进行对数分桶，生成可学习的注意力偏置
- **SampledSoftmaxLoss**：每个正样本动态采样 128 个负样本，温度缩放 softmax，与 Meta 训练方案一致

### 默认超参数

| 参数 | 值 |
|---|---|
| `embed_dim` | 256 |
| `q_dim` / `v_dim` | 32 |
| `num_heads` | 4 |
| `num_layers` | 6 |
| `seq_len` | 256 |
| `dropout` | 0.2 |
| `neg_num` | 128 |
| `temperature` | 0.05 |
| `lr` | 1e-3 |
| `weight_decay` | 1e-2 |
| `batch_size` | 64 |

---

## 项目结构

```
HSTU/
├── net.py          # 模型定义：位置编码、HSTU 层、多层 HSTU 网络
├── data.py         # 数据管道：MovieLens-1M 加载、划分、DataLoader
├── train.py        # 评估：HR@10 和 NDCG@10（排除已交互物品）
├── main.py         # 训练循环：AdamW + CosineAnnealing + SampledSoftmaxLoss
└── ml-1m/          # MovieLens-1M 数据集（ratings.dat, movies.dat, users.dat）
```

---

## 快速开始

### 依赖

```bash
pip install torch numpy
```

### 下载数据

下载 [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/) 数据集，将 `ratings.dat`、`movies.dat`、`users.dat` 放入 `ml-1m/` 目录。

### 训练

```bash
python main.py
```

- 自动选择显存占用最少的 GPU
- 若存在 `best_model.pth` 则加载继续训练
- 每 10 个 epoch 评估 HR@10 和 NDCG@10
- 最优模型保存至 `best_model.pth`

---

## MovieLens-1M 实验结果

| 指标 | HR | NDCG |
|---|---|---|
| @10 | 0.3296 | 0.1912 |
| @50 | 0.5778 | 0.2467 |
| @100 | 0.6755 | 0.2626 |
| @200 | 0.7654 | 0.2752 |

训练日志和评估指标分别保存在 `train_log.txt` 和 `test_log.txt` 中。

---

## 参考文献

- [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152) (ICML 2024)
- [Meta 官方实现](https://github.com/facebookresearch/generative-recommenders)
