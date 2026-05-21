import torch.nn.functional as F
from data import *
import os
import subprocess
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from net import *
from train import test


def get_least_used_gpu():
    """选择显存占用最少的 GPU，返回 'cuda:N'"""
    if not torch.cuda.is_available():
        return 'cpu'
    # 优先用 pynvml，其次解析 nvidia-smi
    try:
        import pynvml
        pynvml.nvmlInit()
        best, best_mem = 0, float('inf')
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used
            if mem < best_mem:
                best, best_mem = i, mem
        pynvml.nvmlShutdown()
        return f'cuda:{best}'
    except ImportError:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        mems = [int(x.strip()) for x in result.stdout.strip().split('\n')]
        best = int(min(enumerate(mems), key=lambda x: x[1])[0])
        return f'cuda:{best}'


device = get_least_used_gpu()
torch.cuda.set_device(device)  # 设置默认 CUDA 设备
print(f"使用设备: {device}")

user_movie_dict, item_num = load_ratings(os.path.join(os.getcwd(), 'ml-1m/ratings.dat'))
train_dict, test_dict = split_data(user_movie_dict, neg_num=128)
traindata = MovieDataset(train_dict)
testdata = MovieDataset(test_dict)
train_loader = DataLoader(traindata, batch_size=64, shuffle=True, collate_fn=lambda b: collate_fn(b, rating_threshold=1))
test_loader = DataLoader(testdata, batch_size=64, shuffle=False, collate_fn=lambda b: collate_fn(b, rating_threshold=1))
print("数据加载完成！")
net = Net(item_num=item_num, embed_dim=256, q_dim=32, v_dim=32, num_heads=4, num_layers=6, seq_len=256, dropout=0.2).to(device)
if os.path.exists(os.path.join(os.getcwd(), 'best_model.pth')):
    net.load_state_dict(torch.load(os.path.join(os.getcwd(), 'best_model.pth'), map_location=device))
    print("加载预训练模型完成！")
epoch = 1000
optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=1e-5)
all_items = torch.arange(1, item_num + 1, dtype=torch.long, device=device)
best_hr = 0
best_ndcg = 0
test_hr, test_ndcg = test(net, test_loader, all_items, k=200)
with open(os.path.join(os.getcwd(), 'test_log.txt'), 'a') as f:
    f.write(f"HR: {test_hr}, NDCG: {test_ndcg}\n")
# for e in range(epoch):
#     net.train()
#     total_loss = 0
#     for batch in train_loader:
#         sequences = batch['sequences'].to(device)
#         timestamps = batch['timestamps'].to(device)
#         targets = batch['targets'].to(device)
#         masks = batch['masks'].to(device)
#         vaild_masks = batch['vaild_masks'].to(device)
#         optimizer.zero_grad()
#         if e == 0 and total_loss == 0:
#             print(f"数据设备: {sequences.device}, 模型设备: {next(net.parameters()).device}")
#         outputs = net(sequences, masks, timestamps) # batch * seq * embbed_dim
#         seq = torch.cat((sequences, targets), dim=1)  # batch * (seq+1)

#         # # 有效位置的 (batch_idx, seq_idx)
#         valid_idx = vaild_masks.nonzero(as_tuple=False)  # (total_valid, 2)
#         b_idx = valid_idx[:, 0]  # (total_valid,)
#         s_idx = valid_idx[:, 1]  # (total_valid,)
#         output = outputs[b_idx, s_idx]                     # (total_valid, embbed_dim)

#         # ---- 旧版：全量分类 CrossEntropyLoss ----
#         # logits = torch.matmul(output, net.get_item_embbed(all_items).T)  # (total_valid, item_num)
#         # label = seq[b_idx, s_idx + 1] - 1  # (total_valid,)
#         # loss = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1)(logits, label)
#         # ---- 新版：SampledSoftmaxLoss + 动态采样（与 Meta 原版一致）----
#         # 每正样本动态采样 128 个负样本，只在 1 正 + 128 负上做 softmax
#         neg_num = 128  # Meta 原版默认 num_to_sample=128
#         pos_item = seq[b_idx, s_idx + 1]                    # (total_valid,) — 物品 id
#         # 动态采样：每次 forward 从均匀分布 [1, item_num] 中随机采样，而非使用预采样的固定负样本
#         neg_items_valid = torch.randint(1, item_num + 1, (pos_item.shape[0], neg_num), device=device)
#         pos_emb = F.normalize(net.item_embedding(pos_item), dim=-1)              # (total_valid, embed_dim) L2 归一 → 余弦相似度
#         neg_emb = F.normalize(net.item_embedding(neg_items_valid), dim=-1)       # (total_valid, neg_num, embed_dim)
#         pos_logit = (output * pos_emb).sum(dim=-1, keepdim=True)                    # (total_valid, 1)
#         neg_logit = torch.bmm(neg_emb, output.unsqueeze(-1)).squeeze(-1)            # (total_valid, neg_num)
#         # 屏蔽负样本中恰好是正样本的情况（设为 -5e4 ≈ -inf）
#         neg_is_pos = (neg_items_valid == pos_item.unsqueeze(-1))                    # (total_valid, neg_num)
#         neg_logit = neg_logit.masked_fill(neg_is_pos, float('-5e4'))
#         logits = torch.cat([pos_logit, neg_logit], dim=-1)  # (total_valid, 1 + neg_num)
#         logits = logits / 0.05                               # softmax_temperature
#         loss = -torch.log_softmax(logits, dim=-1)[:, 0].mean()  # 只在正样本上计算 loss
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     with open(os.path.join(os.getcwd(), 'train_log.txt'), 'a') as f:
#         f.write(f"Epoch {e+1}/{epoch}, Loss: {total_loss/len(train_loader)}\n")
#     scheduler.step()
#     if (e + 1) % 10 == 0:
#         test_hr, test_ndcg = test(net, test_loader, all_items, k=10)
#         with open(os.path.join(os.getcwd(), 'test_log.txt'), 'a') as f:
#             f.write(f"Epoch {e+1}/{epoch}, HR: {test_hr}, NDCG: {test_ndcg}\n")
#         if test_hr > best_hr and test_ndcg > best_ndcg:
#             best_hr = test_hr
#             best_ndcg = test_ndcg
#             torch.save(net.state_dict(), os.path.join(os.getcwd(), 'best_model.pth'))


                
                

    

