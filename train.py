import torch
import numpy as np
def test(net, test_loader, all_items, k=10):
    """_summary_
    计算HR和NDCG，判断模型在测试集上的表现
    Args:
        net (_type_): 模型
        test_loader (_type_): 测试集
        all_items (_type_): 所有物品的id
        k (int, optional): 召回个数 Defaults to 10.

    Returns:
        _type_: HR和NDCG
    """
    status = net.training
    net.eval()
    recalls = []
    ndcgs = []
    with torch.no_grad():
        for batch in test_loader:
            sequences = batch['sequences'].to(next(net.parameters()).device)
            timestamps = batch['timestamps'].to(next(net.parameters()).device)
            targets = batch['targets'].to(next(net.parameters()).device)
            masks = batch['masks'].to(next(net.parameters()).device)
            outputs = net(sequences, masks, timestamps) # batch * seq * embbed_dim
            all_items_embbed = net.get_item_embbed(all_items) #item_num * embbed_dim
            output = outputs[:, -1, :] # batch * embbed_dim
            score = torch.matmul(output, all_items_embbed.T)  # batch * item_num
            # 排除用户已交互过的物品，避免"推荐已看过=作弊"
            history_mask = torch.zeros(score.shape[0], score.shape[1] + 1, dtype=torch.bool, device=score.device)
            history_mask.scatter_(1, sequences, True)  # 用 item_id 直接做索引
            history_mask = history_mask[:, 1:]          # 去掉 padding(0) 列，shape -> (batch, item_num)
            score = score.masked_fill(history_mask, float('-inf'))
            if len(recalls) == 0:
                print(f"score range: {score.min():.4f} ~ {score.max():.4f}")
                print(f"score std per sample: {score.std(dim=-1).mean():.4f}")
                print(f"sample top-5: {score[0].topk(5).indices.tolist()}")
                print(f"target[0]: {targets[0].item()}")
            _, indices = torch.topk(score, k=k) # batch * k
            targets = targets.cpu().numpy().squeeze(-1) # batch
            indices = indices.cpu().numpy() + 1 # batch * k
            for target, index in zip(targets, indices):   
                target = target.item()                                                                                                                                                                                                                                                                      
                rank = np.where(index == target)[0]                                                                                                                                           
                if len(rank) > 0:                                                                                                                                                             
                    recalls.append(1)                                                                                                                                                         
                    ndcgs.append(1.0 / np.log2(rank[0] + 2))
                else:
                    recalls.append(0)
                    ndcgs.append(0)
    if status:
        net.train()
    return sum(recalls) / len(recalls), sum(ndcgs) / len(ndcgs)