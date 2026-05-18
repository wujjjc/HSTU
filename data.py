"""格式
所有评分都包含在文件 "ratings.dat" 中，格式如下：

UserID::MovieID::Rating::Timestamp

UserID 范围在 1 到 6040 之间

MovieID 范围在 1 到 3952 之间

评分为 5 星制（仅整数星）

Timestamp 表示自纪元以来的秒数，由 time(2) 返回

每个用户至少有 20 条评分


用户信息包含在文件 "users.dat" 中，格式如下：

UserID::Gender::Age::Occupation::Zip-code

所有人口统计信息均由用户自愿提供，未经准确性校验。数据集中仅包含提供了一些人口统计信息的用户。

Gender 以 "M" 表示男性，"F" 表示女性

Age 从以下范围中选择：

1: "18 岁以下"

18: "18-24 岁"

25: "25-34 岁"

35: "35-44 岁"

45: "45-49 岁"

50: "50-55 岁"

56: "56 岁以上"

Occupation 从以下选项中选择：

0: "其他" 或未指定

1: "学术/教育工作者"

2: "艺术家"

3: "文员/行政"

4: "大学生/研究生"

5: "客户服务"

6: "医生/医疗保健"

7: "高管/经理"

8: "农民"

9: "家庭主妇"

10: "K-12 学生"

11: "律师"

12: "程序员"

13: "退休人员"

14: "销售/市场营销"

15: "科学家"

16: "自雇人士"

17: "技术员/工程师"

18: "技工/手工艺者"

19: "失业"

20: "作家"



电影信息包含在文件 "movies.dat" 中，格式如下：

MovieID::Title::Genres

标题与 IMDB 提供的标题相同（包含发行年份）

Genres 以竖线分隔，并从以下类型中选择：

Action（动作）

Adventure（冒险）

Animation（动画）

Children's（儿童）

Comedy（喜剧）

Crime（犯罪）

Documentary（纪录片）

Drama（剧情）

Fantasy（奇幻）

Film-Noir（黑色电影）

Horror（恐怖）

Musical（音乐剧）

Mystery（悬疑）

Romance（爱情）

Sci-Fi（科幻）

Thriller（惊悚）

War（战争）

Western（西部）

部分 MovieID 由于意外的重复条目和/或测试条目而不对应实际电影

电影大部分为手工录入，因此可能存在错误和不一致之处。
"""

def load_ratings(rating_file):
    """_summary_
    读取数据
    Args:
        rating_file (_type_): ratings.dat 文件路径
    return:
        user_movie_dict: {user_id: [(movie_id, timestamp, rating), ...], ...}} (排序好了的)
    """
    user_movie_dict = {}
    print("开始读取数据...")
    for line in open(rating_file):
        user_id, movie_id, rating, timestamp = line.strip().split("::")
        user_id = int(user_id)
        movie_id = int(movie_id)
        rating = int(rating)
        timestamp = int(timestamp)
        user_movie_dict.setdefault(user_id, [])
        user_movie_dict[user_id].append((movie_id, timestamp, rating))
    print("数据读取完成！")
    for user, lst in user_movie_dict.items():
        user_movie_dict[user] = sorted(lst, key=lambda x: x[1]) # 按照时间戳排序
    return user_movie_dict

def split_data(user_movie_dict):
    """_summary_
    进行训练集验证集划分，最后一个movie作为测试集，其他的作为训练集

    Args:
        user_movie_dict (_type_): {user_id: [(movie_id, timestamp, rating), ...], ...}} (排序好了的)

    Returns:
        _type_: train_dict:{
            'sequence': [[movie_id1, movie_id2, ...], ...],
            'timestamp': [[timestamp1, timestamp2, ...], ...],
            'rating': [[rating1, rating2, ...], ...],
        }
        test_dict:{
            'sequence': [[movie_id1, movie_id2, ...], ...],
            'timestamp': [[timestamp1, timestamp2, ...], ...],
            'rating': [[rating1, rating2, ...], ...],
            'target':[[movie_id1], [movie_id2], ...] 召回目标
        }
    """
    train_dict = {
        'sequence': [],
        'timestamp': [],
        'rating': [],
        'target':[]
    }
    test_dict = {
        'sequence': [],
        'timestamp': [],
        'rating': [],
        'target':[]
    }
    for user, lst in user_movie_dict.items():
        if len(lst) < 2:
            continue
        seq = [x[0] for x in lst]
        timestamp = [x[1] for x in lst]
        rating = [x[2] for x in lst]
        train_dict['sequence'].append(seq[:-1])
        train_dict['timestamp'].append(timestamp[:-1])
        train_dict['rating'].append(rating[:-1])
        train_dict['target'].append([seq[-1]])
        test_dict['sequence'].append(seq[:-1])
        test_dict['timestamp'].append(timestamp[:-1])
        test_dict['rating'].append(rating[:-1])
        test_dict['target'].append([seq[-1]])
    return train_dict, test_dict

import torch
import torch.utils.data as data
class MovieDataset(data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
    
    def __len__(self):
        return len(self.data['sequence'])
    
    def __getitem__(self, idx):
        sequence = self.data['sequence'][idx]
        timestamp = self.data['timestamp'][idx]
        rating = self.data['rating'][idx]
        target = self.data['target'][idx]
        return sequence, timestamp, rating, target

def collate_fn(batch, max_len=1024, rating_threshold=1):
    """_summary_
    返回所需要的数据格式，主要是进行padding
    Args:
        batch (_type_): _description_
    Returns:
        {
            'sequences': [batch_size, seq_len]
            'timestamps': [batch_size, seq_len]
            'targets': [batch_size, 1]
            'masks': [batch_size, seq_len]
            'vaild_masks': [batch_size, seq_len] 这个为了区别是否需要在这个位置进行训练，不是所有的评分电影用户都喜欢的
        }
        """
    sequences = []
    timestamps = []
    targets = [torch.tensor(x[-1], dtype=torch.long) for x in batch]
    pad_masks = []
    vaild_masks = []
    for seq, times, rat, _ in batch:
        seq_len = len(seq)
        if seq_len > max_len:
            seq = seq[-max_len:]
            times = times[-max_len:]
            rat = rat[-max_len:]
            padding_len = 0
        else:
            padding_len = max_len - seq_len
            seq = [0] * padding_len + seq
            times = [0] * padding_len + times
            rat = [0] * padding_len + rat
        padding_mask = [0] * padding_len + [1] * (max_len - padding_len)
        vaild_mask = [1 if r >= rating_threshold else 0 for r in rat] # 只有评分大于等于rating_threshold的电影才是用户喜欢的电影，才需要在这个位置进行训练
        sequences.append(torch.tensor(seq, dtype=torch.long))
        timestamps.append(torch.tensor(times, dtype=torch.long))
        pad_masks.append(torch.tensor(padding_mask, dtype=torch.bool))
        vaild_masks.append(torch.tensor(vaild_mask, dtype=torch.bool))
    return {
        'sequences': torch.stack(sequences),
        'timestamps': torch.stack(timestamps),
        'targets': torch.stack(targets),
        'masks': torch.stack(pad_masks),
        'vaild_masks': torch.stack(vaild_masks)
    }