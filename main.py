from data import *
import os
from torch.utils.data import DataLoader
user_movie_dict = load_ratings(os.path.join(os.getcwd(), 'ml-1m/ratings.dat'))
train_dict, test_dict = split_data(user_movie_dict)
train = MovieDataset(train_dict)
test = MovieDataset(test_dict)
train_loader = DataLoader(train, batch_size=1024, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test, batch_size=1024, shuffle=False, collate_fn=collate_fn)
print("数据加载完成！")

