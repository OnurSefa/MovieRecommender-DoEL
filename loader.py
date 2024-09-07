from torch.utils.data import Dataset
import random
import torch
from encode import Encoder
import os
import math
import json


class MovieDataset(Dataset):
    def __init__(self, names_path, movie_data_dir, aux_path, device, mode='train'):
        self.encoder = Encoder(movie_data_dir, aux_path, device)
        self.mode = mode
        self.device = device
        with open(names_path, 'r') as f:
            self.names = json.load(f)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        path = self.names[idx]
        with open(path, 'r') as f:
            line = f.readline()
        return self.encoder.encode(line)


def split_dataset(dataset_dir, train_path, same_user_test_path, different_user_test_path, same_user_ratio=0.1, different_user_ratio=0.1, random_seed=42):
    random.seed(random_seed)

    all_names = {}
    for name in os.listdir(dataset_dir):
        user_id, example_id = name.split('.')[0].split('_')
        if user_id not in all_names:
            all_names[user_id] = [example_id]
        else:
            all_names[user_id].append(example_id)

    user_ids = list(all_names.keys())
    random.shuffle(user_ids)

    train_user_count = math.floor(len(user_ids)*(1-different_user_ratio))
    train_user_ids = user_ids[:train_user_count]
    different_user_ids = user_ids[train_user_count:]

    train_names = []
    test_same_user_names = []
    test_diff_user_names = []

    for train_user_id in train_user_ids:
        example_ids = all_names[train_user_id]
        example_count = math.ceil(len(example_ids)*(1-same_user_ratio))
        for example_id in example_ids[:example_count]:
            train_names.append(f'{dataset_dir}/{train_user_id}_{example_id}.txt')
        for example_id in example_ids[example_count:]:
            test_same_user_names.append(f'{dataset_dir}/{train_user_id}_{example_id}.txt')
    for test_user_id in different_user_ids:
        example_ids = all_names[test_user_id]
        for example_id in example_ids:
            test_diff_user_names.append(f'{dataset_dir}/{test_user_id}_{example_id}.txt')

    with open(train_path, 'w') as f:
        json.dump(train_names, f, indent=4)
    with open(same_user_test_path, 'w') as f:
        json.dump(test_same_user_names, f, indent=4)
    with open(different_user_test_path, 'w') as f:
        json.dump(test_diff_user_names, f, indent=4)


if __name__ == '__main__':
    split_dataset('../movie_recommender_data/dataset', '../movie_recommender_data/train_names.json', '../movie_recommender_data/test_names_same_user.json', '../movie_recommender_data/test_names_different_user.json')
