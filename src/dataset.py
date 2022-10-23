from random import randint
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import pandas as pd 

def load_data(args):
    dataset_folder = os.path.abspath(os.path.join('dataset'))

    train_set = pd.read_csv(os.path.join(dataset_folder, 'train.csv'))
    test_set = pd.read_csv(os.path.join(dataset_folder, 'test.csv'))

    user_num = train_set.user_id.max() + 1
    item_num = train_set.item_id.max() + 1

    train_dataset = CTRDataset(train_set, neg=1, user_num=user_num, item_num=item_num)
    test_dataset = CTRDataset(test_set, neg=1, user_num=user_num, item_num=item_num)

    train_loader = DataLoader(train_dataset, 
                                batch_size=args.batch_size, 
                                shuffle=True,
                                num_workers=2,
                                collate_fn=collate_fn)
                                
    test_loader = DataLoader(test_dataset, 
                                batch_size=args.batch_size, 
                                shuffle=False,
                                num_workers=2,
                                collate_fn=collate_fn)

    return train_loader, test_loader, user_num, item_num


class CTRDataset(Dataset):
    def __init__(self, data, neg=1, user_num=-1, item_num=-1) -> None:
        super().__init__()
        # data: dataset csv file 
        # neg: negative sample number

        self.data = data
        self.neg = neg

        self.user_num, self.item_num = user_num, item_num # use to sample negative sample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index): 

        users = [self.data.iloc[index].user_id]
        items = [self.data.iloc[index].item_id]
        labels = [1]

        # negative sample
        for _ in range(self.neg):
            users.append(randint(0, self.user_num-1))
            items.append(randint(0, self.item_num-1))
            labels.append(0)

        return {'user': users,
                'item': items,
                'label': labels}

def collate_fn(batch_data):
    batch_users = []
    batch_items = []
    batch_labels = []

    for data in batch_data:
        batch_users.extend(data['user'])
        batch_items.extend(data['item'])
        batch_labels.extend(data['label'])

    batch_users = torch.LongTensor(batch_users)
    batch_items = torch.LongTensor(batch_items)
    batch_labels = torch.FloatTensor(batch_labels)

    batch = {'user': batch_users,
                'item': batch_items,
                'label': batch_labels}
    return batch
