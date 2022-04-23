import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaModel, RobertaConfig
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm as tq
from sklearn.metrics import accuracy_score


def tokenize(tok: RobertaTokenizer, df: pd.DataFrame, label_first=True):
    re = []
    for i, row in tq(df.iterrows()):
        if label_first:
            label = row[0]
            text = row[1]
        else:
            label = row[1]
            text = row[0]
        ids = tok.encode(text)
        re.append({'ids': ids, 'label': label})
    return re


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        hidden = self.roberta(x)[1]  # bs, 768
        hidden = self.relu(hidden)
        hidden = self.linear(hidden)  # bs, 2
        return hidden


def collate_fn(batch):
    global tokenizer
    ids = [torch.tensor(x['ids'], dtype=torch.long) for x in batch]
    labels = [x['label'] for x in batch]
    return pad_sequence(ids, batch_first=True, padding_value=tokenizer.pad_token_id), torch.tensor(labels,
                                                                                                   dtype=torch.long)


class MyDataset(Dataset):
    def __init__(self, data_list):
        super(MyDataset, self).__init__()
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


best_all_acc = 0


def test():
    sst_pred = []
    sst_labels = []
    for ids, l in tq(sst_test_loader):
        ids = ids.to(device)
        with torch.no_grad():
            logit = model(ids).cpu().numpy().tolist()
            sst_labels += l.cpu().numpy().tolist()
            for row in logit:
                if row[0] < row[1]:
                    sst_pred.append(1)
                else:
                    sst_pred.append(0)
    sst_acc = accuracy_score(sst_labels, sst_pred)
    mr_pred = []
    mr_labels = []
    for ids, l in tq(mr_test_loader):
        ids = ids.to(device)
        with torch.no_grad():
            logit = model(ids).cpu().numpy().tolist()
            mr_labels += l.cpu().numpy().tolist()
            for row in logit:
                if row[0] < row[1]:
                    mr_pred.append(1)
                else:
                    mr_pred.append(0)
    mr_acc = accuracy_score(mr_labels, mr_pred)
    cr_pred = []
    cr_labels = []
    for ids, l in tq(cr_test_loader):
        ids = ids.to(device)
        with torch.no_grad():
            logit = model(ids).cpu().numpy().tolist()
            cr_labels += l.cpu().numpy().tolist()
            for row in logit:
                if row[0] < row[1]:
                    cr_pred.append(1)
                else:
                    cr_pred.append(0)
    cr_acc = accuracy_score(cr_labels, cr_pred)
    all_pred = sst_pred + mr_pred + cr_pred
    all_labels = sst_labels + mr_labels + cr_labels
    all_acc = accuracy_score(all_labels, all_pred)
    print("sst acc=", sst_acc, 'mr_acc=', mr_acc, 'cr_acc=', cr_acc, 'all_acc=', all_acc)
    global best_all_acc
    if all_acc > best_all_acc:
        torch.save(model.state_dict(), "/home/tanghaihong/workspace/saved_model/Transfer-Learning"
                                       "/cross_task_roberta.pt", _use_new_zipfile_serialization=False)
        print("saving model to /home/tanghaihong/workspace/saved_model/Transfer-Learning"
              "/cross_task_roberta.pt")
        best_all_acc = all_acc


def train():
    model.train()
    model.to(device)
    losses = []
    print("Training")
    for epoch in range(n_epochs):
        print("epoch =", epoch)
        for ids, labels in tq(all_train_loader):
            ids = ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logit = model(ids)
            loss = loss_fn(logit, labels)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print("loss=", np.mean(losses))
        losses.clear()
        test()


train_batch_size = 32
test_batch_size = 64
gpu_num = 0
n_epochs = 30
device = torch.device("cuda:" + str(gpu_num))
model_name = "roberta-base"
print("Loading model")
config = RobertaConfig.from_pretrained(model_name)
model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
optimizer_to(optimizer, device)
loss_fn = nn.CrossEntropyLoss()
print("Loading data")
sst_train_df = pd.read_csv("/home/tanghaihong/workspace/"
                           "original/SST-2/train.tsv", sep="\t")
sst_dev_df = pd.read_csv("/home/tanghaihong/workspace/"
                         "original/SST-2/dev.tsv", sep="\t")
sst_test_df = pd.read_csv("/home/tanghaihong/workspace/"
                          "original/SST-2/test.tsv", sep="\t")
mr_train_df = pd.read_csv("/home/tanghaihong/workspace/"
                          "original/mr/train.csv", sep=",")
mr_test_df = pd.read_csv("/home/tanghaihong/workspace/"
                         "original/mr/test.csv", sep=",")
cr_train_df = pd.read_csv("/home/tanghaihong/workspace/"
                          "original/cr/train.csv", sep=",")
cr_test_df = pd.read_csv("/home/tanghaihong/workspace/"
                         "original/cr/test.csv", sep=",")
print("Tokenizing data")
tokenizer = RobertaTokenizer.from_pretrained(model_name)
sst_train_list = tokenize(tokenizer, sst_train_df, label_first=False)
sst_dev_list = tokenize(tokenizer, sst_dev_df, label_first=False)
sst_test_list = tokenize(tokenizer, sst_test_df, label_first=True)
mr_train_list = tokenize(tokenizer, mr_train_df)
mr_test_list = tokenize(tokenizer, mr_test_df)
cr_train_list = tokenize(tokenizer, cr_train_df)
cr_test_list = tokenize(tokenizer, cr_test_df)
sst_train_loader = DataLoader(dataset=MyDataset(sst_train_list), shuffle=True, batch_size=train_batch_size, collate_fn=collate_fn)
sst_dev_loader = DataLoader(dataset=MyDataset(sst_dev_list), shuffle=True, batch_size=test_batch_size, collate_fn=collate_fn)
sst_test_loader = DataLoader(dataset=MyDataset(sst_test_list), shuffle=True, batch_size=test_batch_size, collate_fn=collate_fn)
mr_train_loader = DataLoader(dataset=MyDataset(mr_train_list), shuffle=True, batch_size=train_batch_size, collate_fn=collate_fn)
mr_test_loader = DataLoader(dataset=MyDataset(mr_test_list), shuffle=True, batch_size=test_batch_size, collate_fn=collate_fn)
cr_train_loader = DataLoader(dataset=MyDataset(cr_train_list), shuffle=True, batch_size=train_batch_size, collate_fn=collate_fn)
cr_test_loader = DataLoader(dataset=MyDataset(cr_test_list), shuffle=True, batch_size=test_batch_size, collate_fn=collate_fn)
all_train_loader = DataLoader(dataset=MyDataset(sst_train_list + mr_train_list + cr_train_list), shuffle=True,
                              batch_size=train_batch_size, collate_fn=collate_fn)
all_test_loader = DataLoader(dataset=MyDataset(sst_test_list + mr_test_list + cr_test_list), shuffle=True,
                             batch_size=test_batch_size, collate_fn=collate_fn)
train()
