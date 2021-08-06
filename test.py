import json
import pandas as pd
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from tqdm import tqdm
import time
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import torch.nn.functional as F
import random
import os


# data = pd.read_csv("./data/sentihood/bert-pair/train_NLI_M.tsv", sep="\t")#.values
# print(data.info())


# str_1 = ["my","you"]
# str_2 = ["he","she"]
# labels = torch.tensor([1,0])
# model_name = "bert-base-uncased"
# tokenizer = BertTokenizer.from_pretrained(model_name)
# config = BertConfig(num_labels=2)
# model = BertForSequenceClassification.from_pretrained(model_name, config = config)
# result = tokenizer(str_1, str_2, padding=True)
# input_ids, token_type_ids, attention_mask = result["input_ids"], result["token_type_ids"], result["attention_mask"]
# print(input_ids)
# print(attention_mask)
# print(token_type_ids)
# input_ids = torch.tensor(input_ids)
# token_type_ids = torch.tensor(token_type_ids)
# attention_mask = torch.tensor(attention_mask)
# outputs = model(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, labels=labels)
# print(outputs[0])
# print(outputs[1])

# class My_Input():
#     def __init__(self,x,y):
#         self.x = x
#         self.y = y
#
# class My_Dataset(Dataset):
#     def __init__(self):
#         super(My_Dataset, self).__init__()
#         self.data = []
#         for i in range(10):
#             self.data.append(My_Input(i,2*i))
#
#     def __getitem__(self, item):
#          return self.data[item]
#
#     def __len__(self):
#         return len(self.data)
#
# def prepare_dataLoader():
#     my_dataset = My_Dataset()
#     return DataLoader(my_dataset, shuffle=False, batch_size=2)
#
# dataLoader = prepare_dataLoader()
# for data in dataLoader:
#     print(data[0])

class A:
    def __init__(self):
        self.b = 2

a = A()
print(a.b)