import os
import pandas as pd
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import head
from tqdm import tqdm

def prep_dataloader(name, mode, batch_size, max_seq_len, n_jobs=1):
    ''' Generates a dataset, then is put into a dataloader. '''
    dataset_dict = {"sentihood_NLI_M":Sentihood_NLI_M_Dataset}
    dataset = dataset_dict[name](mode=mode, max_seq_length = max_seq_len)
    dataLoader = DataLoader(dataset, batch_size, shuffle=(mode=="train"))
                            # drop_last=False,
                            # num_workers=n_jobs,
                            # pin_memory=True)
    return dataset, dataLoader


def convert_examples_to_features(examples, max_seq_length, label_list):
    """Loads a data file into a list of `InputBatch`s."""

    tokenizer = BertTokenizer.from_pretrained(head.model_name)
    first_seq = examples[:,1].tolist()
    second_seq = examples[:,2].tolist()#.lower()

    label_map = {}
    label_ids = []
    for (i, label) in enumerate(label_list):
        label_map[label.lower()] = i

    result = tokenizer(first_seq, second_seq, max_length = max_seq_length, padding=True)
    input_ids, token_type_ids, attention_mask = result["input_ids"], result["token_type_ids"], result["attention_mask"]
    for (ex_index, example) in enumerate(tqdm(examples)):
        label_id = label_map[example[3].lower()]
        label_ids.append(label_id)
    label_ids = torch.tensor(label_ids)
    input_ids = torch.tensor(input_ids)
    token_type_ids = torch.tensor(token_type_ids)
    attention_mask = torch.tensor(attention_mask)
    label_ids = torch.tensor(label_ids)
    return  input_ids, attention_mask, token_type_ids,  label_ids


class Sentihood_NLI_M_Dataset(Dataset):
    def __init__(self, mode, max_seq_length):
        '''
        将数据读到内存中。
        :param type: 任务类型：“single”, "NLI_M", "QA_M", "NLI_B", "QA_B"
        :param mode: 数据集类型："train", "dev", test
        '''
        super(Sentihood_NLI_M_Dataset, self)
        file_path = "./data/sentihood/bert-pair/{}_NLI_M.tsv".format(mode)
        data = pd.read_csv(file_path, sep="\t").values
        self.input_ids, self.attention_mask, self.token_type_ids, self.label_ids = \
            convert_examples_to_features(data, max_seq_length,self.get_labels())

    def get_labels(self):
        """See base class."""
        return ['None', 'Positive', 'Negative']

    def __getitem__(self, item):
        return self.input_ids[item], self.attention_mask[item],\
                self.token_type_ids[item], self.label_ids[item]

    def __len__(self):
        return len(self.label_ids)
