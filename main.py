from transformers import BertForSequenceClassification, BertConfig
import argparse
import random
import numpy as np
from head import *
import data_process
import utilities
import torch.nn.functional as F
import torch.nn as nn
import os


def validation_result(val_dataLoader, model):
    model.eval()
    total_loss, test_accuracy = 0, 0
    nb_test_steps, nb_test_examples = 0, 0
    for i, data in enumerate(val_dataLoader):
        input_ids, attention_mask, \
        token_type_ids, label_ids = \
            data[0].to(device), data[1].to(device), \
            data[2].to(device), data[3].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, labels=label_ids)
            loss, logits = outputs[0], outputs[1]
        logits = F.softmax(logits, dim=-1)
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        outputs = np.argmax(logits, axis=1)
        tmp_test_accuracy = np.sum(outputs == label_ids)
        test_accuracy += tmp_test_accuracy
        #loss = sum(loss) / 2
        total_loss += loss.item()
        nb_test_examples += input_ids.size(0)
        nb_test_steps += 1
    total_loss = total_loss / nb_test_steps
    test_accuracy = test_accuracy / nb_test_examples
    return total_loss, test_accuracy


def train(train_dataLoader, dev_dataLoader, model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_record, val_accuracy = [], []
    min_val_loss = 1e3
    early_step = 0
    epoch = 0
    model.to(device)
    #model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])
    for _ in range(args.epoch_size):
        model.train()
        for data in train_dataLoader:
            input_ids, attention_mask,\
            token_type_ids, label_ids =\
                data[0].to(device), data[1].to(device),\
                data[2].to(device), data[3].to(device)
            outputs = model(input_ids = input_ids, attention_mask = attention_mask,
                                  token_type_ids = token_type_ids, labels = label_ids)
            loss = outputs[0]
            optimizer.zero_grad()
            #loss = sum(loss) / 2
            loss.backward()
            optimizer.step()

        print(epoch,"nice")
        # Testing the accuracy of validation data every epoch.
        val_loss, temp_accuracy = validation_result(dev_dataLoader, model)
        loss_record.append(val_loss)
        val_accuracy.append(temp_accuracy)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(),model_path)
            early_step = 0
        else:
            early_step += 1
        epoch += 1
        if early_step > args.early_step_thrshd:
            break
    print('Finished training after {} epochs'.format(epoch))
    print(min_val_loss)
    utilities.plot_learning_curve(loss_record,val_accuracy, args,title = 'train')
    return min_val_loss, loss_record

def test(output_path, model, test_dataLoader):
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    with open(output_path, "w") as f_pred:
        for data in test_dataLoader:
            input_ids, attention_mask, \
            token_type_ids, label_ids = \
                data[0].to(device), data[1].to(device), \
                data[2].to(device), data[3].to(device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids, labels=label_ids)
                logits = outputs[1]
            #logits = logits.detach().cpu().numpy()
            pred = F.softmax(logits, dim=-1)
            pred_index = pred.argmax(dim=1).tolist()
            pred = pred.tolist()
            for i in range(len(pred_index)):
                f_pred.write(str(pred_index[i]))
                for item in pred[i]:
                    f_pred.write(" " + str(item))
                f_pred.write("\n")

def main():
    # 0_准备
    # 命令行解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', default='sentihood_NLI_M', type=str,
                        choices=["sentihood_single", "sentihood_NLI_M", "sentihood_QA_M", \
                                 "sentihood_NLI_B", "sentihood_QA_B", "semeval_single", \
                                 "semeval_NLI_M", "semeval_QA_M", "semeval_NLI_B", "semeval_QA_B"]
                        )
    parser.add_argument("--output_dir", default="./results/")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--epoch_size", default=6, type = int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--early_stop", default=5, type = int)
    parser.add_argument("--early_step_thrshd", default=10, type=int)
    parser.add_argument("--max_seq_len", default=200, type = int)
    parser.add_argument("--lr", default = 2e-5, type = float)
    args = parser.parse_args()
    # 设置种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 1_预处理
    # 准备数据
    _, train_dataLoader = data_process.prep_dataloader(name = args.task_name,
                                                    batch_size=args.batch_size,
                                                    mode = "train",
                                                    max_seq_len=args.max_seq_len)
    _, dev_dataLoader = data_process.prep_dataloader(name = args.task_name,
                                                  batch_size=args.batch_size,
                                                  mode = "dev",
                                                  max_seq_len=args.max_seq_len)
    # 定义模型部分
    config = BertConfig(num_labels=3)
    model = BertForSequenceClassification.from_pretrained(model_name, config=config)
    # 2_训练
    train(train_dataLoader, dev_dataLoader, model, args)

    # 3_测试&存储预测结果
    _, test_dataLoader = data_process.prep_dataloader(name=args.task_name,
                                                                 batch_size=args.batch_size,
                                                                 mode="test",
                                                                 max_seq_len=args.max_seq_len)
    output_path = args.output_dir + args.task_name + ".txt"
    test(output_path, model, test_dataLoader)
if __name__ == '__main__':
    main()