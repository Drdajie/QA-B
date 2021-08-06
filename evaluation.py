import argparse

def get_y_pred(task_name, pred_data_dir):
    pred, score = [], []
    if task_name in ["sentihood_NLI_B", "sentihood_QA_B"]:
        with open(pred_data_dir,"r",encoding="utf-8") as f:
            s=f.readlines()
            for i in range(len(s)):
                temp = s[i]
                temp = temp.strip().split()
                pred.append(int(temp[0]))
                score.append([float(temp[1]), float(temp[2]), float(temp[3])])
    return pred, score

def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("--task_name", default="sentihood_NLI_M",
                       type = str,
                       choices = ["sentihood_single", "sentihood_NLI_M", "sentihood_QA_M", \
                                 "sentihood_NLI_B", "sentihood_QA_B", "semeval_single", \
                                 "semeval_NLI_M", "semeval_QA_M", "semeval_NLI_B", "semeval_QA_B"])
    parse.add_argument("--pred_data_dir", default="")
    args = parse.parse_args()
    y_pred, _ = get_y_pred(args.task_name, args.pred_data_dir)

if __name__ == "__main__":
    main()