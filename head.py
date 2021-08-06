import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_name = 'bert-base-uncased'
model_path = "./model_parameters/model_state"
#gpus = [3,1]