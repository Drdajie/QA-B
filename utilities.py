import matplotlib.pyplot as plt
import torch
from head import device

def plot_learning_curve(loss_record, accuracy, args, title=''):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    total_steps = len(loss_record)
    temp = 15008 // args.batch_size
    x_1 = [temp * i for i in range(len(loss_record))]
    x_2 = x_1
    plt.figure(figsize=(6, 4))
    plt.plot(x_1, accuracy, c="red", label = "val_accuracy")
    plt.plot(x_2, loss_record, c='tab:cyan', label='val_loss')
    #plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('CrossEntry loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()


def plot_pred(dv_set, model, lim=35., preds=None, targets=None):
    ''' Plot prediction of your DNN '''
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    plt.figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()