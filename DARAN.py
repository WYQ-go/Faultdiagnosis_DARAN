import torch
import torch.nn as nn
from Config import DataConfig as Config
from SeriesDataset import prepDataloader, TLTrainDataLoader
from tools import Log, mmd
import os
from sklearn.metrics import accuracy_score
import numpy as np
from loss.daan_loss import DAANLoss

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(Config.torch_seed)
torch.manual_seed(Config.torch_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.torch_seed)


class DARAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=32, padding=16, stride=2),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=3, stride=2),

            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=64, padding=32, stride=1),
            nn.BatchNorm1d(8),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=3, stride=2),

            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=96, padding=48, stride=1),
            nn.BatchNorm1d(8),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=3, stride=1),

            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=64, padding=32, stride=1),
            nn.ReLU(True),
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=32, padding=16, stride=1),
            nn.ReLU(True),

            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=16, padding=8, stride=2),
            nn.BatchNorm1d(8),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Flatten(start_dim=1),

            nn.Linear(200, 128),
            nn.ReLU(True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.25),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.25),
        )
        self.softmax = nn.Softmax(dim=1)
        self.lamb = 1.0
        self.ResidualBlock = nn.Sequential(
            nn.Linear(Config.class_num, Config.class_num),
            nn.ReLU(True),
            nn.Linear(Config.class_num, Config.class_num),
            nn.ReLU(True),
        )
        self.labelClassifer = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            nn.Linear(32, Config.class_num),
            nn.BatchNorm1d(Config.class_num),
            # nn.Softmax(dim=1)
        )
        self.classfierLoss = nn.CrossEntropyLoss()
        self.targetLoss = nn.CrossEntropyLoss()
        self.domainLoss = DAANLoss(Config.class_num, 1.0, max_iter=Config.n_epoches, use_lambda_scheduler=True)

    def forward(self, x_source=None, x_target=None, mode='train'):
        if mode == 'train':
            fcs = self.convolution(x_source)
            fct = self.convolution(x_target)
            fccs = self.labelClassifer(fcs)
            fcct = self.labelClassifer(fct)
            fs = self.ResidualBlock(fccs) + fccs
            targetPreds = self.softmax(fcct)
            sourcePreds = self.softmax(fs)
            return fcs, fct, sourcePreds, targetPreds
        elif mode == 'source':
            fcs = self.convolution(x_source)
            fccs = self.labelClassifer(fcs)
            fs = self.ResidualBlock(fccs) + fccs
            sourcePreds = self.softmax(fs)
            return sourcePreds
        elif mode == 'target':
            fct = self.convolution(x_target)
            fcct = self.labelClassifer(fct)
            targetPreds = self.softmax(fcct)
            return targetPreds

    def cal_loss(self, sourcePreds, sourceTargets, targetPreds, fcs, fct, gama):
        return self.classfierLoss(sourcePreds, sourceTargets) + \
               gama * self.classfierLoss(targetPreds, targetPreds) \
               + self.domainLoss.forward(fcs, fct, sourcePreds, targetPreds)


def train(tr_set, tt_set, model, device):
    optimizer = getattr(torch.optim, Config.optimizer)(model.parameters(), **Config.optim_hparas)
    min_accuracy = 0
    # logger = Log(Config.log_Path).getLog()
    #############################################
    # if Config.resume:
    #     if os.path.isfile(Config.save_path + 'point.pth'):
    #         check_point = torch.load(Config.save_path + os.path.basename(__file__)[:-3] + 'point.pth')
    #         model.load_state_dict(check_point['model'])
    #         optimizer.load_state_dict(check_point['optimizer'])
            # logger.info("=> loaded checkpoint (epoch {})".format(check_point['epoch']))
        # else:
            # logger.info("=> no checkpoint found")
    ##############################################
    i = 0
    for epoch in range(Config.n_epoches):
        model.train()
        loss_record = []
        for x_source, y_source, x_target in tr_set.__iter__():
            gamma = 2. / (1. + np.exp(-10.0 * epoch / Config.n_epoches)) - 1
            optimizer.zero_grad()
            x_source, y_source, x_target = x_source.to(device), y_source.to(device), x_target.to(device)
            fcs, fct, sourcePreds, targetPreds = model(x_source=x_source, x_target=x_target, mode='train')
            loss = model.cal_loss(sourcePreds, y_source, targetPreds, fcs, fct,gamma)
            loss.backward()
            optimizer.step()
            loss_record.append(loss.detach().cpu().item())
        model.domainLoss.update_dynamic_factor(4)
        test_pred, test_target = test(tt_set, model, device)
        train_pred, train_target = test_train(tr_set.__iter__(), model, device)
        accuracy_labeled = accuracy_score(train_pred, train_target)
        accuracy = accuracy_score(test_pred, test_target)
        check_point = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        # if epoch % 10 == 0:
        #         torch.save(check_point, Config.save_path + os.path.basename(__file__)[:-3] + 'point.pth')
        if accuracy > min_accuracy:
            # torch.save(check_point, Config.save_path + os.path.basename(__file__)[:-3] + 'opt_model.pth')
            min_accuracy = accuracy
            i = 0
        else:
            i += 1
        if i == Config.early_stopping:
            break
        print(epoch, np.mean(loss_record), accuracy_labeled, accuracy, min_accuracy, model.domainLoss.dynamic_factor)
    # logger.info(f'Train {os.path.basename(__file__)},optim:{Config.optim_hparas},'
    #             f'Dropout:{Config.dp},gama:{Config.gama}')
    # logger.info(f"model{os.path.basename(__file__)},End{epoch},loss{np.mean(loss_record)},optAccuracy{min_accuracy}")


def test(tt_set, model, device):
    model.eval()
    preds, target = [], []
    for x, y in tt_set:
        x = x.to(device)
        target.append(y)
        with torch.no_grad():
            pred = model(x_target=x, mode='target')
            preds.append(pred.detach().cpu())
    target = torch.cat(target, dim=0).numpy()
    preds = torch.cat(preds, dim=0).numpy()
    idx = preds.argmax(axis=1)
    out = np.zeros_like(preds, dtype=float)
    out[np.arange(preds.shape[0]), idx] = 1
    return out, target


def test_train(tt_set, model, device):
    model.eval()
    preds, target = [], []
    for x, y, _ in tt_set:
        x = x.to(device)
        target.append(y)
        with torch.no_grad():
            pred = model(x_source=x, mode='source')
            preds.append(pred.detach().cpu())
    target = torch.cat(target, dim=0).numpy()
    preds = torch.cat(preds, dim=0).numpy()
    idx = preds.argmax(axis=1)
    out = np.zeros_like(preds, dtype=float)
    out[np.arange(preds.shape[0]), idx] = 1
    return out, target


def dev(dev_set, model, device):
    model.eval()
    total_loss = 0
    for x, y in dev_set:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred, _ = model(x)
            loss = model.cal_loss(pred, y)
        total_loss += loss.detach().cpu().item()
    total_loss = total_loss / len(dev_set.dataset)
    return total_loss


if __name__ == '__main__':
    tr_set = TLTrainDataLoader(Config.Sfile_path, Config.TTRfile_path, Config.batch_size)
    st_set = prepDataloader(Config.STE_path, 'test', Config.batch_size)
    tt_set = prepDataloader(Config.TTEfile_path, 'test', Config.batch_size)
    print(Config.Sfile_path)
    print(Config.TTEfile_path)
    model = DARAN().to(Config.device)
    train(tr_set, tt_set, model, Config.device)
    pass
