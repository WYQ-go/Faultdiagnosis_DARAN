import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function




class LambdaSheduler(nn.Module):
    def __init__(self, gamma=1.0, max_iter=1500, **kwargs):
        super(LambdaSheduler, self).__init__()
        self.gamma = gamma
        self.max_iter = max_iter
        self.curr_iter = 0

    def lamb(self):
        p = self.curr_iter / self.max_iter
        lamb = 2. / (1. + np.exp(-self.gamma * p)) - 1
        return lamb

    def step(self):
        self.curr_iter = min(self.curr_iter + 1, self.max_iter)


class AdversarialLoss(nn.Module):
    '''
    Acknowledgement: The adversarial loss implementation is inspired by http://transfer.thuml.ai/
    '''

    def __init__(self, gamma=1.0, max_iter=1000, use_lambda_scheduler=True, **kwargs):
        super(AdversarialLoss, self).__init__()
        self.domain_classifier = Discriminator()
        self.use_lambda_scheduler = use_lambda_scheduler
        if self.use_lambda_scheduler:
            self.lambda_scheduler = LambdaSheduler(gamma, max_iter)
        self.lambd = 1.0

    def forward(self, source, target, gamma=None):
        lamb = 1.0
        if self.use_lambda_scheduler:
            lamb = self.lambda_scheduler.lamb()
            self.lambda_scheduler.step()
        source_loss = self.get_adversarial_result(source, True, lamb, gamma)
        target_loss = self.get_adversarial_result(target, False, lamb)
        adv_loss = (source_loss + target_loss)
        return adv_loss

    def forward2(self, source, target, t, gamma=None):
        if self.use_lambda_scheduler and t == 1:
            self.lamb = self.lambda_scheduler.lamb()
            self.lambda_scheduler.step()
        source_loss = self.get_adversarial_result(source, source=True, lamb=self.lamb, gamma=gamma)
        target_loss = self.get_adversarial_result(target, False, self.lamb)
        adv_loss = (len(source) * source_loss + len(target) * target_loss) / (len(source) + len(target))
        return adv_loss

    def forward_smooth(self, source, target, t, epoch_ratio, gamma=None):
        if self.use_lambda_scheduler and t == 1:
            self.lamb = self.lambda_scheduler.lamb()
            self.lambda_scheduler.step()

        smooth_param = 1.0 - 0.5 * epoch_ratio
        source_loss = self.get_smooth_ad_result(source, smooth_param, domain_label=1,
                                                domain_num=2, lamb=self.lamb, gamma=gamma)
        target_loss = self.get_smooth_ad_result(target, smooth_param, domain_label=1,
                                                domain_num=2, lamb=self.lamb, gamma=gamma)
        adv_loss = (len(source) * source_loss + len(target) * target_loss) / (len(source) + len(target))
        return adv_loss

    def forward_smooth2(self, source, target, t, epoch_ratio, gamma=None):
        if self.use_lambda_scheduler and t == 1:
            self.lamb = self.lambda_scheduler.lamb()
            self.lambda_scheduler.step()

        smooth_param = 1.0 - 0.5 * epoch_ratio


    def get_adversarial_result(self, x, source=True, lamb=1.0, gamma=None):
        x = ReverseLayerF.apply(x, lamb)
        domain_pred = self.domain_classifier(x)
        device = domain_pred.device
        if source:
            domain_label = torch.ones(len(x), 1).long().to(device)
        else:
            domain_label = torch.zeros(len(x), 1).long().to(device)
        try:
            loss_fn = nn.BCELoss(weight=gamma)
            loss_adv = loss_fn(domain_pred, domain_label.float().to(device))
        except:
            loss_adv = 0
        return loss_adv

    def get_smooth_ad_result(self, x, smooth_param, domain_label, domain_num, lamb=1.0, gamma=None):
        x = ReverseLayerF.apply(x, lamb)
        domain_pred = self.domain_classifier(x)
        device = domain_pred.device
        smooth_domain_label = torch.zeros(len(x), domain_num).long().to(device) + (1 - smooth_param) / (domain_num - 1)
        smooth_domain_label[:, domain_label] = smooth_param
        try:
            loss_fn = nn.CrossEntropyLoss()
            loss_adv = loss_fn(domain_pred, smooth_domain_label)
        except:
            loss_adv = 0
        return loss_adv


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class Discriminator(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=32, hidden_dim2=16):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        ]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
