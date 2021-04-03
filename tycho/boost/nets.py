from copy import deepcopy
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torch_optimizer

from tycho.util.torch import device
from tycho.util.log import log


class WeakNet(torch.nn.Module):
    def __init__(self, in_size, out_size, spot):
        super(WeakNet, self).__init__()
        self.spot = spot
        self.net = torch.nn.Sequential(
            #torch.nn.Linear(in_size, out_size),
            torch.nn.Linear(in_size, 3*(in_size+out_size)),
            torch.nn.SELU(),
            torch.nn.Linear(3*(in_size+out_size), 3*(in_size+out_size)),
            torch.nn.SELU(),
            torch.nn.Linear(3*(in_size+out_size), out_size),
        ).to(device)

        self.net.apply(self.init_weights)


    def init_weights(self, m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform(m.weight, gain=(3/4))
            m.bias.data.fill_(0.01)

    def forward(self, X):
        Y = self.net(X)
        return Y

class BoostNet():
    def __init__(self, target_type):
        self.weak_nets = []
        self.target_type = target_type

    def fit(self, folds):
        loaders = [DataLoader(dataset,batch_size=50,shuffle=True) for dataset in folds]
        sample_x, sample_y = folds[0][0]

        # Train a bunch of weaknets
        nr_nets = 10
        best_overall_loss = pow(2,63)
        for i in range(nr_nets):
            net_nr = i  + 1
            log.info(f'Training weaknet nr {net_nr}')
            weak_net = WeakNet(len(sample_x),len(sample_y),net_nr)


            if self.target_type in ['int', 'float']:
                loss_fn = torch.nn.MSELoss().to(device)
            elif self.target_type in ['category']:
                loss_fn = torch.nn.CrossEntropyLoss().to(device)

            scaler = GradScaler()

            if net_nr == nr_nets:
                optim = torch_optimizer.Ranger(weak_net.parameters(), lr=0.1)
                valid_indexes = [1]
            else:
                optim = torch_optimizer.Ranger(weak_net.parameters(), lr=0.3/(net_nr))
                valid_indexes = [0]

            train_loaders = []
            valid_loaders = []
            for i, loader in enumerate(loaders):
                if i in valid_indexes:
                    valid_loaders.append(loader)
                else:
                    train_loaders.append(loader)

            valid_loss_arr = []
            best_weak_net = None
            best_valid_loss = pow(2,63)
            for epoch in range(10000):
                #log.debug(f'Training weaknet nr {net_nr}, epoch: {epoch}')
                weak_net = weak_net.train()
                for loader in train_loaders:
                    for X, Y in loader:
                        X = X.to(device)
                        Y = Y.to(device)
                        optim.zero_grad()
                        with autocast():
                            Yp = weak_net(X)

                            if len(self.weak_nets) > 0:
                                Y_adj = Y - self.infer(X)
                            else:
                                Y_adj = Y

                            loss = loss_fn(Yp,Y_adj)

                        scaler.scale(loss).backward()
                        scaler.step(optim)
                        scaler.update()

                weak_net = weak_net.eval()

                total_loss = 0
                nr_samples = 0
                for loader in valid_loaders:
                    for X, Y in loader:
                        nr_samples += 1
                        X = X.to(device)
                        Y = Y.to(device)
                        with torch.no_grad():
                            with autocast():
                                Yp = weak_net(X)

                                if len(self.weak_nets) > 0:
                                    Yp = Yp + self.infer(X)

                                loss = loss_fn(Yp,Y)
                        total_loss += loss.item()

                total_loss = total_loss/nr_samples

                log.debug(f'Total loss: {total_loss}')
                if total_loss < best_valid_loss:
                    best_valid_loss = total_loss
                    best_weak_net = deepcopy(weak_net)

                if total_loss < 0.00001:
                    log.info('Loss to small to matter, stopping.')
                    break
                valid_loss_arr.append(total_loss)

                if len(valid_loss_arr) > 4:
                    if valid_loss_arr[-1] > np.mean(valid_loss_arr[-3:-1]):
                        log.debug('Stopping the trainig of weaknet, valid loss not improving')
                        break
            if best_valid_loss < best_overall_loss:
                best_overall_loss = best_valid_loss
                log.info(f'Appending weaknet to ensemble after {epoch} epochs')
                self.weak_nets.append(best_weak_net)
            else:
                log.warning(f'Ignoring this weaknet, it had a total loss of {best_valid_loss}, compared to the best overall of: {best_overall_loss}')


        self.net = weak_net

    def infer(self, X):
        with autocast():
            with torch.no_grad():
                X = X.to(device)
                Yp = None
                for weak_net in self.weak_nets:
                    if Yp is None:
                        Yp = weak_net(X)
                    else:
                        Yp = Yp + weak_net(X)
                return Yp
