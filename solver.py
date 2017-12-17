import progressbar as pbar
import torch
from torch.autograd import Variable
import numpy as np

class Solver(object):
    def __init__(self, optimizer, loss_func):
        self.optimizer = optimizer
        self.loss_func = loss_func

    def train(self, model, train_loader, epoch):
        # appearance of the progress bar
        widgets = [
            '  ',
            pbar.Percentage(),
            ' ',
            pbar.Bar(marker='█', left='|', right='|'),
            ' ',
            pbar.ETA(),
            " | ",
            pbar.DynamicMessage('loss'),
            '   '
        ]
        bar = pbar.ProgressBar(widgets=widgets, initial_value=1, max_value=len(train_loader))

        # start training
        for epoch_id in range(epoch):
            print('[Epoch %d/%d]' % (epoch_id+1, epoch))
            train_losses = []
            train_acc = []
            bar.start()
            for i, (inputs, targets) in enumerate(train_loader, 1):
                inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda() 
                self.optimizer.zero_grad()
                outputs = model(inputs)
                loss = 0
                for s in range(inputs.size()[0]):
                    #loss += criterion(outputs[s].view(23, -1).transpose(1, 0), targets[s].view(-1))
                    loss += self.loss_func(outputs[s].view(23, -1).transpose(1, 0), targets[s].view(-1))
                loss.backward()
                self.optimizer.step()
                _, preds = torch.max(outputs, 1)
                targets_mask = targets >= 0
                train_acc.append(np.mean((preds == targets)[targets_mask].data.cpu().numpy()))
                train_losses.append(loss.data[0] / inputs.size()[0])
                bar.update(i, loss=train_losses[-1])
            
            bar.finish()
            print('[Epoch %d/%d] TRAIN acc/loss: %.3f/%.3f\n' % (epoch_id + 1,
                                                               epoch,
                                                               train_acc[-1],
                                                               train_losses[-1]))

        return model
