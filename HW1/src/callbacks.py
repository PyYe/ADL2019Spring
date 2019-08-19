import math
import json
import torch
torch.cuda.manual_seed_all(518)
import numpy as np
from metrics import Recall


class Callback:
    def __init__():
        pass

    def on_epoch_end(log_train, log_valid, model):
        pass


class MetricsLogger(Callback):
    def __init__(self, log_dest):
        self.history = {
            'train': [],
            'valid': []
        }
        self.log_dest = log_dest

    def on_epoch_end(self, log_train, log_valid, model):
        log_train['epoch'] = model.epoch
        log_valid['epoch'] = model.epoch
        self.history['train'].append(log_train)
        self.history['valid'].append(log_valid)
        with open(self.log_dest, 'w') as f:
            json.dump(self.history, f, indent='    ')

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, filepath,
                 monitor='loss',
                 verbose=0,
                 mode='min', patience=7):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self._filepath = filepath
        self._verbose = verbose
        self._monitor = monitor
        self._best = math.inf if mode == 'min' else - math.inf
        self._mode = mode
        self._best_epoch = 0
        
        self.counter = 0
        self.early_stop = False

    def on_epoch_end(self, log_train, log_valid, model):

        score = log_valid[self._monitor]
        if self._mode == 'min':
            if score < self._best:
                self._best_epoch = model.epoch
                self.counter = 0
                self.save_checkpoint(score, model)
                self._best = score
            else:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
        elif self._mode == 'max':
            if score > self._best:
                self._best_epoch = model.epoch
                self.counter = 0
                self.save_checkpoint(score, model)
                self._best = score
            else:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience} (Best score : {self._best} ; best epoch : {self._best_epoch})')
                if self.counter >= self.patience:
                    self.early_stop = True
        else:
            print ('unknown mode!')
    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        if self._verbose:
            print(self._monitor,'improved : ', self._best, ' --> ', score, ' Saving model ...')
        model.save('{}.{}'
                       .format(self._filepath, 'best'))
        print('Best model saved (%s : %f)' % (self._monitor, score))
        self._best = score
        
class ModelCheckpoint(Callback):
    def __init__(self, filepath,
                 monitor='loss',
                 verbose=0,
                 mode='min'):
        self._filepath = filepath
        self._verbose = verbose
        self._monitor = monitor
        self._best = math.inf if mode == 'min' else - math.inf
        self._mode = mode

    def on_epoch_end(self, log_train, log_valid, model):
        score = log_valid[self._monitor]
        if self._mode == 'min':
            if score < self._best:
                self._best = score
                model.save(self._filepath)
                if self._verbose > 0:
                    print('Best model saved (%f)' % score)

        elif self._mode == 'max':
            if score > self._best:
                self._best = score
                model.save(self._filepath)
                if self._verbose > 0:
                    print('Best model saved (%f)' % score)

        elif self._mode == 'all':
            model.save('{}.{}'
                       .format(self._filepath, model.epoch))
