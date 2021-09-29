import os
import re
import yaml
from copy import deepcopy
from typing import Any, Dict, Optional, Union
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import _logger as log
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn, rank_zero_info
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class Checkpoint(Callback):
    def __init__(self, evaluate_fn, predict_fn, dev_data, y_true, *args, filepath=None, **kwargs):
        self.best = 0
        self.evaluate_fn = evaluate_fn
        self.predict_fn = predict_fn
        self.dev_data = dev_data
        self.y_true = y_true
        self.filepath = filepath or 'best_model'
        self.args = args
        self.kwargs = kwargs

    def evaluate(self, y_pred):
        return self.evaluate_fn(self.y_true, y_pred)

    def on_epoch_end(self, trainer, pl_module):
        y_pred = self.predict_fn(pl_module, self.dev_data, *self.args, **self.kwargs)
        metrics = self.evaluate(y_pred)
        epoch = trainer.current_epoch
        if metrics > self.best:
            if epoch != 0:
                trainer.save_checkpoint(self.filepath, weights_only=True)
            self.best = metrics
        pl_module.print('Epoch {}, best metrics {}'.format(epoch, self.best))
