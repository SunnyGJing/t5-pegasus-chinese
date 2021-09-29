import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import datetime
from pytorch_lightning.core.decorators import auto_move_data


class LightModel(pl.LightningModule):

    def __init__(self, net, optimizer, loss_fn=None):
        super().__init__()
        self.net = net
        self.save_hyperparameters("net")
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def configure_optimizers(self):
        return self.optimizer

    @auto_move_data
    def forward(self, batch):
        with torch.no_grad():
            return self.net(batch)

    def training_step(self, batch, batch_idx):
        logits = self.net(batch)
        loss = self.net.compute_loss(logits, batch)
        return loss

    def print_bar(self):
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.print("\n" + "=" * 80 + "%s" % nowtime)

    def predict(self, data):
        self.net.eval()
        ret = []
        for batch in data:
            res = self(batch)
            if self.on_gpu:
                res = res.cpu()
            res = res.numpy().tolist()
            ret.extend(res)
        return ret
