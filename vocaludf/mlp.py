import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

# From https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#starter-example
class MLP(pl.LightningModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_features)
            )
        # self.hidden = nn.Linear(in_features, 128)
        # self.nonlinear = nn.ReLU()
        # self.l1 = nn.Linear(hidden_features, out_features)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        return x # x are the model's logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        return self.softmax(logits)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=3e-4)
