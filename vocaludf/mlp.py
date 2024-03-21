import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

# From https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#starter-example
class MLP(pl.LightningModule):
    def __init__(self, in_features, out_features, logger, weight=None):
        super().__init__()
        self.my_logger = logger
        self.weight = weight
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

        self.val_acc = torchmetrics.classification.BinaryAccuracy()
        self.val_f1 = torchmetrics.classification.BinaryF1Score()
        self.test_acc = torchmetrics.classification.BinaryAccuracy()
        self.test_f1 = torchmetrics.classification.BinaryF1Score()

    def forward(self, x):
        x = self.model(x)
        return x # x are the model's logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, self.weight)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        self.my_logger.info(f'train_loss: {self.trainer.callback_metrics["train_loss"]}')

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, self.weight)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        y_pred_probs = self.softmax(logits)
        y_pred = torch.argmax(y_pred_probs, axis=1)
        self.val_acc.update(y_pred, y)
        self.val_f1.update(y_pred, y)

    def on_validation_epoch_end(self):
        val_acc = self.val_acc.compute()
        val_f1 = self.val_f1.compute()
        self.log('val_acc', val_acc)
        self.log('val_f1', val_f1)
        self.val_acc.reset()
        self.val_f1.reset()
        self.my_logger.info(f'val_loss: {self.trainer.callback_metrics["val_loss"]}')
        self.my_logger.info(f'val_acc: {val_acc}, val_f1: {val_f1}')

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y_pred_probs = self.softmax(logits)
        y_pred = torch.argmax(y_pred_probs, axis=1)
        self.test_acc.update(y_pred, y)
        self.test_f1.update(y_pred, y)

    def on_test_epoch_end(self):
        test_acc = self.test_acc.compute()
        test_f1 = self.test_f1.compute()
        self.log('test_acc', test_acc)
        self.log('test_f1', test_f1)
        self.test_acc.reset()
        self.test_f1.reset()
        self.my_logger.info(f'test_acc: {test_acc}, test_f1: {test_f1}')

    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        return self.softmax(logits)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=3e-4)
