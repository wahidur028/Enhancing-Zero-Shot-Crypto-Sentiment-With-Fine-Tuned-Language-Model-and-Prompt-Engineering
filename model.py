import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForSequenceClassification
from torchmetrics.functional.classification import (
    binary_accuracy, binary_f1_score, binary_recall, binary_precision, binary_confusion_matrix
)

class MyModel(pl.LightningModule):
    def __init__(self, model_ckp, num_labels, learning_rate, loss_fn):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_ckp)
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.loss_function = loss_fn
           
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
    
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, label = batch
        outputs = self(input_ids, attention_mask)
        loss = self.loss_function(outputs, label)
        preds = torch.argmax(outputs, dim=1)
        accuracy = binary_accuracy(preds, label)
        f1_score = binary_f1_score(preds, label)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_accuracy", accuracy, prog_bar=True, logger=True)
        self.log("train_f1", f1_score, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, label = batch
        outputs = self(input_ids, attention_mask)
        loss = self.loss_function(outputs, label)
        preds = torch.argmax(outputs, dim=1)
        accuracy = binary_accuracy(preds, label)
        f1_score = binary_f1_score(preds, label)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_accuracy", accuracy, prog_bar=True, logger=True)
        self.log("val_f1", f1_score, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
