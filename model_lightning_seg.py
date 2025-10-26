import lightning as L
import torch
from torch import nn
import utils
import torchmetrics
from models_arq.att_unet_seg import attention_unet_seg
from loss import TopologicalPoolingLoss
from torchmetrics.classification import BinaryJaccardIndex, BinaryPrecision, BinaryAccuracy, Dice
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassPrecision, MulticlassAccuracy


class MyModel(L.LightningModule):
    def __init__(self, model_opts, train_par):
        super().__init__()
        model_def = globals()[model_opts.name]
        self.model = model_def(input_channels=1)

        self.train_par = train_par
        self.eval_threshold = train_par.eval_threshold
        self.num_classes=train_par.num_classes
        self.init_metrics('val')
        self.init_metrics('test')

    def init_metrics(self,stage):

        if self.num_classes == 2:           
            metrics = {
                'Dice (lightning AI)': Dice(average='micro',threshold=self.eval_threshold),
                'IoU (lightning AI)': BinaryJaccardIndex(threshold=self.eval_threshold),
                'precision': BinaryPrecision(threshold=self.eval_threshold),
                'accuracy': BinaryAccuracy(threshold=self.eval_threshold),
            }  
        else:
            metrics = {
                'Dice (lightning AI)': Dice(average='micro',threshold=self.eval_threshold),
                'IoU (lightning AI)': MulticlassJaccardIndex(num_classes=self.num_classes),
                'precision': MulticlassPrecision(num_classes=self.num_classes),
                'accuracy': MulticlassAccuracy(num_classes=self.num_classes),
            }  
        
        for metric_name, metric in metrics.items():
            setattr(self, f"{stage}_{metric_name}", metric)

    def log_metrics(self, stage, pred, label):
        metrics = ['Dice (lightning AI)', 'IoU (lightning AI)', 'precision', 'accuracy']
        for metric in metrics:
            metric_fn = getattr(self, f"{stage}_{metric}")
            metric_fn(pred, label)
            self.log(f"{stage}_{metric}", metric_fn, on_step=False, on_epoch=True)
        
        dice_metric,_=utils.dice_coeff_batch(pred.float(), label.float())
        self.log(f"{stage}_Dice (formula)", dice_metric, on_step=False, on_epoch=True)
            
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        img, label = batch
        pred_mask = self(img)
        label = (label >= self.eval_threshold).to(torch.int64)
        loss = self.get_loss(pred_mask.float(), label.float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        pred_mask = self(img)
        label = (label >= self.eval_threshold).to(torch.int64)
        loss = self.get_loss(pred_mask.float(), label.float())
        self.log('val_loss', loss)
        pred_prob = torch.sigmoid(pred_mask)
        pred = (pred_prob >= self.eval_threshold).to(torch.int64)
        self.log_metrics('val', pred, label)
    
    def test_step(self, batch, batch_idx):
        img, label = batch
        pred_mask = self(img)
        label = (label >= self.eval_threshold).to(torch.int64)
        loss = self.get_loss(pred_mask.float(), label.float())
        self.log('test_loss', loss)
        pred_prob = torch.sigmoid(pred_mask)
        pred = (pred_prob >= self.eval_threshold).to(torch.int64)
        self.log_metrics('test', pred, label)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        img, label = batch
        pred_mask = self(img)
        pred_prob = torch.sigmoid(pred_mask)
        pred = (pred_prob >= self.eval_threshold).to(torch.float)
        label = (label >= self.eval_threshold).to(torch.float)

        return pred


    def predict_step_aux(self, batch, batch_idx, dataloader_idx=None):
        img, label = batch
        pred_mask = self(img)
        pred_prob = torch.sigmoid(pred_mask)
        pred = (pred_prob >= self.eval_threshold).to(torch.float)
        label = (label >= self.eval_threshold).to(torch.float)
        return pred, pred_prob, pred_mask, label
    
    def configure_optimizers(self):
        optimizer=torch.optim.AdamW(self.parameters(), lr=self.train_par.lr, weight_decay=self.train_par.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def get_loss(self, y_hat, y):
        if self.train_par.loss_opts.name == 'TopologicalPoolingLoss':
            return self.loss_f(y_hat, y)         
