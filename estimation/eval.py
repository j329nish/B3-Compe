import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import cohen_kappa_score
import torch.nn.functional as F

# 検証データのパス
val_text_path = 'dataset/dev/dev.txt'
val_label_path = 'dataset/dev/dev.label'

# ファイルからデータを読み込む関数
def load_data(text_path, label_path):
    with open(text_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    with open(label_path, 'r', encoding='utf-8') as f:
        labels = f.readlines()
    
    texts = [text.strip() for text in texts]
    labels_L, labels_H = [], []
    for label in labels:
        l1, l2 = map(int, label.strip().split("\t"))
        labels_L.append(min(l1, l2))
        labels_H.append(max(l1, l2))

    return texts, labels_L, labels_H

# データの読み込み
val_texts, val_labels_L, val_labels_H = load_data(val_text_path, val_label_path)

# トークナイザーの読み込み
model_name = "xlm-roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
max_length = 128

def make_dataset(tokenizer, max_length, texts, labels_L, labels_H):
    dataset_for_loader = list()
    for text, label_L, label_H in zip(texts, labels_L, labels_H):
        encoding = tokenizer(text, max_length=max_length, padding="max_length", truncation=True)
        encoding["label_L"] = label_L - 1
        encoding["label_H"] = label_H - 1
        encoding = {key: torch.tensor(value) for key, value in encoding.items()}
        dataset_for_loader.append(encoding)
    return dataset_for_loader

# 検証データの前処理
dataset_val = make_dataset(tokenizer, max_length, val_texts, val_labels_L, val_labels_H)
dataloader_val = DataLoader(dataset_val, batch_size=256, shuffle=False)

# ベストモデルのパス
best_model_path = "./model1/epoch=1-step=876.ckpt"

def get_gold_labels(predictions, lower_labels, higher_labels):
    if np.sum(predictions == lower_labels) >= np.sum(predictions == higher_labels):
        gold_labels = lower_labels.copy()
        gold_labels[predictions == higher_labels] = higher_labels[predictions == higher_labels]
    else:
        gold_labels = higher_labels.copy()
        gold_labels[predictions == lower_labels] = lower_labels[predictions == lower_labels]
    return gold_labels

class Bert4Classification(pl.LightningModule):
    def __init__(self, model_name, num_labels, lr):
        super().__init__()
        self.save_hyperparameters()
        self.bert_sc = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # 訓練ステップ
    def training_step(self, batch, batch_idx):
        labels_L = batch.pop("label_L")  
        labels_H = batch.pop("label_H")  
        labels = labels_H.to(self.device).long().view(-1)
        output = self.bert_sc(**batch)
        logits = output.logits
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss)
        return loss

    # 検証ステップ
    def validation_step(self, batch, batch_idx):
        labels_L = batch.pop("label_L")
        labels_H = batch.pop("label_H")
        labels = labels_H.to(self.device).long().view(-1)
        output = self.bert_sc(**batch)
        logits = output.logits
        loss = F.cross_entropy(logits, labels)
        labels_predicted = logits.argmax(dim=-1).cpu().numpy()
        gold_labels = get_gold_labels(labels_predicted, labels_L.cpu().numpy(), labels_H.cpu().numpy())
        qwk = cohen_kappa_score(labels_predicted, gold_labels, weights='quadratic')
        self.log("val_loss", loss)
        self.log("val_qwk", qwk)

    # テストステップ
    def test_step(self, batch, batch_idx):
        labels_L = batch.pop("label_L")
        labels_H = batch.pop("label_H")
        labels = labels_H.to(self.device).long().view(-1)
        output = self.bert_sc(**batch)
        logits = output.logits
        labels_predicted = logits.argmax(dim=-1).cpu().numpy()
        gold_labels = get_gold_labels(labels_predicted, labels_L.cpu().numpy(), labels_H.cpu().numpy())
        qwk = cohen_kappa_score(labels_predicted, gold_labels, weights='quadratic')
        self.log("test_qwk", qwk)

    # 最適化手法
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        
# モデルのロード
model = Bert4Classification.load_from_checkpoint(best_model_path, model_name=model_name, num_labels=6, lr=1e-5)

# Trainerの作成
trainer = pl.Trainer(accelerator="gpu", devices=1)

# 検証データで評価
trainer.validate(model, dataloaders=dataloader_val)
