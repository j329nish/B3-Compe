# ====================
# ライブラリの読み込み
# ====================

import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import cohen_kappa_score
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ファイルのパスを設定
train_text_path = 'dataset/train/train.txt'
train_label_path = 'dataset/train/train.label'
val_text_path = 'dataset/dev/dev.txt'
val_label_path = 'dataset/dev/dev.label'

# ファイルからデータを読み込む関数（ラベルの大小を自動判定）
def load_data(text_path, label_path):
    with open(text_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    with open(label_path, 'r', encoding='utf-8') as f:
        labels = f.readlines()
    
    texts = [text.strip() for text in texts]

    # ラベルを2つの値に分割し、大小関係を自動で判定
    labels_L, labels_H = [], []
    for label in labels:
        l1, l2 = map(int, label.strip().split("\t"))  # タブ区切りで取得
        labels_L.append(min(l1, l2))  # 小さい方を低いラベル
        labels_H.append(max(l1, l2))  # 大きい方を高いラベル

    return texts, labels_L, labels_H

# データの読み込み
train_texts, train_labels_L, train_labels_H = load_data(train_text_path, train_label_path)
val_texts, val_labels_L, val_labels_H = load_data(val_text_path, val_label_path)

# データセットの辞書形式での保持
dataset_dict = {
    "train": Dataset.from_dict({
        "text": train_texts, 
        "label_L": train_labels_L, 
        "label_H": train_labels_H
    }),
    "validation": Dataset.from_dict({
        "text": val_texts, 
        "label_L": val_labels_L, 
        "label_H": val_labels_H
    })
}

# データセットの辞書をDatasetDictに変換
dataset = DatasetDict(dataset_dict)

# データセットの確認
print(f"Training data sample: {dataset['train'][0]}")
print(f"Validation data sample: {dataset['validation'][0]}")
print(dataset["train"].features)

# ====================
# 前処理
# ====================

# 単語分割器の読み込み
model_name = "xlm-roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 最大文長の設定
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

dataset_train = make_dataset(
    tokenizer, max_length, 
    [dataset["train"][i]["text"] for i in range(len(dataset["train"]))], 
    [dataset["train"][i]["label_L"] for i in range(len(dataset["train"]))],
    [dataset["train"][i]["label_H"] for i in range(len(dataset["train"]))],
)

dataset_val = make_dataset(
    tokenizer, max_length, 
    [dataset["validation"][i]["text"] for i in range(len(dataset["validation"]))], 
    [dataset["validation"][i]["label_L"] for i in range(len(dataset["validation"]))],
    [dataset["validation"][i]["label_H"] for i in range(len(dataset["validation"]))],
)

# データローダの作成
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=256, shuffle=False)

# ====================
# BERTによるテキスト分類
# ====================

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

# ====================
# 訓練
# ====================

model = Bert4Classification(model_name, num_labels=6, lr=1e-5)

# 訓練中にモデルを保存するための設定
checkpoint = pl.callbacks.ModelCheckpoint(
    # 検証用データにおける損失が最も小さいモデルを保存する
    monitor="val_loss", mode="min", save_top_k=1,
    # モデルファイル（重みのみ）を "model" というディレクトリに保存する
    save_weights_only=True, dirpath="./model1"
)

early_stopping = EarlyStopping(
    monitor='val_loss',  # モニターするメトリクス
    patience=3,          # 何エポック改善が見られなければ早期停止するか
    verbose=True,        # ログに早期停止の詳細を表示する
    mode='min'           # val_lossを最小化することが目標（損失は小さい方が良いため）
)

# 訓練
trainer = pl.Trainer(max_epochs=100, callbacks=[checkpoint], accelerator="gpu", devices=1)
trainer.fit(model, dataloader_train, dataloader_val)

# ベストモデルの確認
print("ベストモデル: ", checkpoint.best_model_path)
print("ベストモデルの検証用データにおける損失: ", checkpoint.best_model_score)