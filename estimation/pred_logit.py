import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import cohen_kappa_score

model_name = "xlm-roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
max_length = 128

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
        
model_path = "./model1/epoch=1-step=876.ckpt"
model = Bert4Classification.load_from_checkpoint(model_path)
model.eval()  # 評価モードにする

# 評価用データの読み込み
eval_text_path = 'dataset/dev/dev.txt'

# テキストファイルからデータを読み込む
with open(eval_text_path, 'r', encoding='utf-8') as f:
    eval_texts = f.readlines()

# 改行を削除
eval_texts = [text.strip() for text in eval_texts]

print(f"Loaded {len(eval_texts)} evaluation texts")

# 評価用データの前処理
def preprocess_eval_data(tokenizer, max_length, texts):
    dataset_for_loader = list()
    for text in texts:
        encoding = tokenizer(text, max_length=max_length, padding="max_length", truncation=True)
        encoding = {key: torch.tensor(value) for key, value in encoding.items()}
        dataset_for_loader.append(encoding)
    return dataset_for_loader

dataset_eval = preprocess_eval_data(tokenizer, max_length, eval_texts)

# DataLoaderの作成
dataloader_eval = DataLoader(dataset_eval, batch_size=256, shuffle=False)

# 予測
predictions = []
probs = []  # 各ラベルの確率を保存

model.eval()  # モデルを評価モードにする

with torch.no_grad():
    for batch in dataloader_eval:
        output = model.bert_sc(**batch)
        
        # ロジットから確率を計算
        prob = F.softmax(output.logits, dim=-1)
        prob_values, labels_predicted = prob.max(dim=-1)
        
        predictions.extend(labels_predicted.tolist())
        probs.extend(prob.tolist())

# ラベルマッピング
label_map = [1, 2, 3, 4, 5, 6]
pred_labels = [label_map[pred] for pred in predictions]

print(f"Predicted {len(pred_labels)} labels")

output_path = "estimation-with-probs.txt"  # 保存先のファイルパス

# 予測ラベルと確率をテキストファイルに保存
with open(output_path, "w", encoding="utf-8") as f:
    for label, prob in zip(pred_labels, probs):
        prob_str = ', '.join([f"{i}: {p:.4f}" for i, p in enumerate(prob)])  # 各ラベルの確率
        f.write(f"{label}, {prob_str}\n")

print(f"Predictions with probabilities saved to {output_path}")