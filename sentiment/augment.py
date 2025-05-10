from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset
import torch
import numpy as np
from sklearn.metrics import cohen_kappa_score
import nlpaug.augmenter.word as naw

print(torch.cuda.is_available())  # GPUが利用可能ならTrueが返される

# モデルの指定
model_name = "nlp-waseda/roberta-large-japanese-seq512"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# データセットの読み込み
train_dataset = load_dataset('json', data_files={'train': './dataset/train-tokenized.json'})['train']
valid_dataset = load_dataset('json', data_files={'validation': './dataset/valid-tokenized.json'})['validation']

# 日本語のデータ拡張（SynonymAug は英語用なので、BackTranslationAug などの適切なものを選ぶ）
aug = naw.ContextualWordEmbsAug(model_path='tohoku-nlp/bert-base-japanese', action="substitute")

import random

def augment_text(examples):
    num_samples = len(examples["sentence"])
    num_to_augment = max(1, int(0.01 * num_samples))  # 1% を選択（最低1つ）
    indices = random.sample(range(num_samples), num_to_augment)  # ランダムに選択

    for i in indices:
        examples["sentence"][i] = aug.augment(examples["sentence"][i])[0]  # リストの最初の要素を取得
    
    return examples

# データ拡張
train_dataset = train_dataset.map(augment_text, batched=True)

# データの前処理（トークナイズ）
def preprocess_function(examples):
    return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=128)

train_dataset = train_dataset.map(preprocess_function, batched=True)
valid_dataset = valid_dataset.map(preprocess_function, batched=True)

# ラベルの変換
def transform_labels(examples):
    examples['label'] = [label + 2 for label in examples['writer_sentiment']]
    return examples

train_dataset = train_dataset.map(transform_labels, batched=True)
valid_dataset = valid_dataset.map(transform_labels, batched=True)

# 必要なカラムだけに絞る
train_dataset = train_dataset.remove_columns(['sentence', 'writer_sentiment'])
valid_dataset = valid_dataset.remove_columns(['sentence', 'writer_sentiment'])

# モデルのロード
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

# トレーニングの設定
training_args = TrainingArguments(
    output_dir='./model6',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-6,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=100,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)  # 最大確率のクラスを選択
    qwk_score = cohen_kappa_score(predictions, labels, weights='quadratic')
    return {"qwk": qwk_score}

# Trainerのインスタンス作成
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# モデルの訓練
trainer.train()

# 訓練後に最適なモデルを保存
trainer.save_model()
