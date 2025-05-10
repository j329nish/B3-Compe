from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json

# モデルとトークナイザーのロード
model_name = "nlp-waseda/roberta-large-japanese-seq512"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained("./model5/checkpoint-2814")

# JSONファイルからデータを読み込む
def load_jsonl(json_path):
    data = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

eval_json_path = './dataset/valid-tokenized.json'
data = load_jsonl(eval_json_path)
eval_texts = [item["sentence"] for item in data]

# 評価用データの前処理
def preprocess_eval_data(tokenizer, texts, max_length=128):
    encodings = tokenizer(texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
    return encodings

dataset_eval = preprocess_eval_data(tokenizer, eval_texts)

# モデルを評価モードに
model.eval()

# 予測
with torch.no_grad():
    outputs = model(**dataset_eval)
    predictions = torch.argmax(outputs.logits, dim=-1)

# ラベルマッピング
label_map = [-2, -1, 0, 1, 2]
pred_labels = [label_map[p.item()] for p in predictions]

# 予測ラベルをテキストファイルに保存
output_path = "sentiment-dev.txt"
with open(output_path, "w", encoding="utf-8") as f:
    for label in pred_labels:
        f.write(f"{label}\n")

print(f"Predictions saved to {output_path}")