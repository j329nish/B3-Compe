# 卒論テーマ決めコンペ

## 1. 日本語の感情分析（WRIME）

- テキストの感情極性を5クラスに分類（-2, -1, 0, 1, 2）
- 使用したデータセット：WRIME [[link](https://github.com/ids-cv/wrime)]
- 評価指標：QWK

### ファイル構成

<pre>
sentiment/
├ juman-token.py　・・・ juman++の適用
├ sentiment.py　  ・・・ 訓練の実行と評価
├ augment.py　    ・・・ データ拡張を用いた訓練の実行と評価
├ pred.py　       ・・・ 予測の出力
├ pred_logit.py　 ・・・ 予測確率の出力
└ ensemble.ipynb　・・・ 予測確率のアンサンブル
</pre>

### 結果
| ID | モデル | データ拡張 | 学習率 | QWK<br>（Valid） | QWK<br>（Test） |
|-|-|-|-|-|-|
| 1 | ku-nlp/roberta-large-japanese-char-wwm | - | 1e-06 | 0.612 | - |
| 2 | nlp-waseda/roberta-large-japanese | - | 1e-05 | 0.622 | - |
| 3 | nlp-waseda/roberta-large-japanese | - | 1e-06 | 0.634 | - |
| 4 | nlp-waseda/roberta-large-japanese | 〇 | 1e-06 | 0.635 | - |
| 5 | nlp-waseda/roberta-large-japanese-seq512 | - | 1e-06 | 0.643 | - |
| 6 | nlp-waseda/roberta-large-japanese-seq512 | 〇 | 1e-06 | 0.642 | - |
| 7 | ku-nlp/deberta-v2-large-japanese-char-wwm | - | 1e-06 | 0.604 | - |
| 8 | ku-nlp/deberta-v2-large-japanese-char-wwm | 〇 | 1e-06 | 0.604 | - |
| 9 | studio-ousia/luke-japanese-large | - | 1e-05 | 0.621 | - |
| 10 | studio-ousia/luke-japanese-large | 〇 | 1e-05 | 0.617 | - |
| 11 | アンサンブル（1, 2, 4, 5） | - | - | 0.656 | 0.645 |

最適化手法はいずれもAdamWである。

## 2. 英語の難易度推定（CEFR-SP）

- テキストの難易度を6クラスに分類（1, 2, 3, 4, 5, 6）
- 使用したデータセット：CEFR-SP [[link](https://github.com/yukiar/CEFR-SP)]
- 評価指標：QWK

### ファイル構成

<pre>
estimation/
├ estimation.py　 ・・・ 訓練の実行
├ eval.py　       ・・・ 評価
├ pred.py　       ・・・ 予測の出力
├ pred_logit.py　 ・・・ 予測確率の出力
└ ensemble.ipynb　・・・ 予測確率のアンサンブル
</pre>

### 結果
| ID | モデル | H/L | 最適化手法 | QWK<br>（Valid） | QWK<br>（Test） |
|-|-|-|-|-|-|
| 1 | microsoft/deberta-v3-large | H | AdamW | 0.912 | - |
| 2 | microsoft/deberta-v3-large | L | AdamW | 0.898 | - |
| 3 | xlm-roberta-large | H | Adam | 0.920 | - |
| 4 | xlm-roberta-large | L | Adam | 0.884 | - |
| 5 | アンサンブル（1, 2, 3） | - | - | 0.932 | 0.922 |

学習率はいずれも1e-05である。<br><br>

(最終更新 2025/5/10)
