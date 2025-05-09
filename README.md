# 卒論テーマ決めコンペ

## 1. 日本語の感情分析（WRIME）

- テキストの感情極性を5クラスに分類（-2, -1, 0, 1, 2）
- 使用したデータセット：WRIME [[link](https://github.com/ids-cv/wrime)]
- 評価指標：QWK

### 結果
| ID | モデル | データ拡張 | 学習率 | Validに対するQWK | Testに対するQWK |
|-|-|-|-|-|-|
| 1 | ku-nlp/deberta-v2-large-japanese-char-wwm | 〇 | 1e-06 | 0.604 | - |
| 5 | アンサンブル（1, 2, 3） | - | - | 0.932 | 0.922 |

## 2. 英語の難易度推定（CEFR-SP）

使用したデータセット：CEFR-SP [[link](https://github.com/yukiar/CEFR-SP/tree/main/CEFR-SP/Wiki-Auto)]

### 結果
| ID | モデル | H/L | 最適化手法 | 学習率 | Validに対するQWK | Testに対するQWK |
|-|-|-|-|-|-|-|
| 1 | microsoft/deberta-v3-large | H | AdamW | 1e-05 | 0.912 | - |
| 2 | microsoft/deberta-v3-large | L | AdamW | 1e-05 | 0.898 | - |
| 3 | xlm-roberta-large | H | Adam | 1e-05 | 0.920 | - |
| 4 | xlm-roberta-large | L | Adam | 1e-05 | 0.884 | - |
| 5 | アンサンブル（1, 2, 3） | - | - | - | 0.932 | 0.922 |

(最終更新 2025/5/10)
