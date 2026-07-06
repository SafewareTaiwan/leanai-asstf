# App 06: 輕量端側文本意圖分類 (Edge NLP)

展示 ASSTF 在 NLP 領域的潛力：以 ASSTF 層替換 Tiny BERT 的前饋網路（FFN），實現參數減半且精度可比。

## 資料

使用 Hugging Face `datasets` 載入 SST-2 的情感分析子集。若網路受限，會自動回退到合成文本分類資料。

## 執行

```bash
python app_06_edge_nlp/train.py
python app_06_edge_nlp/evaluate.py
```

## 預期結果

- Tiny BERT + ASSTF-FFN 的總參數量約為原始 Tiny BERT 的 40~60%
- 在 SST-2 子集上準確率接近或略高於原始 Tiny BERT
