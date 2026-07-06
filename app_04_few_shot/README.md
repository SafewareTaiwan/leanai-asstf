# App 04: 小樣本圖像分類 (Few-Shot Meta-Learner)

展示 ASSTF 將結構參數 θs 作為任務特定元知識的能力。

## 資料

本示範使用 sklearn `load_digits`（8x8 灰階手寫數字）作為小樣本任務來源：
- 每次取樣一個 N-way K-shot 任務
- 訓練階段學習核心參數 θc
- 新任務僅微調結構參數 θs

可替換為 mini-ImageNet、Omniglot、ChestX-ray 等真實小樣本資料集。

## 執行

```bash
python app_04_few_shot/train.py
python app_04_few_shot/evaluate.py
```

## 預期結果

- 在 5-way 1-shot 設定下，ASSTF 用遠少於 MAML-approx 的參數達到相近或更好的準確率
