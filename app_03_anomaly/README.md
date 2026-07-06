# App 03: 時序異常檢測 (Time Series Anomaly Detection)

展示 ASSTF 在持續資料流中對抗概念漂移的能力。

## 資料

合成工業感測器時序：
- 正常模式：多維正弦/趨勢訊號
- 在第 7 天後注入緩慢漂移（頻率改變、偏移）與異常尖峰
- 可替換為 SWaT、NAB、SMD 等真實資料集

## 執行

```bash
python app_03_anomaly/train.py
python app_03_anomaly/evaluate.py
```

## 預期結果

- ASSTF Autoencoder 參數量約 **2,070**，靜態基線約 **2,080**；ASSTF 透過 `latent_dim=3` 與 rank=1 結構投影保持略小於靜態基線。
- 靜態 Autoencoder 在概念漂移後重構誤差基線上升，F1 可能下降。
- ASSTF 透過線上 θs 適應維持較穩定的異常檢測 F1-score 與重構誤差。
- ASSTF 使用 10 倍於靜態基線的 early-stopping patience，以確保結構參數有機會突破 plateau。
