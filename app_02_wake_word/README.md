# App 02: 個人化語音喚醒詞檢測 (Personalized Wake Word Detection)

展示 ASSTF 與 1-D CNN 的結合（`ASSTFConv1d`），並在動態信噪比環境下進行動態秩適應。

## 資料

本示範使用合成 1D 音訊波形：
- 正樣本：內嵌特定頻率模式的喚醒詞波形
- 負樣本：隨機背景語音片段
- 動態注入白噪音，SNR 範圍 -5 dB ~ 10 dB

可輕易替換為 `torchaudio` 載入的 Google Speech Commands 或 Hey Snips 資料集。

## 執行

```bash
python app_02_wake_word/train.py
python app_02_wake_word/evaluate.py
```

## 預期結果

- ASSTF 模型參數量約 **779**，靜態基線約 **1,202**，ASSTF 更小。
- 在 -5 dB 低 SNR 下，ASSTF 約 **93.5%**，靜態基線約 **53.4%**。
- 在 0 dB 及以上，兩者均接近或達到 100% 準確率。
- ASSTF 使用 10 倍於靜態基線的 early-stopping patience，以確保結構參數有機會突破 plateau。
