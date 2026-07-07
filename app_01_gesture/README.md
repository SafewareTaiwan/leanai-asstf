# App 01: 端側動態手勢識別 (Embedded Gesture Recognition)

展示 ASSTF 如何在推理階段針對不同使用者進行線上個人化適應。

## 資料

由於公開 IMU/EMG 資料集（如 Ninapro、UCI HAR）較大，本示範使用**高擬真合成資料**：
- 6 通道 IMU 訊號（3 軸加速度 + 3 軸陀螺儀）
- 每筆樣本 128 個時間點
- 5 種手勢類別
- 6 位虛擬使用者，每位有不同的振幅縮放、時間扭曲與偏移

程式碼中保留真實資料載入介面，只需替換 `load_data()` 即可。

## 執行

```bash
cd leanai-asstf_OpenSource
python app_01_gesture/train.py
python app_01_gesture/evaluate.py
```

## 預期結果

- ASSTF 模型參數量約 1K~3K
- 靜態 MLP 基線參數量約 5K~15K
- 在新使用者資料上經過數十個 batch 的線上適應後，ASSTF 準確率可提升 10~25%
