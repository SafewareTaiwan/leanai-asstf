# 中文發布文案（知乎 / 机器之心 / 掘金）

**標題：** ASSTF 開源：一種會在推論時自我重組的神經網路層

**正文：**

今天，我們正式開源 ASSTF（Adaptive State-Space Transfer Function，自適應狀態空間轉移函數）。

ASSTF 是一個 PyTorch 層級的開源框架，核心理念是：**適應性勝過規模**。與其追求愈來愈大的模型，不如讓小模型在推論時持續自我調整，以應對新用戶、新環境與概念漂移。

ASSTF 把傳統的 `nn.Linear` / `nn.Conv1d` 拆成兩部分：
- **核心參數 θc**：儲存任務的基礎知識；
- **結構參數 θs**：低秩、可微的拓撲調控器，可在訓練與推論時更新。

透過 `SurpriseMinimizer`，模型在部署後仍只更新 θs，就能實現：
- 用戶個人化
- 域適應（domain adaptation）
- 概念漂移補償
- 端上隱私保護（資料不必上雲）

## 本次開源內容

- PyTorch 參考實作：`ASSTFLinear`、`ASSTFConv1d`、`BilevelTrainer`、`SurpriseMinimizer`
- 6 個可復現 demo：手勢辨識、喚醒詞、異常檢測、小樣本學習、強化學習、邊緣 NLP
- App 01–05 使用開源生成腳本的合成數據，便於隔離算法行為
- App 06 在真實 SST-2 數據集上驗證參數效率
- 完整英文文件與 Colab 一鍵執行筆記本

## 授權模式

- **Community License**：個人學習、學術研究、非營利教育免費
- **Commercial License**：商業產品、SaaS、硬體整合需付費授權

## 快速體驗

```bash
pip install leanai-asstf
```

或直接在 Colab 執行：
https://colab.research.google.com/github/SafewareTaiwan/leanai-asstf/blob/main/notebooks/ASSTF_Quickstart.ipynb

我們把這次定位為 **Stage 1：建立可復現的基礎與原創者信用**。Stage 2 將補上真實公開數據集、Hugging Face 整合與學術論文；Stage 3 則會推出企業級工具與邊緣部署 SDK。

歡迎大家試用、提 issue、貢獻真實數據集 benchmark，一起證明 AI 的未來不只是變大，而是變得更聰明、更適應。

GitHub：https://github.com/SafewareTaiwan/leanai-asstf
商業授權洽詢：Bentley@safeware.com.tw
