# App 05: 線上強化學習動態策略 (Online RL Dynamic Policy)

展示 ASSTF 策略網路在環境動態變化時的穩定性與線上適應能力。

## 環境

自訂簡易連續控制環境（類 CartPole），可在執行中動態改變重力/摩擦力：
- 狀態：位置、速度、角度、角速度
- 動作：連續力矩
- 目標：讓擺桿保持直立

可替換為 OpenAI Gym 的 HalfCheetah、Walker2d 或 MuJoCo。

## 執行

```bash
python app_05_rl/train.py
python app_05_rl/evaluate.py
```

## 預期結果

- 當重力突然增加 50% 時，ASSTF 策略能在數十步內恢復累積獎勵
- 靜態策略在擾動後獎勵顯著下降
