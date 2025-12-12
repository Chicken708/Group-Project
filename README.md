# 專案總覽
本專案使用 Python 與 Gymnasium 建立一個 CartPole 強化學習代理人（Agent）。
架構中定義了抽象的 Agent 類別，讓不同的策略（例如隨機策略、Q-learning）都能以統一流程運作。

## 專案內容
- Gymnasium v1.2.2
- Part1
- Part2
- Part3
  
## 安裝依賴

```bash
# 1. 建立虛擬環境
python -m venv .venv

# 2. 啟用虛擬環境
source .venv/bin/activate

# 3. 到 Gymnasium  的目錄
cd group_project/Gymnasium

# 4. 安裝 Gymnasium
pip install -e .

# 5. 安裝其他依賴
pip install "gymnasium[classic_control]"
pip install matplotlib
pip install numpy
```

---

## 執行

### **Part 1: Mountain Car**

```bash
# 訓練 Agent
python mountain_car.py --train --episodes 5000

# 渲染與視覺化
python mountain_car.py --render --episodes 10
```

### **Part 2: Frozen Lake**
執行 Frozen Lake 環境：

```bash
python frozen_lake.py
```

### **Part 3: CartPole**
執行 CartPole 專案：

```bash
# 執行預設版本（Q-Learning）
python part3.py

# 指定執行 Q-Learning 版本
python part3.py --agent qlearning

# 指定執行 Random 版本（不做學習）
python part3.py --agent random
```
---
### **成員貢獻**
| 成員 | 貢獻 |
|------|------|
| 鄧皓元  | 完成 part3 |
| 黃瑋晟  | 完成 part2 及 UML 圖 |
| 王皓渝  | 完成 Demo Slide |