# 類神經網路作業一 - 設計感知機類神經網路
## A. 程式執行說明
- GUI介面如下(左圖)，有Epoch, Learning Rate 輸入框、選擇數據文件的下拉選單、Training按鈕。
- 當選取一個文件後，會如下(右圖)，印出所有數據點。
- 並且在Training按鈕右邊會顯示 Epoch, Train Accuracy, Test Accuracy, Weight
![image](https://hackmd.io/_uploads/S1M_t-2kkx.png =48%x) ![image](https://hackmd.io/_uploads/B1DjYWnkkg.png =48%x)


## B. 程式碼簡介
#### 1. 以下是程式碼大至架構
```python=
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import os
import sys
from sklearn.model_selection import train_test_split

class PerceptronUI:
    def __init__(self, master:)...       ## 初始化GUI介面
    
    def get_file_list(self):...          ## 獲取文件列表
    
    def on_select(self, event):...       ## 選擇數據文件
    
    def plot_data(self, file_path):...   ## 繪製全部數據點
    
    def training(self):...               ## 主要訓練過程+繪製決策線
    
    @staticmethod
    def sgn(x):...                       ## 定義活化函數
    
    def on_closing(self):...             ## 當窗口關閉時 終止程式
    
# 創建主窗口
root = tk.Tk()
app = PerceptronUI(root)
root.mainloop()
```
#### 2. plot_data函數
- 獲取data的前兩column作為X(座標點)，第2 column作為y(label)。
- 再在 Training Data 和 Test Data 畫框中，繪製全部數據的分佈圖。
- 最後保存當前的軸限制 
```python=
def plot_data(self, file_path): 
    data = np.loadtxt(file_path)
    X = data[:, :2]
    y = data[:, 2]

    self.ax_train.clear()
    self.ax_test.clear()

    colors = ['purple', 'yellow']
    for label in np.unique(y):
        self.ax_train.scatter(X[y == label, 0], X[y == label, 1], color=colors[int(label) % len(colors)])
        self.ax_test.scatter(X[y == label, 0], X[y == label, 1], color=colors[int(label) % len(colors)])

    self.ax_train.set_title('Training Data')
    self.ax_test.set_title('Test Data')
    self.ax_train.legend()
    self.ax_test.legend()

    # 保存當前的軸限制
    self.x_lim = self.ax_train.get_xlim()
    self.y_lim = self.ax_train.get_ylim()
    self.x_lim_test = self.ax_test.get_xlim()
    self.y_lim_test = self.ax_test.get_ylim()

    self.canvas.draw()
```
#### 3. training 函數
- 獲取UI介面輸入框中的epoch, learning rate，和選中的數據文件。
- 第17~22行中，我將數據集的標籤統一轉換為 -1 和 1，因為文件內的label有0,1 或是 1,2的
- 第24行，使用```train_test_split```函數將數據集隨機分為 2/3 當作訓練資料，1/3 當做測試資料
- 32~34行，training data畫框設置軸限制為之前保存的值
- 36~81行，主要訓練區域，其中包含
    - 訓練weight
    - 計算訓練辨識率
    - 繪製每次訓練出weight的決策線
    - 最後設置 當train_accuracy = 100時，break
- 86~99行，利用訓練出來的結果(weight)，在test data畫框中繪製最終決策線，與計算並印出測試辨識率
```python=
def training(self): 
        self.test_accuracy_label.config(text="Test Accuracy: N/A")

        epoch = int(self.epoch_entry.get())
        learning_rate = float(self.lr_entry.get())
        selected_file = self.file_var.get()

        if not selected_file:
            print("請先選擇數據文件")
            return

        full_path = os.path.join(self.file_path, selected_file)
        data = np.loadtxt(full_path)
        X = data[:, :2]
        Label = data[:, 2]

        # 將數據集的標籤轉換為 -1 和 1
        unique_labels = np.unique(Label)
        if set(unique_labels) == {0, 1}:
            Label = np.where(Label == 0, -1, 1) # 將 0 轉換為 -1，1 保持不變
        elif set(unique_labels) == {1, 2}:
            Label = np.where(Label == 1, 1, -1) # 將 1 轉換為 1，2 轉換為 -1

        X_train, X_test, Label_train, Label_test = train_test_split(X, Label, test_size=1/3)

        print(f"訓練集大小: {X_train.shape[0]}")
        print(f"測試集大小: {X_test.shape[0]}")
        print(f"Training with Epoch: {epoch}, Learning Rate: {learning_rate}, File: {selected_file}")

        

        # 設置軸限制為之前保存的值
        self.ax_train.set_xlim(self.x_lim)
        self.ax_train.set_ylim(self.y_lim)

        w = np.random.rand(3)
        self.lines = []
        
        for n in range(epoch):
            self.ax_train.clear()
            for label in np.unique(Label_train):
                colors = ['purple', 'yellow']
                # 修正顏色索引：將 -1 對應索引 0，將 1 對應索引 1
                color_index = 0 if label == -1 else 1
                self.ax_train.scatter(X_train[Label_train == label, 0], X_train[Label_train == label, 1], color=colors[color_index])
                

            for i in range(len(X_train)):
                x = np.array([-1] + X_train[i].tolist())
                label = Label_train[i]
                v = np.dot(w, x)
                y = self.sgn(v)
                if label == 1 and y < 0:
                    w += learning_rate * x
                elif label == -1 and y >= 0:
                    w -= learning_rate * x

            # 計算訓練辨識率
            train_predictions = [self.sgn(np.dot(w, np.array([-1] + x.tolist()))) for x in X_train]
            train_accuracy = np.mean(train_predictions == Label_train) * 100

            # 更新標籤 
            self.current_epoch_label.config(text=f"Epoch: {n+1}")
            self.train_accuracy_label.config(text=f"Train Accuracy: {train_accuracy:.2f}%")
            self.weights_label.config(text=f"Weights: {w}")

            x_vals = np.linspace(min(X_train[:, 0]), max(X_train[:, 0]), 100)
            y_vals = (w[0] - w[1] * x_vals) / w[2]
            line = self.ax_train.plot(x_vals, y_vals, 'r-', alpha=0.5) 
            self.lines.append(line)

            self.ax_train.set_title('Training Data')
            self.ax_train.set_xlim(self.ax_test.get_xlim())
            self.ax_train.set_ylim(self.ax_test.get_ylim())

            self.canvas.draw()
            self.master.update_idletasks()
            self.master.update()

            if train_accuracy == 100:
                break

        # 繪製最終的決策邊界在測試數據上
        self.ax_test.clear()
        
        for label in np.unique(Label_test):
            colors = ['purple', 'yellow']
            # 修正顏色索引：將 -1 對應索引 0，將 1 對應索引 1
            color_index = 0 if label == -1 else 1
            self.ax_test.scatter(X_test[Label_test == label, 0], X_test[Label_test == label, 1], color=colors[color_index])

        x_vals = np.linspace(min(X_test[:, 0]), max(X_test[:, 0]), 100)
        y_vals = (w[0] - w[1] * x_vals) / w[2]
        self.ax_test.plot(x_vals, y_vals, 'r-')

        # 計算測試辨識率
        test_predictions = [self.sgn(np.dot(w, np.array([-1] + x.tolist()))) for x in X_test]
        test_accuracy = np.mean(test_predictions == Label_test) * 100
        self.test_accuracy_label.config(text=f"Test Accuracy: {test_accuracy:.2f}%")

        # 設置軸限制為之前保存的值
        self.ax_test.set_xlim(self.x_lim_test)
        self.ax_test.set_ylim(self.y_lim_test)

        self.ax_test.set_title('Test Data')
        self.ax_test.legend()

        self.canvas.draw()

        print("訓練完成")
        print(f"最終權重: {w}")

```
#### 4. sgn 活化函數
- $$
\phi(v) = 
\begin{cases} 
+1 & \text{if } v \geq 0 \\
-1 & \text{if } v < 0 
\end{cases}
$$

```python=
@staticmethod
    def sgn(x):
        return 1 if x >= 0 else -1
```


## C. 實驗結果分析及討論
#### 1. 2Ccircle1.txt
![image](https://hackmd.io/_uploads/By3wnG31Jg.png =70%x)
![image](https://hackmd.io/_uploads/rkWSqf2kkg.png =70%x)
由於資料的非線性分佈特性，感知器無法找到一個完全正確的分離邊界，仍有部分數據點被錯誤分類。
決策線結果在兩圓之外測試準確率較高，我認為是因為紫色點比黃色點數目多，如果犧牲黃色部分，將兩類分在同一邊，就能有較高的準確率。

#### 2. 2Circle1.txt
![image](https://hackmd.io/_uploads/HkMTJQ2y1l.png =70%x)
雖然此資料集線性不可分割，但是決策線大至在兩圓焦點上就可以達到90%的測試準確率。

#### 3. 2CloseS.txt
![image](https://hackmd.io/_uploads/SkfpXA3yJg.png =70%x)
線性可分割資料集，且當epoch=16時，訓練準確率就已達到100%。
由於訓練與測試資料集是隨機分割，所以儘管訓練準確率到達100%，如果測試數據被分得不好，測試準確率有可能無法到100%，會大概落在97~98%左右。


#### 4. 2CloseS2.txt
![image](https://hackmd.io/_uploads/BJdQN0hk1e.png =70%x)
線性可分割資料集，且當epoch=80時，準確率就已達到100%。

#### 5. 2CloseS3.txt
![image](https://hackmd.io/_uploads/B1lHEAhykx.png =70%x)
此資料集並非完美的線性可分割，紫色和黃色的點有重疊的部分，所以訓練準確率達不到100%。

#### 6. 2cring.txt
![image](https://hackmd.io/_uploads/Sye84RnkJl.png =70%x)
此資料集也是線性可分割，並且當epoch=1時，就可以分類成功。

#### 7. 2CS.txt
![image](https://hackmd.io/_uploads/rkVDEA2kkx.png =70%x)
資料集線性可分割。

#### 8. 2Hcircle1.txt
![image](https://hackmd.io/_uploads/H1XuV0hyyx.png =70%x)
資料集線性可分割。

#### 9. 2ring.txt
![image](https://hackmd.io/_uploads/SkXc4AnyJg.png =70%x)
資料集線性可分割。

#### 10. perceptron1.txt
![image](https://hackmd.io/_uploads/Sye2NmaJJe.png =70%x)
**↑** 由於資料集數據太少，所以training data有可能分到同一類的點，對於任何不交集四個點形成的正方型外的決策線都會分類成功，但同時，如果test data都取到另一類的點，此決策線這類不可能分類成功。
![image](https://hackmd.io/_uploads/rk1VrAnykx.png =70%x)
**↑** 由於資料集數據太少，儘管training data被分類成功，如果test data取到此決策線同一邊的點，測試準確率就只會剩下50%。
![image](https://hackmd.io/_uploads/H1cK57pkJl.png =70%x)
**↑** 此情況下，訓練、測試準確率都是100%。

#### 11. perceptron2.txt
![image](https://hackmd.io/_uploads/rytpkNT1yg.png =70%x)
![image](https://hackmd.io/_uploads/HJtAkVaJJl.png =70%x)
此數據集為**xor**問題，無法使用單層感知機解決，所以會有(訓練準確率, 測試準確率)=(100%, 0%)或是 (100%, 50%)的情況發生。

