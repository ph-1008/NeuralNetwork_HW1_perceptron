import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import os
import sys
from sklearn.model_selection import train_test_split

class PerceptronUI:
    def __init__(self, master): # 初始化
        self.master = master
        self.master.title("Perceptron")
        self.master.geometry("1000x600")

        # 設定文件路徑
        self.file_path = r"NN_HW1_DataSet\basic"

        # 創建頂部框架
        self.top_frame = tk.Frame(master)
        self.top_frame.pack(pady=10, fill=tk.X)

        # Epoch 输入框
        self.epoch_label = tk.Label(self.top_frame, text="Epoch:", font=("Arial", 12))
        self.epoch_label.pack(side=tk.LEFT, padx=5)
        self.epoch_entry = tk.Entry(self.top_frame, font=("Arial", 12), width=10)
        self.epoch_entry.pack(side=tk.LEFT, padx=5)
        self.epoch_entry.insert(0, "100")

        # Learning Rate 輸入框
        self.lr_label = tk.Label(self.top_frame, text="Learning Rate:", font=("Arial", 12))
        self.lr_label.pack(side=tk.LEFT, padx=5)
        self.lr_entry = tk.Entry(self.top_frame, font=("Arial", 12), width=10)
        self.lr_entry.pack(side=tk.LEFT, padx=5)
        self.lr_entry.insert(0, "0.01")

        # 選擇數據文件
        self.file_label = tk.Label(self.top_frame, text="選擇數據文件:", font=("Arial", 12))
        self.file_label.pack(side=tk.LEFT, padx=5)
        self.file_var = tk.StringVar()
        self.file_dropdown = ttk.Combobox(self.top_frame, textvariable=self.file_var, width=30, font=("Arial", 12))
        self.file_dropdown['values'] = self.get_file_list()
        self.file_dropdown.bind('<<ComboboxSelected>>', self.on_select)
        self.file_dropdown.pack(side=tk.LEFT, padx=5)

        # 創建第二個框架用於放置Training按鈕和結果標籤
        self.button_frame = tk.Frame(master)
        self.button_frame.pack(pady=10, fill=tk.X)

        # Training 按鈕
        self.train_button = tk.Button(self.button_frame, text="Training", font=("Arial", 14), command=self.training)
        self.train_button.pack(side=tk.LEFT, padx=(10, 0))

        # 創建結果框架用於放置結果標籤
        self.result_frame = tk.Frame(self.button_frame)
        self.result_frame.pack(side=tk.LEFT, padx=10)

        # 當前 Epoch 標籤
        self.current_epoch_label = tk.Label(self.result_frame, text="Epoch: N/A", font=("Arial", 10))
        self.current_epoch_label.pack(anchor='w')

        # 訓練辨識率標籤
        self.train_accuracy_label = tk.Label(self.result_frame, text="Train Accuracy: N/A", font=("Arial", 10))
        self.train_accuracy_label.pack(anchor='w')

        # 測試辨識率標籤 
        self.test_accuracy_label = tk.Label(self.result_frame, text="Test Accuracy: N/A", font=("Arial", 10))
        self.test_accuracy_label.pack(anchor='w')

        # 權重標籤 
        self.weights_label = tk.Label(self.result_frame, text="Weights: N/A", font=("Arial", 10))
        self.weights_label.pack(anchor='w')

        # 創建圖形區域 
        self.fig, (self.ax_train, self.ax_test) = plt.subplots(1, 2, figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # 設置關閉窗口時的操作 
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def get_file_list(self): 
        return [f for f in os.listdir(self.file_path) if f.endswith('.txt')]

    def on_select(self, event):
        selected_file = self.file_var.get()
        if selected_file:
            full_path = os.path.join(self.file_path, selected_file)
            self.plot_data(full_path)

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
        # for label in np.unique(Label_train):
        #     colors = ['purple', 'yellow']
        #     self.ax_train.scatter(X_train[Label_train == label, 0], X_train[Label_train == label, 1], color=colors[int(label) % len(colors)])
        
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

    @staticmethod
    def sgn(x):
        return 1 if x >= 0 else -1
    
    def on_closing(self): 
        self.master.quit() 
        self.master.destroy() 
        sys.exit()

# 創建主窗口
root = tk.Tk()
app = PerceptronUI(root)
root.mainloop()
input("please input any key to exit!")