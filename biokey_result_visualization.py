import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import tkinter as tk
from tkinter import messagebox

def visualization_function(model, features_scaled, plot_title="Biokey Result Visualization"):
    try:
        h = .02
        x_min, x_max = features_scaled[:, 0].min() - 1, features_scaled[:, 0].max() + 1
        y_min, y_max = features_scaled[:, 1].min() - 1, features_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Obtain decision function values for each point in the meshgrid
        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.6)
        plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c='white', edgecolors='k', s=20)

        plt.title(plot_title)
        plt.xlabel('Dwell Time (standardized)')
        plt.ylabel('Flight Time (standardized)')
        plt.legend(['Decision Boundary', 'Keystroke Data'])
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"Visualization Error: {e}")

def run_ai_pipeline():
    try:
        # 1. Start Load
        if not os.path.exists('keystroke_data.json'):
            messagebox.showerror("Error", "No keystroke_data.json กรุณารันเเละทำ SelfTrainDataApplication please Sir!")
            return

        with open('keystroke_data.json', 'r', encoding='utf-8') as f:
            df = pd.DataFrame(json.load(f))

        if len(df) < 10: # เช็คว่ามีข้อมูลพอจะวาดกราฟไหม
             messagebox.showwarning("Warning", "ข้อมูลน้อยเกินไปที่จะแสดงผลกราฟได้สวยงาม ลองพิมพ์เพิ่มอีกหน่อยนะเธอ")
             return

        # 2. Start Process and Std.
        df['dwellTime'] = df['dwellTime'].astype(float)
        df['flightTime'] = df['flightTime'].astype(float)
        df = df[df['flightTime'] < 2000] # ตัด Outliers
        
        scaler = StandardScaler()
        # เราเก็บ features_scaled ไว้ในตัวแปรเพื่อส่งไปวาดกราฟ
        features_scaled = scaler.fit_transform(df[['dwellTime', 'flightTime']])
        
        with open('biokey_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        # 3. Train Start! (สมอง AI)
        # nu=0.1 คือยอมให้มี Error ได้ 10%
        model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
        model.fit(features_scaled)
        
        with open('biokey_model.pkl', 'wb') as f:
            pickle.dump(model, f)

        # 4. Simulate (จำลองการตรวจสอบ)
        avg_dt = df['dwellTime'].mean()
        avg_ft = df['flightTime'].mean()
        test_input = scaler.transform(pd.DataFrame([[avg_dt, avg_ft]], columns=['dwellTime', 'flightTime']))
        prediction = model.predict(test_input)
        status = "Welcome Sir! (Normal)" if prediction[0] == 1 else "Anomaly Detected!"

        # แจ้งเตือนความสำเร็จ
        messagebox.showinfo("BioKey Brain", 
            f"Training Success!\n\n"
            f"Data Points: {len(df)} แถว\n"
            f"Simulation Result: {status}\n\n"
            f"กด OK เพื่อดูกราฟการทำงานของ AI...")
        
        # --- เรียกฟังก์ชันวาดกราฟ ---
        visualization_function(model, features_scaled)
        # --------------------------

    except Exception as e:
        messagebox.showerror("Error", f"เกิดข้อผิดพลาด: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("BioKey: Brain Processor (Visualized)")
    root.geometry("350x250")
    tk.Label(root, text="BioKey AI Center + Visualization", font=("Arial", 12, "bold")).pack(pady=20)
    tk.Label(root, text="เมื่อฝึกเสร็จ ระบบจะเปิดหน้าต่างกราฟขึ้นมา", font=("Arial", 9)).pack()
    tk.Button(root, text="🚀 ฝึก AI และดูกราฟสมอง", command=run_ai_pipeline, 
              bg="#9C27B0", fg="white", font=("Arial", 10, "bold"), padx=20, pady=15).pack(pady=20)
    root.mainloop()
    