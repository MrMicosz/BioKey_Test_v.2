import os
import json
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import tkinter as tk
from tkinter import messagebox

def verify_user():
    try:
        # 1. Load Data
        if not os.path.exists('keystroke_data.json'):
            messagebox.showerror("Error", "No keystroke_data.json กรุณารันเเละทำ SelfTrainDataApplication please Sir!")
            return

        with open('biokey_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('biokey_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        if not os.path.exists('keystroke_data.json'):
            messagebox.showerror("Error", "No keystroke_data.json กรุณารันเเละทำ SelfTrainDataApplication please Sir!")
            return
        
        with open('keystroke_data.json', 'r', encoding='utf-8') as f:
            df = pd.DataFrame(json.load(f))

        df['dwellTime'] = df['dwellTime'].astype(float)
        df['flightTime'] = df['flightTime'].astype(float)
        features_scaled = scaler.transform(df[['dwellTime', 'flightTime']])

        predictions = model.predict(features_scaled)

        accuracy = (list(predictions).count(1) / len(predictions)) * 100

        if accuracy > 70: # ถ้าเหมือนเจ้าของเกิน 70%
            result = f"ยินดีต้อนรับเจ้าของเครื่อง\n(ความแม่นยำ: {accuracy:.2f}%)"
            color = "#4CAF50"
        else:
            result = f"ตรวจพบผู้บุกรุก!\n(ความเหมือนแค่: {accuracy:.2f}%)"
            color = "#F44336"

        messagebox.showinfo("BioKey Result", result)

    except Exception as e:
        messagebox.showerror("Error", f"เกิดข้อผิดพลาด: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("BioKey: Verifier")
    root.geometry("300x200")
    tk.Label(root, text="ระบบตรวจสอบตัวตนล่าสุด", font=("Arial", 12, "bold")).pack(pady=20)
    tk.Button(root, text="🔍 ตรวจสอบตัวตน", command=verify_user, 
              bg="#FF9800", fg="white", font=("Arial", 10, "bold"), padx=20, pady=15).pack()
    root.mainloop()