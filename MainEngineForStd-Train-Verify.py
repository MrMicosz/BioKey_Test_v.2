import os
import json
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import tkinter as tk
from tkinter import messagebox

def run_ai_pipeline():
    try:
        # Start Load
        if not os.path.exists('keystroke_data.json'):
            messagebox.showerror("Error", "No keystroke_data.json กรุณารันเเละทำ SelfTrainDataApplication please Sir!")
            return
        
        with open('keystroke_data.json', 'r', encoding='utf-8') as f:
            df = pd.DataFrame(json.load(f))

        # Start Process and Std.
        df['dwellTime'] = df['dwellTime'].astype(float)
        df['flightTime'] = df['flightTime'].astype(float)
        df = df[df['flightTime'] < 2000] # ตัด Outliers
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(df[['dwellTime', 'flightTime']])
        
        with open('biokey_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        # 3. Train Start!
        model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
        model.fit(features_scaled)
        
        with open('biokey_model.pkl', 'wb') as f:
            pickle.dump(model, f)

        # 4. Simulate
        # Try to see with average values
        avg_dt = df['dwellTime'].mean()
        avg_ft = df['flightTime'].mean()
        test_input = scaler.transform(pd.DataFrame([[avg_dt, avg_ft]], columns=['dwellTime', 'flightTime']))
        prediction = model.predict(test_input)
        
        status = "Welcome Sir!" if prediction[0] == 1 else "Something Worng!"

        messagebox.showinfo("BioKey Brain", 
            f"Process Success\n\n"
            f"Start Learning From : {len(df)} แถว\n"
            f"Status of AI Now\n"
            f"Train Result: {status}")

    except Exception as e:
        messagebox.showerror("Error", f"เกิดข้อผิดพลาด: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("BioKey: Brain Processor")
    root.geometry("350x200")
    tk.Label(root, text="BioKey AI Processing Center", font=("Arial", 12, "bold")).pack(pady=20)
    tk.Button(root, text="🚀 ฝึก AI และตรวจสอบระบบ", command=run_ai_pipeline, 
              bg="#2196F3", fg="white", font=("Arial", 10, "bold"), padx=20, pady=15).pack()
    root.mainloop()