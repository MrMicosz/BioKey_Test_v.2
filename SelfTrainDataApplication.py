import tkinter as tk
from tkinter import messagebox
import json
import time

class BioKeyCollector:
    def __init__(self, root):
        self.root = root
        self.root.title("BioKey: Data Collector")
        self.root.geometry("500x450")

        self.logs = []
        self.key_press_times = {}
        self.last_release_time = 0

        # UI
        tk.Label(root, text="Self Key Time Training Data", font=("Arial", 14, "bold")).pack(pady=10)
        tk.Label(root, text="Please type the text you are comfortable with to help AI remember your typing rhythm:", font=("Arial", 10)).pack()
        
        self.text_area = tk.Text(root, height=10, width=50, font=("Courier", 12))
        self.text_area.pack(pady=10, padx=20)
        self.text_area.bind("<KeyPress>", self.on_press)
        self.text_area.bind("<KeyRelease>", self.on_release)

        self.btn_save = tk.Button(root, text="Success save as JSON", command=self.save_data, 
                                 bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), padx=20, pady=10)
        self.btn_save.pack(pady=10)

        self.lbl_count = tk.Label(root, text="Save!: 0 แถว", fg="blue")
        self.lbl_count.pack()

    def on_press(self, event):
        press_time = time.perf_counter() * 1000
        if event.keysym not in self.key_press_times:
            self.key_press_times[event.keysym] = press_time

    def on_release(self, event):
        release_time = time.perf_counter() * 1000
        press_time = self.key_press_times.get(event.keysym)

        if press_time:
            dwell_time = release_time - press_time
            flight_time = (press_time - self.last_release_time) if self.last_release_time > 0 else 0
            
            self.logs.append({
                "key": event.keysym,
                "dwellTime": round(dwell_time, 2),
                "flightTime": round(flight_time, 2),
                "timestamp": round(release_time, 2)
            })
            self.last_release_time = release_time
            del self.key_press_times[event.keysym]
            self.lbl_count.config(text=f"Save!: {len(self.logs)} แถว")

    def save_data(self):
        if len(self.logs) < 50:
            messagebox.showwarning("Warning", "Too small data at least 50 เเถว for better AI accuracy.")
        
        with open('keystroke_data.json', 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, indent=4)
        messagebox.showinfo("Success", "Save file as keystroke_data.json")

if __name__ == "__main__":
    root = tk.Tk()
    app = BioKeyCollector(root)
    root.mainloop()