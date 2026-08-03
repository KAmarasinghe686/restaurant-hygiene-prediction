import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib

# ══════════════════════════════════════════
# LOAD SAVED MODEL + REFERENCE DATA
# ══════════════════════════════════════════
try:
    model        = joblib.load('best_model.pkl')
    feature_cols = joblib.load('feature_cols.pkl')
    print(f"Model loaded: {type(model).__name__}")
except Exception as e:
    print(f"ERROR loading model: {e}")
    exit()

df_ref = pd.read_csv('restaurant_hygiene_clean.csv')

zipcode_complaints = df_ref.groupby('zipcode')[
    'food_complaints_zipcode'].first().to_dict()
median_complaints  = df_ref['food_complaints_zipcode'].median()

violation_map   = df_ref.set_index(
    'violation_code')['violation_code_encoded'].to_dict()
violation_codes = sorted(df_ref['violation_code'].unique().tolist())

grade_map = {0:'A', 1:'B', 2:'C'}
labels    = {
    'A': 'Excellent — Passed Inspection',
    'B': 'Needs Improvement — Some Violations',
    'C': 'Failed — Serious Violations',
}
colors = {
    'A': '#0f9d58',
    'B': '#f4b400',
    'C': '#db4437',
}

# ══════════════════════════════════════════
# PREDICT FUNCTION
# ══════════════════════════════════════════
def predict_grade(inputs):
    row   = pd.DataFrame([inputs], columns=feature_cols)
    row   = row.fillna(0)
    pred  = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    grade = grade_map[pred]
    conf  = round(max(proba)*100, 1)
    prob_a = round(proba[0]*100, 1)
    prob_b = round(proba[1]*100, 1)
    prob_c = round(proba[2]*100, 1)
    return grade, conf, prob_a, prob_b, prob_c

# ══════════════════════════════════════════
# UI
# ══════════════════════════════════════════
class PredictUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Restaurant Hygiene Grade Predictor")
        self.root.geometry("720x780")
        self.root.configure(bg="#F0F2F5")
        self.build_header()
        self.build_model_info()
        self.build_inputs()
        self.build_button()
        self.build_result()

    # ── HEADER ──
    def build_header(self):
        hdr = tk.Frame(self.root, bg="#1a73e8", pady=14)
        hdr.pack(fill='x')
        tk.Label(hdr,
                 text="🍽  Restaurant Hygiene Grade Predictor",
                 font=("Arial", 16, "bold"),
                 bg="#1a73e8", fg="white").pack()
        tk.Label(hdr,
                 text="NYC Queens — Gradient Boosting Model",
                 font=("Arial", 9),
                 bg="#1a73e8", fg="#c8e0fc").pack()

    # ── MODEL INFO ──
    def build_model_info(self):
        frame = tk.Frame(self.root, bg="#e8f5e9",
                         relief='solid', bd=1)
        frame.pack(fill='x', padx=16, pady=(12,0))
        inner = tk.Frame(frame, bg="#e8f5e9",
                         padx=12, pady=8)
        inner.pack(fill='x')
        info = [
            ("Model"    , type(model).__name__),
            ("Accuracy" , "82.28%"),
            ("F1 Score" , "0.8342"),
            ("CV Score" , "82.92% ± 0.48%"),
        ]
        for i, (label, val) in enumerate(info):
            tk.Label(inner, text=f"{label}:",
                     font=("Arial", 9, "bold"),
                     bg="#e8f5e9",
                     fg="#1b5e20").grid(
                     row=0, column=i*2,
                     padx=(0,2), sticky='e')
            tk.Label(inner, text=val,
                     font=("Arial", 9),
                     bg="#e8f5e9",
                     fg="#2e7d32").grid(
                     row=0, column=i*2+1,
                     padx=(0,16), sticky='w')

    # ── INPUTS ──
    def build_inputs(self):
        frame = tk.LabelFrame(self.root,
                              text="  Inspection Details  ",
                              font=("Arial", 10, "bold"),
                              bg="#F0F2F5",
                              padx=16, pady=12)
        frame.pack(fill='x', padx=16, pady=12)

        # ── Inspection Type ──
        tk.Label(frame, text="Inspection Type",
                 font=("Arial", 9, "bold"),
                 bg="#F0F2F5").grid(row=0, column=0,
                 sticky='w', pady=5)
        self.insp_type = ttk.Combobox(frame,
            values=["Initial",
                    "Reinspection",
                    "Reopening"],
            state='readonly', width=28)
        self.insp_type.set("Initial")
        self.insp_type.grid(row=0, column=1,
                            sticky='w', padx=8, pady=5)
        self.insp_type.bind("<<ComboboxSelected>>",
                            self.on_type_change)

        # ── Violation Code ──
        tk.Label(frame, text="Violation Code",
                 font=("Arial", 9, "bold"),
                 bg="#F0F2F5").grid(row=1, column=0,
                 sticky='w', pady=5)
        self.violation_code = ttk.Combobox(frame,
            values=violation_codes,
            state='readonly', width=28)
        self.violation_code.set(violation_codes[0])
        self.violation_code.grid(row=1, column=1,
                                  sticky='w',
                                  padx=8, pady=5)

        # Auto severity label
        self.severe_label = tk.Label(frame,
            text="",
            font=("Arial", 8, "bold"),
            bg="#F0F2F5")
        self.severe_label.grid(row=1, column=2,
                                sticky='w', padx=4)

        self.violation_code.bind(
            "<<ComboboxSelected>>",
            self.on_violation_change)

        # ── Total Violations ──
        tk.Label(frame, text="Total Violations Found",
                 font=("Arial", 9, "bold"),
                 bg="#F0F2F5").grid(row=2, column=0,
                 sticky='w', pady=5)
        self.violation_count = tk.StringVar(value="2")
        tk.Entry(frame,
                 textvariable=self.violation_count,
                 width=10,
                 font=("Arial", 10)).grid(
                 row=2, column=1,
                 sticky='w', padx=8, pady=5)
        tk.Label(frame,
                 text="A avg=4.93  B avg=7.59  C avg=10.11",
                 font=("Arial", 8),
                 bg="#F0F2F5", fg="#666").grid(
                 row=2, column=2, sticky='w')

        # ── Critical Flag ──
        tk.Label(frame, text="Critical Violation?",
                 font=("Arial", 9, "bold"),
                 bg="#F0F2F5").grid(row=3, column=0,
                 sticky='w', pady=5)
        self.critical = ttk.Combobox(frame,
            values=["Yes", "No"],
            state='readonly', width=10)
        self.critical.set("No")
        self.critical.grid(row=3, column=1,
                           sticky='w', padx=8, pady=5)
        tk.Label(frame,
                 text="Inspector marked violation as critical",
                 font=("Arial", 8),
                 bg="#F0F2F5", fg="#666").grid(
                 row=3, column=2, sticky='w')

        # ── Inspection Month ──
        months = ["January","February","March",
                  "April","May","June","July",
                  "August","September","October",
                  "November","December"]
        tk.Label(frame, text="Inspection Month",
                 font=("Arial", 9, "bold"),
                 bg="#F0F2F5").grid(row=4, column=0,
                 sticky='w', pady=5)
        self.month = ttk.Combobox(frame,
            values=months,
            state='readonly', width=15)
        self.month.set("June")
        self.month.grid(row=4, column=1,
                        sticky='w', padx=8, pady=5)
        tk.Label(frame,
                 text="June=best month  August=worst",
                 font=("Arial", 8),
                 bg="#F0F2F5", fg="#666").grid(
                 row=4, column=2, sticky='w')

        # ── Inspection Year ──
        tk.Label(frame, text="Inspection Year",
                 font=("Arial", 9, "bold"),
                 bg="#F0F2F5").grid(row=5, column=0,
                 sticky='w', pady=5)
        self.year = ttk.Combobox(frame,
            values=["2023","2024","2025"],
            state='readonly', width=10)
        self.year.set("2025")
        self.year.grid(row=5, column=1,
                       sticky='w', padx=8, pady=5)

        # ── Zipcode ──
        tk.Label(frame, text="Zipcode (Queens)",
                 font=("Arial", 9, "bold"),
                 bg="#F0F2F5").grid(row=6, column=0,
                 sticky='w', pady=5)
        zipcode_list = sorted([str(z) for z in
                               zipcode_complaints.keys()])
        self.zipcode = ttk.Combobox(frame,
            values=zipcode_list,
            state='readonly', width=12)
        self.zipcode.set(zipcode_list[0])
        self.zipcode.grid(row=6, column=1,
                          sticky='w', padx=8, pady=5)
        self.zipcode.bind("<<ComboboxSelected>>",
                          self.on_zipcode_change)

        self.complaints_label = tk.Label(frame,
            text="Food complaints: —",
            font=("Arial", 8),
            bg="#F0F2F5", fg="#666")
        self.complaints_label.grid(row=6, column=2,
                                    sticky='w')

        # Initialize labels
        self.on_violation_change(None)
        self.on_zipcode_change(None)

    # ── EVENT HANDLERS ──
    def on_type_change(self, event):
        t = self.insp_type.get()
        if t == "Initial":
            self.violation_count.set("2")
        elif t == "Reinspection":
            self.violation_count.set("7")
        elif t == "Reopening":
            self.violation_count.set("10")

    def on_violation_change(self, event):
        vcode = self.violation_code.get()
        if vcode.startswith('0'):
            self.severe_label.config(
                text="⚠️ Severe violation",
                fg="#db4437")
        else:
            self.severe_label.config(
                text="✅ Non-severe violation",
                fg="#0f9d58")

    def on_zipcode_change(self, event):
        try:
            zc = int(self.zipcode.get())
            complaints = zipcode_complaints.get(
                zc, median_complaints)
            self.complaints_label.config(
                text=f"Food complaints: {complaints:.0f}")
        except:
            pass

    # ── PREDICT BUTTON ──
    def build_button(self):
        tk.Button(self.root,
                  text="  Predict Grade  ",
                  command=self.predict,
                  bg="#1a73e8", fg="white",
                  relief='flat', padx=24, pady=10,
                  font=("Arial", 12, "bold")).pack(pady=8)

    # ── RESULT ──
    def build_result(self):
        self.result_frame = tk.Frame(self.root,
                                     bg="white",
                                     relief='solid', bd=1)
        self.result_frame.pack(fill='x',
                               padx=16, pady=(0,16))
        inner = tk.Frame(self.result_frame,
                         bg="white", padx=20, pady=16)
        inner.pack(fill='x')

        self.grade_label = tk.Label(inner, text="—",
                                     font=("Arial", 56, "bold"),
                                     bg="white", fg="#1a73e8")
        self.grade_label.pack()

        self.meaning_label = tk.Label(inner,
                                       text="Enter details and click Predict",
                                       font=("Arial", 11),
                                       bg="white", fg="#666",
                                       wraplength=500)
        self.meaning_label.pack(pady=(4,8))

        prob_frame = tk.Frame(inner, bg="white")
        prob_frame.pack(fill='x')

        self.prob_labels = {}
        self.prob_bars   = {}

        for grade, color in [('A','#0f9d58'),
                              ('B','#f4b400'),
                              ('C','#db4437')]:
            row = tk.Frame(prob_frame, bg="white")
            row.pack(fill='x', pady=3)

            tk.Label(row, text=f"Grade {grade}",
                     font=("Arial", 9, "bold"),
                     bg="white", fg=color,
                     width=8).pack(side='left')

            bar_bg = tk.Frame(row, bg="#f0f0f0",
                              height=18, relief='flat')
            bar_bg.pack(side='left', fill='x',
                        expand=True, padx=(4,8))
            bar_bg.pack_propagate(False)

            bar = tk.Frame(bar_bg, bg=color, height=18)
            bar.place(x=0, y=0,
                      relwidth=0.0, relheight=1.0)
            self.prob_bars[grade] = bar

            lbl = tk.Label(row, text="0.0%",
                           font=("Arial", 9),
                           bg="white", fg="#666", width=6)
            lbl.pack(side='left')
            self.prob_labels[grade] = lbl

    # ── PREDICT ──
    def predict(self):
        try:
            insp = self.insp_type.get()
            is_initial      = 1 if insp=="Initial"      else 0
            is_reinspection = 1 if insp=="Reinspection" else 0
            is_reopening    = 1 if insp=="Reopening"    else 0

            vcode         = self.violation_code.get()
            vcode_encoded = violation_map.get(vcode, 0)
            is_severe     = 1 if vcode.startswith('0') else 0

            months = ["January","February","March",
                      "April","May","June","July",
                      "August","September","October",
                      "November","December"]
            month_num = months.index(self.month.get()) + 1

            zc = int(self.zipcode.get())
            complaints = zipcode_complaints.get(
                zc, median_complaints)

            inputs = {
                'violation_code_encoded'  : vcode_encoded,
                'food_complaints_zipcode' : complaints,
                'inspection_month'        : month_num,
                'inspection_year'         : int(self.year.get()),
                'is_reinspection'         : is_reinspection,
                'is_reopening'            : is_reopening,
                'is_initial'              : is_initial,
                'is_critical'             : 1 if self.critical.get()=="Yes" else 0,
                'violation_count'         : float(self.violation_count.get()),
                'is_severe_violation'     : is_severe,
            }

            grade, conf, pa, pb, pc = predict_grade(inputs)

            self.grade_label.config(
                text=grade, fg=colors[grade])
            self.meaning_label.config(
                text=f"{labels[grade]}   |   Confidence: {conf}%",
                fg=colors[grade])

            for g, prob in [('A',pa),('B',pb),('C',pc)]:
                self.prob_labels[g].config(text=f"{prob}%")
                self.prob_bars[g].place(
                    x=0, y=0,
                    relwidth=prob/100,
                    relheight=1.0)

        except ValueError:
            messagebox.showerror("Error",
                "Please enter a valid number\n"
                "for Total Violations Found")
        except Exception as e:
            messagebox.showerror("Error", str(e))

# ══════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app  = PredictUI(root)
    root.mainloop()