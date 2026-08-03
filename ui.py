import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.utils.class_weight import compute_sample_weight
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════
# GLOBAL STATE
# ══════════════════════════════════════════
df_global       = None
results_global  = {}
best_model_name = ""
models_trained  = {}
feature_cols    = []
scaler_global   = StandardScaler()

# ══════════════════════════════════════════
# ML FUNCTIONS
# ══════════════════════════════════════════

def prepare_features(df):
    df = df.copy()
    df['inspection_date'] = pd.to_datetime(df['inspection_date'])

    # Add engineered features
    df['violation_count'] = df.groupby(
        ['inspection_date','zipcode','is_reinspection']
    )['violation_code_encoded'].transform('count')

    df['is_severe_violation'] = df['violation_code'].str.startswith('0').astype(int)
    return df

def run_models(df):
    global results_global, best_model_name, feature_cols, models_trained, scaler_global

    df = prepare_features(df)

    cols = [
        'violation_code_encoded',
        'food_complaints_zipcode',
        'inspection_month',
        'inspection_year',
        'is_reinspection',
        'is_reopening',
        'is_initial',
        'is_critical',
        'violation_count',
        'is_severe_violation',
    ]
    feature_cols = [c for c in cols if c in df.columns]

    X = df[feature_cols].fillna(0)
    y = df['grade_encoded']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler_global  = StandardScaler()
    X_train_sc     = scaler_global.fit_transform(X_train)
    X_test_sc      = scaler_global.transform(X_test)

    sample_weights = compute_sample_weight('balanced', y_train)
    skf            = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    try:
        from xgboost import XGBClassifier
        xgb_ok = True
    except ImportError:
        xgb_ok = False

    model_defs = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight='balanced'),
        "Decision Tree"      : DecisionTreeClassifier(
            max_depth=8, random_state=42, class_weight='balanced'),
        "Random Forest"      : RandomForestClassifier(
            n_estimators=200, random_state=42,
            class_weight='balanced', n_jobs=-1),
        "Gradient Boosting"  : GradientBoostingClassifier(
            n_estimators=200, max_depth=5,
            learning_rate=0.1, random_state=42),
    }
    if xgb_ok:
        model_defs["XGBoost"] = XGBClassifier(
            n_estimators=200, random_state=42,
            eval_metric='mlogloss', verbosity=0)

    results_global = {}
    models_trained = {}

    for name, model in model_defs.items():
        try:
            if name == "Logistic Regression":
                model.fit(X_train_sc, y_train)
                y_pred = model.predict(X_test_sc)
                cv     = cross_val_score(model, X_train_sc, y_train,
                                          cv=skf, scoring='accuracy')
            elif name in ["Gradient Boosting", "XGBoost"]:
                model.fit(X_train, y_train, sample_weight=sample_weights)
                y_pred = model.predict(X_test)
                cv     = cross_val_score(model, X_train, y_train,
                                          cv=skf, scoring='accuracy')
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                cv     = cross_val_score(model, X_train, y_train,
                                          cv=skf, scoring='accuracy')

            results_global[name] = {
                'accuracy' : round(accuracy_score(y_test, y_pred)*100, 2),
                'f1'       : round(f1_score(y_test, y_pred, average='weighted'), 4),
                'recall'   : round(recall_score(y_test, y_pred, average='weighted'), 4),
                'precision': round(precision_score(y_test, y_pred,
                                   average='weighted', zero_division=0), 4),
                'error'    : round((1-accuracy_score(y_test, y_pred))*100, 2),
                'cv'       : round(cv.mean()*100, 2),
                'cv_std'   : round(cv.std()*100, 2),
            }
            models_trained[name] = model

        except Exception as e:
            print(f"Model error {name}: {e}")

    best_model_name = max(results_global, key=lambda x: results_global[x]['f1'])

    # Feature importance
    for m in ["Gradient Boosting","XGBoost","Random Forest"]:
        if m in models_trained and hasattr(models_trained[m], 'feature_importances_'):
            return dict(zip(feature_cols, models_trained[m].feature_importances_))
    return {}

def predict_grade(row_values):
    if not models_trained or best_model_name not in models_trained:
        return "No model", 0.0

    row = pd.DataFrame([row_values], columns=feature_cols)
    row = row.fillna(0)

    model = models_trained[best_model_name]

    if best_model_name == "Logistic Regression":
        row_sc = scaler_global.transform(row)
        pred   = model.predict(row_sc)[0]
        proba  = model.predict_proba(row_sc)[0]
    else:
        pred  = model.predict(row)[0]
        proba = model.predict_proba(row)[0] if hasattr(model,'predict_proba') else [0,0,0]

    grade_map = {0:'A', 1:'B', 2:'C'}
    grade     = grade_map.get(pred, '?')
    conf      = round(max(proba)*100, 1)
    return grade, conf

# ══════════════════════════════════════════
# UI CLASS
# ══════════════════════════════════════════

class HygieneUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Restaurant Hygiene Grade Prediction — NYC")
        self.root.geometry("1000x780")
        self.root.configure(bg="#F0F2F5")
        self.file_path = None
        self.importances = {}
        self.build_header()
        self.build_notebook()

    # ── HEADER ──
    def build_header(self):
        hdr = tk.Frame(self.root, bg="#1a73e8", pady=14)
        hdr.pack(fill='x')
        tk.Label(hdr,
                 text="🍽  Restaurant Hygiene Grade Prediction",
                 font=("Arial", 17, "bold"),
                 bg="#1a73e8", fg="white").pack()
        tk.Label(hdr,
                 text="NYC Open Data — Queens 2023–2025  |  CMIS 3253 Data Mining",
                 font=("Arial", 10),
                 bg="#1a73e8", fg="#c8e0fc").pack()

    # ── NOTEBOOK ──
    def build_notebook(self):
        style = ttk.Style()
        style.configure('TNotebook.Tab', font=('Arial', 10, 'bold'), padding=[12,6])

        nb = ttk.Notebook(self.root)
        nb.pack(fill='both', expand=True, padx=12, pady=10)

        self.tab1 = tk.Frame(nb, bg="#F0F2F5")
        self.tab2 = tk.Frame(nb, bg="#F0F2F5")
        self.tab3 = tk.Frame(nb, bg="#F0F2F5")
        self.tab4 = tk.Frame(nb, bg="#F0F2F5")

        nb.add(self.tab1, text="  Load & Run  ")
        nb.add(self.tab2, text="  Model Results  ")
        nb.add(self.tab3, text="  Predict Grade  ")
        nb.add(self.tab4, text="  Key Insights  ")

        self.build_tab1()
        self.build_tab2()
        self.build_tab3()
        self.build_tab4()

    # ════════════════════════════════
    # TAB 1: LOAD & RUN
    # ════════════════════════════════
    def build_tab1(self):
        frame = tk.Frame(self.tab1, bg="#F0F2F5", padx=20, pady=20)
        frame.pack(fill='both', expand=True)

        tk.Label(frame, text="Load Dataset & Run Models",
                 font=("Arial", 14, "bold"),
                 bg="#F0F2F5").pack(anchor='w')
        tk.Label(frame,
                 text="Select restaurant_hygiene_clean.csv to train all models",
                 font=("Arial", 10), bg="#F0F2F5", fg="#666").pack(anchor='w', pady=(2,16))

        # File card
        file_card = tk.Frame(frame, bg="white", relief='solid', bd=1)
        file_card.pack(fill='x', pady=(0,12))
        inner = tk.Frame(file_card, bg="white", padx=16, pady=12)
        inner.pack(fill='x')

        self.file_label = tk.Label(inner, text="No file selected",
                                   font=("Arial", 10), bg="white", fg="#999")
        self.file_label.pack(anchor='w')

        btn_row = tk.Frame(inner, bg="white")
        btn_row.pack(anchor='w', pady=(8,0))

        tk.Button(btn_row, text="Browse CSV",
                  command=self.browse_file,
                  bg="#1a73e8", fg="white", relief='flat',
                  padx=16, pady=7,
                  font=("Arial", 10)).pack(side='left', padx=(0,8))

        tk.Button(btn_row, text="Run All Models",
                  command=self.run_all,
                  bg="#0f9d58", fg="white", relief='flat',
                  padx=16, pady=7,
                  font=("Arial", 10, "bold")).pack(side='left')

        self.status_label = tk.Label(frame, text="",
                                     font=("Arial", 10),
                                     bg="#F0F2F5", fg="#0f9d58")
        self.status_label.pack(anchor='w', pady=6)

        self.progress = ttk.Progressbar(frame, length=500, mode='determinate')
        self.progress.pack(anchor='w', pady=(0,16))

        # Summary metric cards
        self.stats_frame = tk.Frame(frame, bg="#F0F2F5")
        self.stats_frame.pack(fill='x')

        self.stat_vars = {}
        stats = [
            ("Total Records",  "rows",  "#1a73e8"),
            ("Best Accuracy",  "acc",   "#0f9d58"),
            ("Best F1 Score",  "f1",    "#f4b400"),
            ("Error Rate",     "err",   "#db4437"),
        ]

        for i, (label, key, color) in enumerate(stats):
            card = tk.Frame(self.stats_frame, bg="white",
                            relief='solid', bd=1, padx=16, pady=12)
            card.grid(row=0, column=i, padx=6, pady=4, sticky='nsew')
            self.stats_frame.grid_columnconfigure(i, weight=1)
            tk.Label(card, text=label, font=("Arial", 9),
                     bg="white", fg="#666").pack()
            var = tk.StringVar(value="—")
            self.stat_vars[key] = var
            tk.Label(card, textvariable=var,
                     font=("Arial", 20, "bold"),
                     bg="white", fg=color).pack()

        # Dataset info
        self.info_frame = tk.Frame(frame, bg="white",
                                   relief='solid', bd=1)
        self.info_frame.pack(fill='x', pady=(16,0))
        self.info_label = tk.Label(self.info_frame,
                                   text="Load a CSV file to see dataset information",
                                   font=("Arial", 10), bg="white",
                                   fg="#666", pady=12)
        self.info_label.pack()

    def browse_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("CSV files","*.csv"), ("All files","*.*")]
        )
        if path:
            self.file_path = path
            fname = path.split("/")[-1].split("\\")[-1]
            self.file_label.config(text=f"✅  {fname}", fg="#333")

    def run_all(self):
        global df_global
        if not self.file_path:
            messagebox.showerror("Error", "Please select a CSV file first")
            return

        self.status_label.config(text="Loading dataset...")
        self.progress['value'] = 10
        self.root.update()

        try:
            df_global = pd.read_csv(self.file_path)
            n = len(df_global)
            self.status_label.config(
                text=f"Loaded {n:,} rows. Training models — please wait...")
            self.progress['value'] = 25
            self.root.update()

            self.importances = run_models(df_global)

            self.progress['value'] = 100
            best = results_global[best_model_name]
            self.status_label.config(
                text=f"✅  Done!  Best model: {best_model_name} — {best['accuracy']}% accuracy")

            self.stat_vars['rows'].set(f"{n:,}")
            self.stat_vars['acc'].set(f"{best['accuracy']}%")
            self.stat_vars['f1'].set(f"{best['f1']}")
            self.stat_vars['err'].set(f"{best['error']}%")

            grades = df_global['grade'].value_counts()
            self.info_label.config(
                text=(f"Rows: {n:,}   |   "
                      f"Grade A: {grades.get('A',0):,}  "
                      f"Grade B: {grades.get('B',0):,}  "
                      f"Grade C: {grades.get('C',0):,}   |   "
                      f"Features: {len(feature_cols)}   |   "
                      f"Best model: {best_model_name}"),
                fg="#333"
            )

            self.refresh_results()

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_label.config(text=f"Error: {e}", fg="red")

    # ════════════════════════════════
    # TAB 2: MODEL RESULTS
    # ════════════════════════════════
    def build_tab2(self):
        frame = tk.Frame(self.tab2, bg="#F0F2F5", padx=20, pady=20)
        frame.pack(fill='both', expand=True)

        tk.Label(frame, text="Model Results",
                 font=("Arial", 14, "bold"),
                 bg="#F0F2F5").pack(anchor='w', pady=(0,12))

        # Results table
        cols = ("Model","Accuracy","F1 Score","Recall",
                "Precision","Error Rate","CV Score")
        self.tree = ttk.Treeview(frame, columns=cols,
                                  show='headings', height=7)
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=115, anchor='center')
        self.tree.pack(fill='x', pady=(0,16))

        # Feature importance
        tk.Label(frame, text="Feature Importance (Best Model)",
                 font=("Arial", 12, "bold"),
                 bg="#F0F2F5").pack(anchor='w', pady=(0,6))

        self.canvas_imp = tk.Canvas(frame, bg="white",
                                     height=240, relief='solid', bd=1)
        self.canvas_imp.pack(fill='x')

        tk.Label(frame,
                 text="Most important: is_initial (initial inspections always Grade A)  |  "
                      "Red = high importance  Blue = medium  Grey = low",
                 font=("Arial", 9), bg="#F0F2F5", fg="#666",
                 wraplength=700).pack(anchor='w', pady=(6,0))

    def refresh_results(self):
        for row in self.tree.get_children():
            self.tree.delete(row)

        for name, r in sorted(results_global.items(),
                               key=lambda x: x[1]['accuracy'], reverse=True):
            tag = 'best' if name == best_model_name else ''
            self.tree.insert('', 'end', values=(
                name,
                f"{r['accuracy']}%",
                f"{r['f1']}",
                f"{r['recall']}",
                f"{r['precision']}",
                f"{r['error']}%",
                f"{r['cv']}%±{r['cv_std']}%"
            ), tags=(tag,))

        self.tree.tag_configure('best', background='#e6f4ea',
                                 foreground='#0f9d58')

        if self.importances:
            self.draw_importance()

    def draw_importance(self):
        c = self.canvas_imp
        c.delete("all")

        sorted_imp = sorted(self.importances.items(),
                             key=lambda x: x[1], reverse=True)[:8]
        max_val    = max(v for _, v in sorted_imp) if sorted_imp else 1
        margin     = 170
        bar_h      = 22
        gap        = 6

        for i, (feat, val) in enumerate(sorted_imp):
            y     = 12 + i*(bar_h+gap)
            bar_w = int((val/max_val) * 560)
            color = "#db4437" if val > 0.15 else \
                    "#1a73e8" if val > 0.05 else "#9aa0a6"

            c.create_text(margin-8, y+bar_h//2,
                          text=feat, anchor='e',
                          font=("Arial", 9), fill="#444")
            c.create_rectangle(margin, y, margin+bar_w, y+bar_h,
                                fill=color, outline="")
            c.create_text(margin+bar_w+6, y+bar_h//2,
                          text=f"{val:.4f}", anchor='w',
                          font=("Arial", 8), fill="#666")

    # ════════════════════════════════
    # TAB 3: PREDICT GRADE
    # ════════════════════════════════
    def build_tab3(self):
        frame = tk.Frame(self.tab3, bg="#F0F2F5", padx=20, pady=20)
        frame.pack(fill='both', expand=True)

        tk.Label(frame, text="Predict Next Inspection Grade",
                 font=("Arial", 14, "bold"),
                 bg="#F0F2F5").pack(anchor='w')
        tk.Label(frame,
                 text="Enter restaurant inspection details to predict the hygiene grade",
                 font=("Arial", 10), bg="#F0F2F5", fg="#666").pack(
                 anchor='w', pady=(2,16))

        # Input grid
        grid = tk.Frame(frame, bg="#F0F2F5")
        grid.pack(fill='x', pady=(0,12))

        fields = [
            ("Violation Code Encoded (0–61)",  "violation_code_encoded", "20"),
            ("Food Complaints in Zipcode",      "food_complaints_zipcode","250"),
            ("Inspection Month (1–12)",         "inspection_month",       "8"),
            ("Inspection Year",                 "inspection_year",        "2024"),
            ("Is Reinspection? (1=Yes, 0=No)", "is_reinspection",        "0"),
            ("Is Reopening? (1=Yes, 0=No)",    "is_reopening",           "0"),
            ("Is Initial? (1=Yes, 0=No)",      "is_initial",             "1"),
            ("Is Critical? (1=Yes, 0=No)",     "is_critical",            "0"),
            ("Violation Count",                "violation_count",        "2"),
            ("Is Severe Violation? (1=Yes)",   "is_severe_violation",    "0"),
        ]

        self.pred_vars = {}
        for i, (label, key, default) in enumerate(fields):
            row = i // 2
            col = i %  2
            cell = tk.Frame(grid, bg="#F0F2F5", padx=8, pady=4)
            cell.grid(row=row, column=col, sticky='w')
            tk.Label(cell, text=label, font=("Arial", 9),
                     bg="#F0F2F5", fg="#555").pack(anchor='w')
            var = tk.StringVar(value=default)
            self.pred_vars[key] = var
            tk.Entry(cell, textvariable=var, width=28,
                     font=("Arial", 10)).pack(anchor='w')

        tk.Button(frame, text="  Predict Grade  ",
                  command=self.predict,
                  bg="#1a73e8", fg="white", relief='flat',
                  padx=20, pady=8,
                  font=("Arial", 11, "bold")).pack(anchor='w', pady=(8,16))

        # Result card
        result_card = tk.Frame(frame, bg="white",
                               relief='solid', bd=1)
        result_card.pack(fill='x')
        inner = tk.Frame(result_card, bg="white", padx=20, pady=16)
        inner.pack(fill='x')

        self.grade_label = tk.Label(inner, text="—",
                                     font=("Arial", 52, "bold"),
                                     bg="white", fg="#1a73e8")
        self.grade_label.pack()

        self.pred_detail = tk.Label(inner,
                                     text="Load CSV and run models first, then enter details above",
                                     font=("Arial", 11),
                                     bg="white", fg="#666",
                                     wraplength=550)
        self.pred_detail.pack(pady=(4,0))

    def predict(self):
        if not models_trained:
            messagebox.showerror("Error", "Please load CSV and run models first")
            return
        try:
            row_values = []
            for col in feature_cols:
                val = float(self.pred_vars[col].get()) \
                      if col in self.pred_vars else 0.0
                row_values.append(val)

            grade, conf = predict_grade(row_values)

            colors = {'A':'#0f9d58', 'B':'#f4b400', 'C':'#db4437'}
            labels = {
                'A': 'Grade A — Excellent hygiene. Restaurant passed.',
                'B': 'Grade B — Needs improvement. Some violations found.',
                'C': 'Grade C — Failed. Serious violations detected.',
            }

            color = colors.get(grade, '#666')
            self.grade_label.config(text=grade, fg=color)
            self.pred_detail.config(
                text=f"{labels.get(grade, '')}   |   Confidence: {conf}%   |   Model: {best_model_name}",
                fg=color)

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ════════════════════════════════
    # TAB 4: KEY INSIGHTS
    # ════════════════════════════════
    def build_tab4(self):
        frame = tk.Frame(self.tab4, bg="#F0F2F5", padx=20, pady=20)
        frame.pack(fill='both', expand=True)

        tk.Label(frame, text="Key Insights From Analysis",
                 font=("Arial", 14, "bold"),
                 bg="#F0F2F5").pack(anchor='w', pady=(0,12))

        insights = [
            ("#db4437", "1",
             "Initial inspections → always Grade A",
             "100% of initial inspections result in Grade A. "
             "is_initial is the most important feature (49% importance)."),

            ("#db4437", "2",
             "Reopening inspections → always Grade C",
             "100% of reopening inspections result in Grade C. "
             "Restaurants closed due to severe violations always fail."),

            ("#1a73e8", "3",
             "Violation 08A during reinspection → strongly predicts Grade C",
             "Association rule: lift=3.0. Restaurants returning for "
             "reinspection with violation 08A are 3x more likely to get Grade C."),

            ("#1a73e8", "4",
             "August is the worst month (21.4% Grade C)",
             "Summer heat increases food spoilage risk. "
             "June is the best month (9.2% Grade C)."),

            ("#f4b400", "5",
             "Zipcode 11040 has the highest Grade C rate (34.4%)",
             "Significant geographic variation. "
             "Worst zipcodes: 11040, 11428, 11355."),

            ("#f4b400", "6",
             "Grade C restaurants average 10.11 violations per inspection",
             "Grade A averages 4.93, Grade B averages 7.59. "
             "More violations strongly correlates with worse grades."),

            ("#0f9d58", "7",
             "Pakistani and Bangladeshi cuisines have highest Grade C rates",
             "Pakistani: 40% Grade C, Bangladeshi: 39%. "
             "Chinese: 23.9%, American: lower rates."),

            ("#0f9d58", "8",
             "Grade A declining year over year (74% → 66%)",
             "2023: 74% Grade A, 2024: 69%, 2025: 66%. "
             "Concerning hygiene trend in Queens restaurants."),
        ]

        for color, rank, title, desc in insights:
            row = tk.Frame(frame, bg="white", relief='solid',
                           bd=1, padx=12, pady=10)
            row.pack(fill='x', pady=3)

            tk.Label(row, text=rank,
                     font=("Arial", 11, "bold"),
                     bg="white", fg=color, width=2).pack(side='left')

            text_frame = tk.Frame(row, bg="white")
            text_frame.pack(side='left', padx=(8,0), fill='x', expand=True)

            tk.Label(text_frame, text=title,
                     font=("Arial", 10, "bold"),
                     bg="white", fg="#333",
                     anchor='w').pack(anchor='w')

            tk.Label(text_frame, text=desc,
                     font=("Arial", 9),
                     bg="white", fg="#666",
                     wraplength=650, anchor='w').pack(anchor='w')

# ══════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app  = HygieneUI(root)
    root.mainloop()