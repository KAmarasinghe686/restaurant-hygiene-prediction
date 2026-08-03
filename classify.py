import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score, precision_score
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════
df = pd.read_csv('restaurant_hygiene_clean.csv')
print("=" * 60)
print("CLASSIFICATION — Restaurant Hygiene Grade Prediction")
print("=" * 60)
print(f"Dataset shape : {df.shape}")
print(f"Grade distribution:\n{df['grade'].value_counts()}")

# ══════════════════════════════════════════
# FEATURES & TARGET
# Only use strong features based on correlation analysis
# ══════════════════════════════════════════
feature_cols = [
    'violation_code_encoded',   # 0.20 correlation
    'food_complaints_zipcode',  # 0.12 correlation
    'inspection_month',         # 0.02 correlation
    'inspection_year',          # 0.05 correlation
    'is_reinspection',          # 0.57 correlation ← strong
    'is_reopening',             # 0.10 correlation
    'is_initial',               # 0.58 correlation ← strong
    'is_critical',              # 0.17 correlation
]

feature_cols = [c for c in feature_cols if c in df.columns]
X = df[feature_cols].fillna(0)
y = df['grade_encoded']

print(f"\nFeatures used : {len(feature_cols)}")
print(f"Feature list  : {feature_cols}")
print(f"Target classes: A=0, B=1, C=2")

# ══════════════════════════════════════════
# TRAIN / TEST SPLIT
# ══════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining set  : {len(X_train)} samples")
print(f"Testing set   : {len(X_test)} samples")

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ══════════════════════════════════════════
# TRAIN 5 MODELS
# ══════════════════════════════════════════
try:
    from xgboost import XGBClassifier
    xgb_ok = True
except ImportError:
    xgb_ok = False

models = {
    "Logistic Regression" : LogisticRegression(
        max_iter=1000, random_state=42, class_weight='balanced'),
    "Decision Tree"       : DecisionTreeClassifier(
        max_depth=8, random_state=42, class_weight='balanced'),
    "Random Forest"       : RandomForestClassifier(
        n_estimators=200, random_state=42,
        class_weight='balanced', n_jobs=-1),
    "Gradient Boosting"   : GradientBoostingClassifier(
        n_estimators=200, max_depth=5,
        learning_rate=0.1, random_state=42),
}
if xgb_ok:
    models["XGBoost"] = XGBClassifier(
        n_estimators=200, max_depth=6,
        learning_rate=0.1, random_state=42,
        eval_metric='mlogloss', verbosity=0)

results     = {}
predictions = {}

print("\n" + "="*60)
print("CLASSIFICATION RESULTS")
print("="*60)

for name, model in models.items():
    print(f"\n--- {name} ---")
    try:
        if name == "Logistic Regression":
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)
            cv     = cross_val_score(model, X_train_sc, y_train,
                                      cv=skf, scoring='accuracy')
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            cv     = cross_val_score(model, X_train, y_train,
                                      cv=skf, scoring='accuracy')

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average='weighted')
        rec  = recall_score(y_test, y_pred, average='weighted')
        prec = precision_score(y_test, y_pred, average='weighted',
                               zero_division=0)
        err  = 1 - acc

        results[name]     = {
            'accuracy' : round(acc*100, 2),
            'f1'       : round(f1, 4),
            'recall'   : round(rec, 4),
            'precision': round(prec, 4),
            'error'    : round(err*100, 2),
            'cv_mean'  : round(cv.mean()*100, 2),
            'cv_std'   : round(cv.std()*100, 2),
        }
        predictions[name] = y_pred

        print(f"Accuracy   : {acc*100:.2f}%")
        print(f"F1 Score   : {f1:.4f}")
        print(f"Error Rate : {err*100:.2f}%")
        print(f"CV Score   : {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred,
              target_names=['Grade A','Grade B','Grade C']))

    except Exception as e:
        print(f"ERROR: {e}")

# ══════════════════════════════════════════
# BEST MODEL
# ══════════════════════════════════════════
best_name = max(results, key=lambda x: results[x]['f1'])
best      = results[best_name]

print("\n" + "="*60)
print(f"BEST MODEL    : {best_name}")
print(f"ACCURACY      : {best['accuracy']}%")
print(f"F1 SCORE      : {best['f1']}")
print(f"RECALL        : {best['recall']}")
print(f"PRECISION     : {best['precision']}")
print(f"ERROR RATE    : {best['error']}%")
print(f"CV SCORE      : {best['cv_mean']}% ± {best['cv_std']}%")
print("="*60)

# ══════════════════════════════════════════
# PLOT 1: Model Comparison
# ══════════════════════════════════════════
model_names = list(results.keys())
accuracies  = [results[m]['accuracy'] for m in model_names]
f1_scores   = [results[m]['f1']*100   for m in model_names]
error_rates = [results[m]['error']    for m in model_names]

x     = np.arange(len(model_names))
width = 0.25

fig, ax = plt.subplots(figsize=(13,6))
b1 = ax.bar(x - width, accuracies,  width,
            label='Accuracy (%)',   color='#3498db', edgecolor='white')
b2 = ax.bar(x,         f1_scores,   width,
            label='F1 Score (%)',   color='#2ecc71', edgecolor='white')
b3 = ax.bar(x + width, error_rates, width,
            label='Error Rate (%)', color='#e74c3c', edgecolor='white')

for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3,
            f'{bar.get_height():.1f}%',
            ha='center', fontsize=8, fontweight='bold')
for bar in b2:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3,
            f'{bar.get_height():.1f}%',
            ha='center', fontsize=8, fontweight='bold')
for bar in b3:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3,
            f'{bar.get_height():.1f}%',
            ha='center', fontsize=8, fontweight='bold')

ax.set_title('Model Comparison — Accuracy, F1 Score & Error Rate',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=15, ha='right')
ax.set_ylabel('Score (%)')
ax.set_ylim(0, 115)
ax.legend()
plt.tight_layout()
plt.savefig('classify_plot1_model_comparison.png', dpi=150)
plt.show()
print("Saved: classify_plot1_model_comparison.png")

# ══════════════════════════════════════════
# PLOT 2: Cross Validation Scores
# ══════════════════════════════════════════
cv_means = [results[m]['cv_mean'] for m in model_names]
cv_stds  = [results[m]['cv_std']  for m in model_names]

fig, ax = plt.subplots(figsize=(10,5))
bars = ax.bar(model_names, cv_means,
              color='#9b59b6', edgecolor='white',
              yerr=cv_stds, capsize=5)
for bar, val in zip(bars, cv_means):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f'{val:.1f}%',
            ha='center', fontsize=10, fontweight='bold')
ax.set_title('5-Fold Cross Validation Scores',
             fontsize=14, fontweight='bold')
ax.set_ylabel('CV Accuracy (%)')
ax.set_ylim(0, 110)
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig('classify_plot2_cross_validation.png', dpi=150)
plt.show()
print("Saved: classify_plot2_cross_validation.png")

# ══════════════════════════════════════════
# PLOT 3: Confusion Matrix — Best Model
# ══════════════════════════════════════════
cm = confusion_matrix(y_test, predictions[best_name])
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Grade A','Grade B','Grade C'],
            yticklabels=['Grade A','Grade B','Grade C'])
plt.title(f'Confusion Matrix — {best_name}',
          fontsize=14, fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('classify_plot3_confusion_matrix.png', dpi=150)
plt.show()
print("Saved: classify_plot3_confusion_matrix.png")

# ══════════════════════════════════════════
# PLOT 4: Feature Importance
# ══════════════════════════════════════════
best_tree = None
for m in ["Gradient Boosting","XGBoost","Random Forest"]:
    if m in models and hasattr(models[m], 'feature_importances_'):
        best_tree = models[m]
        tree_name = m
        break

if best_tree is not None:
    imp_df = pd.DataFrame({
        'Feature'   : feature_cols,
        'Importance': best_tree.feature_importances_
    }).sort_values('Importance', ascending=True)

    plt.figure(figsize=(10,6))
    colors_imp = ['#e74c3c' if v > 0.15 else
                  '#3498db' if v > 0.05 else '#95a5a6'
                  for v in imp_df['Importance']]
    bars = plt.barh(imp_df['Feature'], imp_df['Importance'],
                    color=colors_imp, edgecolor='white')
    for bar, val in zip(bars, imp_df['Importance']):
        plt.text(bar.get_width() + 0.001,
                 bar.get_y() + bar.get_height()/2,
                 f'{val:.4f}', va='center', fontsize=9)
    plt.title(f'Feature Importance — {tree_name}\n'
              f'(red=high, blue=medium, grey=low)',
              fontsize=13, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('classify_plot4_feature_importance.png', dpi=150)
    plt.show()
    print("Saved: classify_plot4_feature_importance.png")

# ══════════════════════════════════════════
# FINAL SUMMARY TABLE
# ══════════════════════════════════════════
print("\n" + "="*70)
print("FINAL CLASSIFICATION SUMMARY")
print("="*70)
print(f"{'Model':<25} {'Accuracy':>10} {'F1':>8} {'Recall':>8} {'Precision':>10} {'Error':>8} {'CV':>10}")
print("-"*70)
for name in results:
    r = results[name]
    best_marker = " ←" if name == best_name else ""
    print(f"{name:<25} {r['accuracy']:>9.2f}% "
          f"{r['f1']:>8.4f} "
          f"{r['recall']:>8.4f} "
          f"{r['precision']:>10.4f} "
          f"{r['error']:>7.2f}% "
          f"{r['cv_mean']:>9.2f}%{best_marker}")
print("-"*70)
print(f"\nBest Model    : {best_name}")
print(f"Best Accuracy : {best['accuracy']}%")
print(f"Best F1 Score : {best['f1']}")
print(f"Error Rate    : {best['error']}%")
print(f"CV Score      : {best['cv_mean']}% ± {best['cv_std']}%")
print("="*70)
print("\nDONE! 4 classification plots saved.")