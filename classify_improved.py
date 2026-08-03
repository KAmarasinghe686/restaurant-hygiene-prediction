import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score,
                              recall_score, precision_score)
from sklearn.utils.class_weight import compute_sample_weight
import warnings
warnings.filterwarnings("ignore")


# LOAD DATA
df = pd.read_csv('restaurant_hygiene_clean.csv')
df['inspection_date'] = pd.to_datetime(df['inspection_date'])


print("IMPROVED CLASSIFICATION")
print("No SMOTE — uses class weights + new features")

print(f"Dataset shape : {df.shape}")
print(f"Grade distribution:")
for g, c in df['grade'].value_counts().items():
    print(f"  Grade {g}: {c:,} ({c/len(df)*100:.1f}%)")


# ADD NEW FEATURES
# ══════════════════════════════════════════
print("\n--- Adding new features ---")

# Feature 1: violation_count
df['violation_count'] = df.groupby(
    ['inspection_date', 'zipcode', 'is_reinspection']
)['violation_code_encoded'].transform('count')

print("violation_count mean by grade:")
print(df.groupby('grade')['violation_count'].mean().round(2))

# Feature 2: is_severe_violation
df['is_severe_violation'] = df['violation_code'].str.startswith('0').astype(int)

print("\nis_severe_violation rate by grade:")
print(df.groupby('grade')['is_severe_violation'].mean().round(3))

# ══════════════════════════════════════════
# FEATURES & TARGET
# ══════════════════════════════════════════
feature_cols = [
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

feature_cols = [c for c in feature_cols if c in df.columns]
X = df[feature_cols].fillna(0)
y = df['grade_encoded']

print(f"\nTotal features : {len(feature_cols)}")
print(f"Features       : {feature_cols}")
print(f"Target classes : A=0, B=1, C=2")

# ══════════════════════════════════════════
# TRAIN / TEST SPLIT
# ══════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set   : {len(X_train):,} rows (80%)")
print(f"Testing set    : {len(X_test):,} rows (20%)")
print(f"Class distribution in train:")
for cls, cnt in zip(*np.unique(y_train, return_counts=True)):
    grade = {0:'A', 1:'B', 2:'C'}[cls]
    print(f"  Grade {grade}: {cnt:,} ({cnt/len(y_train)*100:.1f}%)")

# ══════════════════════════════════════════
# SAVE TRAIN / TEST CSV FILES
# ══════════════════════════════════════════
train_df = df.iloc[X_train.index].copy()
test_df  = df.iloc[X_test.index].copy()
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv',   index=False)
print(f"\nSaved: train.csv ({len(train_df):,} rows)")
print(f"Saved: test.csv  ({len(test_df):,} rows)")

# ══════════════════════════════════════════
# CLASS WEIGHTS
# ══════════════════════════════════════════
sample_weights = compute_sample_weight('balanced', y_train)
print(f"\nClass weights applied:")
print(f"  Grade A: {sample_weights[y_train==0][0]:.4f}")
print(f"  Grade B: {sample_weights[y_train==1][0]:.4f}")
print(f"  Grade C: {sample_weights[y_train==2][0]:.4f}")

# Scale for Logistic Regression only
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ══════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════
try:
    from xgboost import XGBClassifier
    xgb_ok = True
except ImportError:
    xgb_ok = False
    print("XGBoost not installed — skipping")

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42,
        class_weight='balanced'),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8, random_state=42,
        class_weight='balanced'),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42,
        class_weight='balanced', n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(
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
        elif name in ["Gradient Boosting", "XGBoost"]:
            model.fit(X_train, y_train,
                      sample_weight=sample_weights)
            y_pred = model.predict(X_test)
            cv     = cross_val_score(model, X_train, y_train,
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
# SAVE BEST MODEL
# ══════════════════════════════════════════
import joblib

joblib.dump(models[best_name], 'best_model.pkl')
joblib.dump(scaler,            'scaler.pkl')
joblib.dump(feature_cols,      'feature_cols.pkl')

print(f"\nModel saved!")
print(f"  best_model.pkl  ← {best_name}")
print(f"  scaler.pkl      ← StandardScaler")
print(f"  feature_cols.pkl ← {len(feature_cols)} features")

# ══════════════════════════════════════════
# PLOT 1: Model Comparison
# ══════════════════════════════════════════
model_names = list(results.keys())
accuracies  = [results[m]['accuracy'] for m in model_names]
f1_scores   = [results[m]['f1']*100   for m in model_names]
error_rates = [results[m]['error']    for m in model_names]
cv_means    = [results[m]['cv_mean']  for m in model_names]
cv_stds     = [results[m]['cv_std']   for m in model_names]

x     = np.arange(len(model_names))
width = 0.25

fig, ax = plt.subplots(figsize=(13,6))
b1 = ax.bar(x-width, accuracies,  width,
            label='Accuracy (%)',   color='#3498db', edgecolor='white')
b2 = ax.bar(x,       f1_scores,   width,
            label='F1 Score (%)',   color='#2ecc71', edgecolor='white')
b3 = ax.bar(x+width, error_rates, width,
            label='Error Rate (%)', color='#e74c3c', edgecolor='white')
for bar in list(b1)+list(b2)+list(b3):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.3,
            f'{bar.get_height():.1f}%',
            ha='center', fontsize=8, fontweight='bold')
ax.set_title('Improved Model Comparison\n'
             '(class weights + violation_count + is_severe_violation)',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=15, ha='right')
ax.set_ylabel('Score (%)')
ax.set_ylim(0, 115)
ax.legend()
plt.tight_layout()
plt.savefig('classify_improved_plot1_comparison.png', dpi=150)
plt.show()
print("Saved: classify_improved_plot1_comparison.png")

# ══════════════════════════════════════════
# PLOT 2: Cross Validation WITH error bars
# ══════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10,5))
bars = ax.bar(model_names, cv_means,
              color='#9b59b6', edgecolor='white',
              yerr=cv_stds,
              capsize=8,
              error_kw={'linewidth':2, 'color':'black'})
for bar, val, std in zip(bars, cv_means, cv_stds):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+std+0.5,
            f'{val:.1f}%\n±{std:.2f}%',
            ha='center', fontsize=9, fontweight='bold')
ax.set_title('5-Fold Cross Validation Scores\n(error bars show stability)',
             fontsize=13, fontweight='bold')
ax.set_xticklabels(model_names, rotation=15, ha='right')
ax.set_ylabel('CV Accuracy (%)')
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig('classify_improved_plot2_cv.png', dpi=150)
plt.show()
print("Saved: classify_improved_plot2_cv.png")

# ══════════════════════════════════════════
# PLOT 3: Confusion Matrix — Best Model
# ══════════════════════════════════════════
cm = confusion_matrix(y_test, predictions[best_name])
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Grade A','Grade B','Grade C'],
            yticklabels=['Grade A','Grade B','Grade C'])
plt.title(f'Confusion Matrix — {best_name} (Improved)',
          fontsize=14, fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('classify_improved_plot3_confusion.png', dpi=150)
plt.show()
print("Saved: classify_improved_plot3_confusion.png")

# ══════════════════════════════════════════
# PLOT 4: Feature Importance
# ══════════════════════════════════════════
best_tree = None
for m in ["Gradient Boosting","XGBoost","Random Forest"]:
    if m in models and hasattr(models[m],'feature_importances_'):
        best_tree = models[m]
        tree_name = m
        break

if best_tree is not None:
    imp_df = pd.DataFrame({
        'Feature'   : feature_cols,
        'Importance': best_tree.feature_importances_
    }).sort_values('Importance', ascending=True)

    plt.figure(figsize=(10,7))
    bar_colors = ['#e74c3c' if v > 0.15 else
                  '#3498db' if v > 0.05 else '#95a5a6'
                  for v in imp_df['Importance']]
    bars = plt.barh(imp_df['Feature'], imp_df['Importance'],
                    color=bar_colors, edgecolor='white')
    for bar, val in zip(bars, imp_df['Importance']):
        plt.text(bar.get_width()+0.001,
                 bar.get_y()+bar.get_height()/2,
                 f'{val:.4f}', va='center', fontsize=9)
    plt.title(f'Feature Importance — {tree_name} (Improved)\n'
              f'red=high importance, blue=medium, grey=low',
              fontsize=13, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('classify_improved_plot4_importance.png', dpi=150)
    plt.show()
    print("Saved: classify_improved_plot4_importance.png")

# ══════════════════════════════════════════
# PLOT 5: Before vs After Recall
# ══════════════════════════════════════════
old_results = {'Grade A': 0.89, 'Grade B': 0.42, 'Grade C': 0.45}

report = classification_report(
    y_test, predictions[best_name],
    target_names=['Grade A','Grade B','Grade C'],
    output_dict=True
)
new_results = {
    'Grade A': report['Grade A']['recall'],
    'Grade B': report['Grade B']['recall'],
    'Grade C': report['Grade C']['recall'],
}

grades   = list(old_results.keys())
old_vals = [old_results[g]*100 for g in grades]
new_vals = [new_results[g]*100 for g in grades]

x     = np.arange(len(grades))
width = 0.35

fig, ax = plt.subplots(figsize=(9,5))
b1 = ax.bar(x-width/2, old_vals, width,
            label='Before (no fix)', color='#95a5a6', edgecolor='white')
b2 = ax.bar(x+width/2, new_vals, width,
            label='After (improved)', color='#3498db', edgecolor='white')
for bar in list(b1)+list(b2):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.5,
            f'{bar.get_height():.1f}%',
            ha='center', fontsize=10, fontweight='bold')
ax.set_title('Per-Class Recall: Before vs After Improvement',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(grades)
ax.set_ylabel('Recall (%)')
ax.set_ylim(0, 110)
ax.legend()
plt.tight_layout()
plt.savefig('classify_improved_plot5_before_after.png', dpi=150)
plt.show()
print("Saved: classify_improved_plot5_before_after.png")

# ══════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════
print("\n" + "="*70)
print("IMPROVED CLASSIFICATION SUMMARY")
print("="*70)
print(f"Improvements applied:")
print(f"  ✅ violation_count     — counts violations per inspection")
print(f"  ✅ is_severe_violation  — flags severe violation codes")
print(f"  ✅ class_weight         — balances class importance")
print(f"  ✅ No SMOTE             — no synthetic data generated")
print(f"  ✅ Train/Test CSV saved — train.csv and test.csv")
print(f"  ✅ CV error bars        — shows model stability")
print()
print(f"{'Model':<25} {'Accuracy':>10} {'F1':>8} {'Recall':>8}"
      f" {'Precision':>10} {'Error':>8} {'CV':>12}")
print("-"*80)
for name in results:
    r = results[name]
    marker = " ←" if name == best_name else ""
    print(f"{name:<25} {r['accuracy']:>9.2f}%"
          f" {r['f1']:>8.4f}"
          f" {r['recall']:>8.4f}"
          f" {r['precision']:>10.4f}"
          f" {r['error']:>7.2f}%"
          f" {r['cv_mean']:>6.2f}%±{r['cv_std']:.2f}%{marker}")
print("-"*80)
print(f"\nBest Model    : {best_name}")
print(f"Best Accuracy : {best['accuracy']}%")
print(f"Best F1 Score : {best['f1']}")
print(f"Error Rate    : {best['error']}%")
print(f"CV Score      : {best['cv_mean']}% ± {best['cv_std']}%")
print("="*70)
print("\nDONE! 5 improved classification plots saved.")
print("      train.csv and test.csv saved.")