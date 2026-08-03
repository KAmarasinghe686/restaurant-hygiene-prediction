import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('restaurant_hygiene_clean.csv')
print(f"Dataset shape: {df.shape}")
print(f"Grade distribution:\n{df['grade'].value_counts()}")

# ══════════════════════════════════════════
# PREPARE TRANSACTIONS
# Include grade as item so rules can
# predict grade as consequent
# ══════════════════════════════════════════
print("\nPreparing transactions...")

transactions = []
for _, row in df.iterrows():
    transaction = [
        f"violation_{row['violation_code']}",
        f"critical_{row['critical_flag']}",
        f"reinspection_{row['is_reinspection']}",
        f"initial_{row['is_initial']}",
        f"reopening_{row['is_reopening']}",
        f"grade_{row['grade']}"            # ← grade included
    ]
    transactions.append(transaction)

print(f"Total transactions: {len(transactions)}")

te       = TransactionEncoder()
te_array = te.fit_transform(transactions)
df_enc   = pd.DataFrame(te_array, columns=te.columns_)
print(f"Encoded shape    : {df_enc.shape}")

# ══════════════════════════════════════════
# RUN APRIORI
# ══════════════════════════════════════════
print("\nRunning Apriori...")

frequent = apriori(
    df_enc,
    min_support=0.02,
    use_colnames=True,
    max_len=4
)
print(f"Frequent itemsets: {len(frequent)}")

rules = association_rules(
    frequent,
    metric="confidence",
    min_threshold=0.3
)
print(f"Total rules      : {len(rules)}")

# ══════════════════════════════════════════
# FILTER — Only rules predicting grade
# consequent must be grade_A, grade_B, grade_C
# ══════════════════════════════════════════
grade_rules = rules[
    rules['consequents'].astype(str).str.contains('grade_')
].copy()

# Remove rules where antecedent also contains grade
grade_rules = grade_rules[
    ~grade_rules['antecedents'].astype(str).str.contains('grade_')
]

# Keep only lift > 1.2
grade_rules = grade_rules[grade_rules['lift'] > 1.2]

# Sort by lift
grade_rules = grade_rules.sort_values('lift', ascending=False)

print(f"\nGrade-predicting rules: {len(grade_rules)}")

# ══════════════════════════════════════════
# SPLIT BY PREDICTED GRADE
# ══════════════════════════════════════════
rules_pred_a = grade_rules[
    grade_rules['consequents'].astype(str).str.contains('grade_A')
]
rules_pred_b = grade_rules[
    grade_rules['consequents'].astype(str).str.contains('grade_B')
]
rules_pred_c = grade_rules[
    grade_rules['consequents'].astype(str).str.contains('grade_C')
]

print(f"\nRules predicting Grade A: {len(rules_pred_a)}")
print(f"Rules predicting Grade B: {len(rules_pred_b)}")
print(f"Rules predicting Grade C: {len(rules_pred_c)}")

# ══════════════════════════════════════════
# PRINT TOP RULES PER GRADE
# ══════════════════════════════════════════
def print_top_rules(rules_df, grade, n=5):
    print(f"\n{'='*60}")
    print(f"TOP RULES PREDICTING GRADE {grade}")
    print(f"{'='*60}")
    if rules_df.empty:
        print("No rules found")
        return
    for i, (_, r) in enumerate(rules_df.head(n).iterrows()):
        ant = list(r['antecedents'])
        con = list(r['consequents'])
        print(f"\nRule {i+1}:")
        print(f"  IF   : {ant}")
        print(f"  THEN : {con}")
        print(f"  Support   : {r['support']:.3f} "
              f"({r['support']*len(df):.0f} restaurants)")
        print(f"  Confidence: {r['confidence']:.3f} "
              f"({r['confidence']*100:.1f}%)")
        print(f"  Lift      : {r['lift']:.3f}")

print_top_rules(rules_pred_a, 'A')
print_top_rules(rules_pred_b, 'B')
print_top_rules(rules_pred_c, 'C')

# ══════════════════════════════════════════
# PLOT 1: Top rules by lift per grade
# ══════════════════════════════════════════
colors = {'A': '#2ecc71', 'B': '#f39c12', 'C': '#e74c3c'}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (grade, rules_df) in zip(axes, [
    ('A', rules_pred_a),
    ('B', rules_pred_b),
    ('C', rules_pred_c)
]):
    if not rules_df.empty:
        top = rules_df.head(6).copy()
        labels = []
        for _, r in top.iterrows():
            ant = list(r['antecedents'])
            label = ' + '.join([a.replace('violation_','')
                                  .replace('critical_','crit_')
                                  .replace('reinspection_1','reinsp')
                                  .replace('reinspection_0','no_reinsp')
                                  .replace('initial_1','initial')
                                  .replace('reopening_1','reopen')
                                  for a in ant])[:30]
            labels.append(label)
        bars = ax.barh(labels, top['lift'].values,
                       color=colors[grade], edgecolor='white')
        for bar, val in zip(bars, top['lift'].values):
            ax.text(bar.get_width()+0.02,
                    bar.get_y()+bar.get_height()/2,
                    f'{val:.2f}', va='center', fontsize=9)
        ax.axvline(x=1.0, color='red', linestyle='--',
                   alpha=0.5, label='random (lift=1)')
        ax.set_title(f'Grade {grade} — top rules by lift',
                     fontweight='bold')
        ax.set_xlabel('Lift score')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No rules found',
                ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title(f'Grade {grade}', fontweight='bold')

plt.suptitle('Association Rules Predicting Grade\n'
             '(lift > 1 = better than random chance)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('assoc_plot1_grade_prediction_rules.png', dpi=150)
plt.show()
print("Saved: assoc_plot1_grade_prediction_rules.png")

# ══════════════════════════════════════════
# PLOT 2: Confidence vs Support scatter
# ══════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (grade, rules_df) in zip(axes, [
    ('A', rules_pred_a),
    ('B', rules_pred_b),
    ('C', rules_pred_c)
]):
    if not rules_df.empty:
        sc = ax.scatter(
            rules_df['support'],
            rules_df['confidence'],
            c=rules_df['lift'],
            cmap='RdYlGn', s=80, alpha=0.7
        )
        plt.colorbar(sc, ax=ax, label='Lift')
        ax.set_title(f'Predicting Grade {grade}',
                     fontweight='bold')
        ax.set_xlabel('Support')
        ax.set_ylabel('Confidence')
    else:
        ax.text(0.5, 0.5, 'No rules found',
                ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title(f'Grade {grade}', fontweight='bold')

plt.suptitle('Support vs Confidence for Grade-Predicting Rules',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('assoc_plot2_support_confidence.png', dpi=150)
plt.show()
print("Saved: assoc_plot2_support_confidence.png")

# ══════════════════════════════════════════
# PLOT 3: Top violations per grade
# ══════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, grade in zip(axes, ['A', 'B', 'C']):
    subset = df[df['grade'] == grade]
    top_v  = subset['violation_code'].value_counts().head(8)
    ax.barh(top_v.index, top_v.values,
            color=colors[grade], edgecolor='white')
    ax.set_title(f'Top violations — Grade {grade}',
                 fontweight='bold')
    ax.set_xlabel('Count')
plt.suptitle('Top violation codes per grade',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('assoc_plot3_violations_per_grade.png', dpi=150)
plt.show()
print("Saved: assoc_plot3_violations_per_grade.png")

# ══════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════
print("\n" + "="*60)
print("ASSOCIATION MINING SUMMARY")
print("="*60)
print(f"Algorithm         : Apriori")
print(f"Min support       : 0.02")
print(f"Min confidence    : 0.30")
print(f"Min lift          : 1.20")
print(f"Total rules found : {len(grade_rules)}")
print(f"Grade A rules     : {len(rules_pred_a)}")
print(f"Grade B rules     : {len(rules_pred_b)}")
print(f"Grade C rules     : {len(rules_pred_c)}")

print(f"\nBest rule per grade:")
for grade, rules_df in [('A', rules_pred_a),
                         ('B', rules_pred_b),
                         ('C', rules_pred_c)]:
    if not rules_df.empty:
        top = rules_df.iloc[0]
        print(f"\n  Grade {grade}:")
        print(f"    IF   : {list(top['antecedents'])}")
        print(f"    THEN : Grade {grade}")
        print(f"    Confidence: {top['confidence']*100:.1f}%")
        print(f"    Lift      : {top['lift']:.3f}")
        print(f"    Support   : {top['support']*len(df):.0f} restaurants")
    else:
        print(f"\n  Grade {grade}: No rules found")

print("="*60)
print("\nDONE! 3 association plots saved.")