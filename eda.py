import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('restaurant_hygiene_clean.csv')
df['inspection_date'] = pd.to_datetime(df['inspection_date'])
print(f"Dataset loaded: {df.shape}")

colors = {'A': '#2ecc71', 'B': '#f39c12', 'C': '#e74c3c'}
plt.style.use('seaborn-v0_8')

# PLOT 1: Grade Distribution
plt.figure(figsize=(8,5))
grade_counts = df['grade'].value_counts()
bars = plt.bar(grade_counts.index, grade_counts.values,
               color=[colors[g] for g in grade_counts.index],
               edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, grade_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 100,
             f'{val:,}\n({val/len(df)*100:.1f}%)',
             ha='center', fontsize=11, fontweight='bold')
plt.title('Grade Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Hygiene Grade')
plt.ylabel('Number of Inspections')
plt.tight_layout()
plt.savefig('eda_plot1_grade_distribution.png', dpi=150)
plt.show()
print("Saved: eda_plot1_grade_distribution.png")


# PLOT 2: Inspection Type vs Grade
df['insp_group'] = 'Other'
df.loc[df['is_initial']==1,      'insp_group'] = 'Initial'
df.loc[df['is_reinspection']==1, 'insp_group'] = 'Reinspection'
df.loc[df['is_reopening']==1,    'insp_group'] = 'Reopening'

insp_grade = df.groupby(['insp_group','grade']).size().unstack(fill_value=0)
insp_pct   = insp_grade.div(insp_grade.sum(axis=1), axis=0) * 100

plt.figure(figsize=(9,5))
insp_pct[[c for c in ['A','B','C'] if c in insp_pct.columns]].plot(
    kind='bar',
    color=[colors[g] for g in ['A','B','C'] if g in insp_pct.columns],
    edgecolor='white', figsize=(9,5))
plt.title('Inspection Type vs Grade', fontsize=14, fontweight='bold')
plt.xlabel('Inspection Type')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=0)
plt.legend(title='Grade')
plt.tight_layout()
plt.savefig('eda_plot2_inspection_type_grade.png', dpi=150)
plt.show()
print("Saved: eda_plot2_inspection_type_grade.png")

# PLOT 3: Critical Flag vs Grade
crit_grade = df.groupby(['critical_flag','grade']).size().unstack(fill_value=0)
crit_pct   = crit_grade.div(crit_grade.sum(axis=1), axis=0) * 100

plt.figure(figsize=(8,5))
crit_pct[[c for c in ['A','B','C'] if c in crit_pct.columns]].plot(
    kind='bar',
    color=[colors[g] for g in ['A','B','C'] if g in crit_pct.columns],
    edgecolor='white', figsize=(8,5))
plt.title('Critical Flag vs Grade', fontsize=14, fontweight='bold')
plt.xlabel('Critical Flag')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=0)
plt.legend(title='Grade')
plt.tight_layout()
plt.savefig('eda_plot3_critical_flag_grade.png', dpi=150)
plt.show()
print("Saved: eda_plot3_critical_flag_grade.png")


# PLOT 4: Top 15 Cuisines by Violation Rate
cuisine_grade = df.groupby('cuisine_description')['grade'].value_counts(
    normalize=True).unstack(fill_value=0)
cuisine_grade['violation_rate'] = cuisine_grade.get('B',0) + cuisine_grade.get('C',0)
top_c = cuisine_grade['violation_rate'].sort_values(ascending=False).head(15)

plt.figure(figsize=(12,6))
bars = plt.barh(top_c.index, top_c.values*100,
                color='#e74c3c', edgecolor='white')
for bar, val in zip(bars, top_c.values):
    plt.text(bar.get_width()+0.3,
             bar.get_y()+bar.get_height()/2,
             f'{val*100:.1f}%', va='center', fontsize=9)
plt.title('Top 15 Cuisines by Violation Rate (B or C Grade)',
          fontsize=14, fontweight='bold')
plt.xlabel('Violation Rate (%)')
plt.tight_layout()
plt.savefig('eda_plot4_cuisine_violations.png', dpi=150)
plt.show()
print("Saved: eda_plot4_cuisine_violations.png")


# PLOT 5: Monthly Grade Trends
monthly     = df.groupby(['inspection_month','grade']).size().unstack(fill_value=0)
monthly_pct = monthly.div(monthly.sum(axis=1), axis=0) * 100
month_names = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']

plt.figure(figsize=(12,5))
for grade in ['A','B','C']:
    if grade in monthly_pct.columns:
        plt.plot(monthly_pct.index, monthly_pct[grade],
                 marker='o', label=f'Grade {grade}',
                 color=colors[grade], linewidth=2.5)
plt.title('Monthly Grade Trends', fontsize=14, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Percentage (%)')
plt.xticks(range(1,13), month_names)
plt.legend(title='Grade')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('eda_plot5_monthly_trends.png', dpi=150)
plt.show()
print("Saved: eda_plot5_monthly_trends.png")


# PLOT 6: Binary Features vs Grade
fig, axes = plt.subplots(1, 3, figsize=(15,5))
for ax, col, title in zip(axes,
    ['is_reinspection', 'is_reopening', 'is_critical'],
    ['Reinspection vs Grade', 'Reopening vs Grade', 'Critical vs Grade']):
    data     = df.groupby([col,'grade']).size().unstack(fill_value=0)
    data_pct = data.div(data.sum(axis=1), axis=0) * 100
    data_pct[[c for c in ['A','B','C'] if c in data_pct.columns]].plot(
        kind='bar', ax=ax,
        color=[colors[g] for g in ['A','B','C'] if g in data_pct.columns],
        edgecolor='white')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Percentage (%)')
    ax.set_xticklabels(['No','Yes'], rotation=0)
    ax.legend(title='Grade')
plt.tight_layout()
plt.savefig('eda_plot6_binary_features.png', dpi=150)
plt.show()
print("Saved: eda_plot6_binary_features.png")

# PLOT 7: Food Complaints vs Grade
plt.figure(figsize=(8,5))
for grade in ['A','B','C']:
    subset = df[df['grade']==grade]['food_complaints_zipcode']
    plt.hist(subset, bins=30, alpha=0.6,
             label=f'Grade {grade}',
             color=colors[grade], edgecolor='white')
plt.title('Food Complaints by Zipcode vs Grade',
          fontsize=14, fontweight='bold')
plt.xlabel('Food Complaints Count')
plt.ylabel('Frequency')
plt.legend(title='Grade')
plt.tight_layout()
plt.savefig('eda_plot7_complaints_grade.png', dpi=150)
plt.show()
print("Saved: eda_plot7_complaints_grade.png")


# PLOT 8: Correlation Heatmap
numeric_cols = [
    'violation_code_encoded',
    'food_complaints_zipcode',
    'inspection_month',
    'inspection_year',
    'is_reinspection',
    'is_reopening',
    'is_initial',
    'is_critical',
    'grade_encoded'
]
numeric_cols = [c for c in numeric_cols if c in df.columns]
corr = df[numeric_cols].corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, square=True, linewidths=0.5,
            annot_kws={'size':9})
plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_plot8_correlation_heatmap.png', dpi=150)
plt.show()
print("Saved: eda_plot8_correlation_heatmap.png")


# PLOT 9: Top Violations Per Grade
fig, axes = plt.subplots(1, 3, figsize=(15,5))
for ax, grade in zip(axes, ['A','B','C']):
    subset = df[df['grade']==grade]
    top_v  = subset['violation_code'].value_counts().head(8)
    ax.barh(top_v.index, top_v.values,
            color=colors[grade], edgecolor='white')
    ax.set_title(f'Top Violations — Grade {grade}', fontweight='bold')
    ax.set_xlabel('Count')
plt.tight_layout()
plt.savefig('eda_plot9_violations_per_grade.png', dpi=150)
plt.show()
print("Saved: eda_plot9_violations_per_grade.png")


# PLOT 10: Top Zipcodes by Grade C Rate
zip_grade       = df.groupby('zipcode')['grade'].value_counts(
    normalize=True).unstack(fill_value=0)
zip_grade['total']  = df.groupby('zipcode').size()
zip_grade['c_rate'] = zip_grade.get('C', 0)
zip_grade           = zip_grade[zip_grade['total'] >= 50]
top_zip             = zip_grade['c_rate'].sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
bars = plt.bar([str(int(z)) for z in top_zip.index],
               top_zip.values * 100,
               color='#e74c3c', edgecolor='white')
for bar, val in zip(bars, top_zip.values):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.3,
             f'{val*100:.1f}%',
             ha='center', fontsize=9)
plt.title('Top 10 Zipcodes with Highest Grade C Rate',
          fontsize=14, fontweight='bold')
plt.xlabel('Zipcode')
plt.ylabel('Grade C Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('eda_plot10_zipcode_hotspots.png', dpi=150)
plt.show()
print("Saved: eda_plot10_zipcode_hotspots.png")


# SUMMARY
print('EDA SUMMARY')
print(f'Total inspections     : {len(df):,}')
print(f'Unique cuisines       : {df["cuisine_description"].nunique()}')
print(f'Unique zipcodes       : {df["zipcode"].nunique()}')
print(f'Unique violation codes: {df["violation_code"].nunique()}')
print(f'Date range            : {df["inspection_date"].min().date()} to {df["inspection_date"].max().date()}')
print(f'\nGrade distribution:')
for g, c in df['grade'].value_counts().items():
    print(f'  Grade {g}: {c:,} ({c/len(df)*100:.1f}%)')
print(f'\nInspection types:')
for t, c in df['insp_group'].value_counts().items():
    print(f'  {t:<15}: {c:,} ({c/len(df)*100:.1f}%)')
print(f'\nCritical violations   : {df["is_critical"].sum():,} ({df["is_critical"].mean()*100:.1f}%)')
print(f'Reinspections         : {df["is_reinspection"].sum():,} ({df["is_reinspection"].mean()*100:.1f}%)')
print(f'Reopening inspections : {df["is_reopening"].sum():,} ({df["is_reopening"].mean()*100:.1f}%)')
print(f'\nTop correlations with grade:')
grade_corr = corr['grade_encoded'].drop('grade_encoded').abs().sort_values(ascending=False)
for col, val in grade_corr.items():
    bar = '█' * int(val * 30)
    print(f'  {col:<30} {val:.4f}  {bar}')
print('='*50)
print('\nDONE! 10 EDA plots saved.')
# After creating insp_grade
print("Counts per inspection type and grade:")
print(insp_grade)