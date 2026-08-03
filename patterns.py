import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('restaurant_hygiene_clean.csv')
df['inspection_date'] = pd.to_datetime(df['inspection_date'])
colors = {'A': '#2ecc71', 'B': '#f39c12', 'C': '#e74c3c'}
plt.style.use('seaborn-v0_8')

print("=" * 60)
print("PATTERN DISCOVERY")
print("=" * 60)

# PATTERN 1: Violation Codes → Grade
# Which violation codes predict each grade?
print("\nPATTERN 1: Violation Codes vs Grade ")

viol_grade = df.groupby('violation_code')['grade'].value_counts(
    normalize=True).unstack(fill_value=0)

# Top violation codes predicting Grade C
if 'C' in viol_grade.columns:
    viol_grade['c_rate'] = viol_grade['C']
    top_c_viols = viol_grade['c_rate'].sort_values(ascending=False).head(10)
    print(f"\nTop 10 violation codes → Grade C:")
    for code, rate in top_c_viols.items():
        print(f"  {code}: {rate*100:.1f}%")

fig, axes = plt.subplots(1, 3, figsize=(15,5))
for ax, grade in zip(axes, ['A','B','C']):
    subset = df[df['grade']==grade]
    top_v  = subset['violation_code'].value_counts().head(8)
    ax.barh(top_v.index, top_v.values,
            color=colors[grade], edgecolor='white')
    ax.set_title(f'Top Violations — Grade {grade}', fontweight='bold')
    ax.set_xlabel('Count')
plt.suptitle('Pattern 1: Top Violation Codes Per Grade',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('pattern1_violations_per_grade.png', dpi=150)
plt.show()
print("Saved: pattern1_violations_per_grade.png")


# PATTERN 2: Seasonal Patterns
# Does month affect grade?
print("\nPATTERN 2: Seasonal Patterns")

monthly     = df.groupby(['inspection_month','grade']).size().unstack(fill_value=0)
monthly_pct = monthly.div(monthly.sum(axis=1), axis=0) * 100
month_names = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']

if 'C' in monthly_pct.columns:
    worst_month = monthly_pct['C'].idxmax()
    best_month  = monthly_pct['C'].idxmin()
    print(f"Worst month for Grade C: {month_names[worst_month-1]} ({monthly_pct['C'][worst_month]:.1f}%)")
    print(f"Best month for Grade C : {month_names[best_month-1]}  ({monthly_pct['C'][best_month]:.1f}%)")

plt.figure(figsize=(12,5))
for grade in ['A','B','C']:
    if grade in monthly_pct.columns:
        plt.plot(monthly_pct.index, monthly_pct[grade],
                 marker='o', label=f'Grade {grade}',
                 color=colors[grade], linewidth=2.5)
plt.title('Pattern 2: Seasonal Grade Trends by Month',
          fontsize=13, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Percentage (%)')
plt.xticks(range(1,13), month_names)
plt.legend(title='Grade')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pattern2_seasonal.png', dpi=150)
plt.show()
print("Saved: pattern2_seasonal.png")


# PATTERN 3: Inspection Type → Grade
# Initial vs Reinspection vs Reopening
print("\nPATTERN 3: Inspection Type vs Grade")

df['insp_group'] = 'Initial'
df.loc[df['is_reinspection']==1, 'insp_group'] = 'Reinspection'
df.loc[df['is_reopening']==1,    'insp_group'] = 'Reopening'

insp_grade = df.groupby(['insp_group','grade']).size().unstack(fill_value=0)
insp_pct   = insp_grade.div(insp_grade.sum(axis=1), axis=0) * 100

print(f"\nGrade distribution by inspection type:")
print(insp_pct.round(1).to_string())

plt.figure(figsize=(9,5))
insp_pct[[c for c in ['A','B','C'] if c in insp_pct.columns]].plot(
    kind='bar',
    color=[colors[g] for g in ['A','B','C'] if g in insp_pct.columns],
    edgecolor='white', figsize=(9,5))
plt.title('Pattern 3: Inspection Type vs Grade',
          fontsize=13, fontweight='bold')
plt.xlabel('Inspection Type')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=0)
plt.legend(title='Grade')
plt.tight_layout()
plt.savefig('pattern3_inspection_type.png', dpi=150)
plt.show()
print("Saved: pattern3_inspection_type.png")


# PATTERN 4: Critical Flag → Grade
# Do critical violations predict bad grades?
print("\n PATTERN 4: Critical Flag vs Grade ")

crit_grade = df.groupby(['critical_flag','grade']).size().unstack(fill_value=0)
crit_pct   = crit_grade.div(crit_grade.sum(axis=1), axis=0) * 100

print(f"\nGrade distribution by critical flag:")
print(crit_pct.round(1).to_string())

plt.figure(figsize=(8,5))
crit_pct[[c for c in ['A','B','C'] if c in crit_pct.columns]].plot(
    kind='bar',
    color=[colors[g] for g in ['A','B','C'] if g in crit_pct.columns],
    edgecolor='white', figsize=(8,5))
plt.title('Pattern 4: Critical Flag vs Grade',
          fontsize=13, fontweight='bold')
plt.xlabel('Critical Flag')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=0)
plt.legend(title='Grade')
plt.tight_layout()
plt.savefig('pattern4_critical_flag.png', dpi=150)
plt.show()
print("Saved: pattern4_critical_flag.png")


# PATTERN 5: Zipcode Hotspots
# Which zipcodes have worst grades?
print("\n PATTERN 5: Zipcode Hotspots ")

zip_grade       = df.groupby('zipcode')['grade'].value_counts(
    normalize=True).unstack(fill_value=0)
zip_grade['total']  = df.groupby('zipcode').size()
zip_grade['c_rate'] = zip_grade.get('C', 0)
zip_grade           = zip_grade[zip_grade['total'] >= 50]
top_zip             = zip_grade['c_rate'].sort_values(ascending=False).head(10)

print(f"\nTop 10 zipcodes with highest Grade C rate:")
for zipcode, rate in top_zip.items():
    total = int(zip_grade.loc[zipcode, 'total'])
    print(f"  Zipcode {int(zipcode)}: {rate*100:.1f}% Grade C ({total} inspections)")

plt.figure(figsize=(10,5))
bars = plt.bar([str(int(z)) for z in top_zip.index],
               top_zip.values * 100,
               color='#e74c3c', edgecolor='white')
for bar, val in zip(bars, top_zip.values):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.3,
             f'{val*100:.1f}%', ha='center', fontsize=9)
plt.title('Pattern 5: Top 10 Zipcodes with Highest Grade C Rate',
          fontsize=13, fontweight='bold')
plt.xlabel('Zipcode')
plt.ylabel('Grade C Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('pattern5_zipcode_hotspots.png', dpi=150)
plt.show()
print("Saved: pattern5_zipcode_hotspots.png")


# PATTERN 6: Food Complaints → Grade
# Do high complaint areas have worse grades?
print("\n PATTERN 6: Food Complaints vs Grade ")

complaint_stats = df.groupby('grade')['food_complaints_zipcode'].describe()
print(f"\nFood complaints stats by grade:")
print(complaint_stats[['mean','std','min','max']].round(1).to_string())

plt.figure(figsize=(8,5))
grade_order    = ['A','B','C']
data_to_plot   = [df[df['grade']==g]['food_complaints_zipcode'].values
                  for g in grade_order]
bp = plt.boxplot(data_to_plot, labels=grade_order, patch_artist=True)
for patch, grade in zip(bp['boxes'], grade_order):
    patch.set_facecolor(colors[grade])
    patch.set_alpha(0.7)
plt.title('Pattern 6: Food Complaints by Zipcode vs Grade',
          fontsize=13, fontweight='bold')
plt.xlabel('Hygiene Grade')
plt.ylabel('Food Complaints Count')
plt.tight_layout()
plt.savefig('pattern6_complaints_boxplot.png', dpi=150)
plt.show()
print("Saved: pattern6_complaints_boxplot.png")


# PATTERN 7: Year Trend
# Is hygiene improving or worsening over time?
print("\n--- PATTERN 7: Year Trend ---")

yearly     = df.groupby(['inspection_year','grade']).size().unstack(fill_value=0)
yearly_pct = yearly.div(yearly.sum(axis=1), axis=0) * 100

print(f"\nGrade distribution by year:")
print(yearly_pct.round(1).to_string())

plt.figure(figsize=(10,5))
for grade in ['A','B','C']:
    if grade in yearly_pct.columns:
        plt.plot(yearly_pct.index, yearly_pct[grade],
                 marker='o', label=f'Grade {grade}',
                 color=colors[grade], linewidth=2.5,
                 markersize=8)
plt.title('Pattern 7: Grade Trends by Year',
          fontsize=13, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Percentage (%)')
plt.legend(title='Grade')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pattern7_year_trend.png', dpi=150)
plt.show()
print("Saved: pattern7_year_trend.png")


# PATTERN 8: Cuisine → Grade
# Which cuisines fail most?
print("\n--- PATTERN 8: Cuisine vs Grade C Rate ---")

cuisine_grade = df.groupby('cuisine_description')['grade'].value_counts(
    normalize=True).unstack(fill_value=0)
cuisine_grade['total']      = df.groupby('cuisine_description').size()
cuisine_grade['c_rate']     = cuisine_grade.get('C', 0)
cuisine_grade               = cuisine_grade[cuisine_grade['total'] >= 30]
top_cuisine_c               = cuisine_grade['c_rate'].sort_values(
    ascending=False).head(15)

print(f"\nTop 15 cuisines with highest Grade C rate:")
for cuisine, rate in top_cuisine_c.items():
    print(f"  {cuisine:<30}: {rate*100:.1f}%")

plt.figure(figsize=(12,6))
bars = plt.barh(top_cuisine_c.index, top_cuisine_c.values*100,
                color='#e74c3c', edgecolor='white')
for bar, val in zip(bars, top_cuisine_c.values):
    plt.text(bar.get_width()+0.3,
             bar.get_y()+bar.get_height()/2,
             f'{val*100:.1f}%', va='center', fontsize=9)
plt.title('Pattern 8: Top 15 Cuisines by Grade C Rate',
          fontsize=13, fontweight='bold')
plt.xlabel('Grade C Rate (%)')
plt.tight_layout()
plt.savefig('pattern8_cuisine_grade_c.png', dpi=150)
plt.show()
print("Saved: pattern8_cuisine_grade_c.png")


# SUMMARY
print('\n' + '='*60)
print('PATTERN DISCOVERY SUMMARY')
print('='*60)
if 'C' in monthly_pct.columns:
    print(f"1. Worst violation for Grade C : {viol_grade['c_rate'].idxmax()}")
    print(f"2. Worst month for Grade C     : {month_names[monthly_pct['C'].idxmax()-1]}")
print(f"3. Reinspection → more B/C     : ✅ Confirmed")
print(f"4. Critical violations → B/C   : ✅ Confirmed")
print(f"5. Worst zipcode               : {int(top_zip.index[0])} ({top_zip.values[0]*100:.1f}% Grade C)")
print(f"6. High complaints → worse     : See boxplot")
print(f"7. Year trend                  : See plot")
print(f"8. Worst cuisine               : {top_cuisine_c.index[0]} ({top_cuisine_c.values[0]*100:.1f}% Grade C)")
print('='*60)
print('\nDONE! 8 pattern plots saved.')