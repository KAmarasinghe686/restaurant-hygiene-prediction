import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('restaurant_hygiene_raw.csv')

print('RESTAURANT HYGIENE — FINAL PREPROCESSING')

print(f'Original shape : {df.shape}')

# STEP 1: Drop useless columns
drop_cols = [
    'camis', 'dba', 'boro', 'score', 'action',
    'violation_description', 'pct_poverty', 'median_income',
    'latitude', 'longitude'
]
drop_cols = [c for c in drop_cols if c in df.columns]
df = df.drop(columns=drop_cols)
print(f'\nSTEP 1 — Dropped: {drop_cols}')
print(f'Shape after    : {df.shape}')

# STEP 2: Filter grades A, B, C only
before = len(df)
df = df[df['grade'].isin(['A','B','C'])].reset_index(drop=True)
print(f'\nSTEP 2 — Removed {before - len(df)} rows (N,Z,P,NaN)')
print(f'Shape after    : {df.shape}')
print(f'Grade counts   :\n{df["grade"].value_counts()}')

# STEP 3: Drop null rows
before = len(df)
df = df.dropna(subset=['zipcode', 'inspection_date'])
print(f'\nSTEP 3 — Dropped {before - len(df)} null rows')
print(f'Shape after    : {df.shape}')

# STEP 4: Fill remaining nulls
df['violation_code'] = df['violation_code'].fillna('No Violation')
median_c = df['food_complaints_zipcode'].median()
df['food_complaints_zipcode'] = df['food_complaints_zipcode'].fillna(median_c)
print(f'\nSTEP 4 — Nulls filled')
print(f'Nulls remaining:\n{df.isnull().sum()}')

# STEP 5: Convert types
df['inspection_date'] = pd.to_datetime(df['inspection_date'], errors='coerce')
df['zipcode']         = pd.to_numeric(df['zipcode'], errors='coerce').astype(int)
print(f'\nSTEP 5 — Types converted')

# STEP 6: Extract date features
df['inspection_month'] = df['inspection_date'].dt.month
df['inspection_year']  = df['inspection_date'].dt.year
print(f'\nSTEP 6 — Date features extracted')

# STEP 7: Binary flags from inspection_type
# Keep all 3 meaningful types
# is_initial kept because it has 0.58 correlation with grade
df['is_reinspection'] = df['inspection_type'].str.contains(
    'Re-inspection', case=False, na=False).astype(int)
df['is_reopening']    = df['inspection_type'].str.contains(
    'Reopening', case=False, na=False).astype(int)
df['is_initial']      = df['inspection_type'].str.contains(
    'Initial', case=False, na=False).astype(int)

print(f'\nSTEP 7 — Inspection type binary flags:')
print(f'  is_reinspection : {df["is_reinspection"].sum():,} ({df["is_reinspection"].mean()*100:.1f}%)')
print(f'  is_reopening    : {df["is_reopening"].sum():,} ({df["is_reopening"].mean()*100:.1f}%)')
print(f'  is_initial      : {df["is_initial"].sum():,} ({df["is_initial"].mean()*100:.1f}%)')

# STEP 8: Binary flag from critical_flag
df['is_critical'] = (df['critical_flag'] == 'Critical').astype(int)
print(f'\nSTEP 8 — Critical binary:')
print(f'  is_critical : {df["is_critical"].sum():,} ({df["is_critical"].mean()*100:.1f}%)')

# STEP 9: Encode remaining columns
le = LabelEncoder()
df['violation_code_encoded'] = le.fit_transform(df['violation_code'].astype(str))
grade_le = LabelEncoder()
df['grade_encoded'] = grade_le.fit_transform(df['grade'])
print(f'\nSTEP 9 — Encoded:')
print(f'  violation_code : {df["violation_code"].nunique()} unique values')
print(f'  grade          : A={grade_le.transform(["A"])[0]} B={grade_le.transform(["B"])[0]} C={grade_le.transform(["C"])[0]}')

# STEP 10: Select final columns
final_cols = [
    # Raw text — for reference and EDA
    'cuisine_description',
    'inspection_type',
    'violation_code',
    'critical_flag',
    'inspection_date',
    'zipcode',
    'food_complaints_zipcode',

    # Encoded — for model
    'violation_code_encoded',

    # Date features — for model
    'inspection_month',
    'inspection_year',

    # Binary flags — for model
    'is_reinspection',
    'is_reopening',
    'is_initial',
    'is_critical',

    # Target
    'grade',
    'grade_encoded',
]

final_cols = [c for c in final_cols if c in df.columns]
df_final   = df[final_cols].copy().reset_index(drop=True)
df_final.to_csv('restaurant_hygiene_clean.csv', index=False)

print(f'\nSTEP 10 — Saved: restaurant_hygiene_clean.csv')
print(f'\nFinal rows     : {len(df_final)}')
print(f'Final columns  : {len(df_final.columns)}')
print(f'\nAll columns:')
for i, col in enumerate(df_final.columns, 1):
    tag = '[RAW]'     if col in ['cuisine_description','inspection_type',
                                  'violation_code','critical_flag',
                                  'inspection_date','zipcode',
                                  'food_complaints_zipcode'] else \
          '[TARGET]'  if 'grade' in col else '[FEATURE]'
    print(f'  {i:2}. {col:<30} {tag}')
print(f'\nGrade distribution:')
total = len(df_final)
for grade, count in df_final['grade'].value_counts().items():
    print(f'  {grade}: {count:,} ({count/total*100:.1f}%)')
print(f'\nNull values:')
nulls = df_final.isnull().sum()
print(nulls[nulls > 0] if nulls.sum() > 0 else '  None! ✅')

# Show correlations with grade
print(f'\nCorrelations with grade:')
numeric = df_final[['violation_code_encoded','food_complaints_zipcode',
                     'inspection_month','inspection_year',
                     'is_reinspection','is_reopening',
                     'is_initial','is_critical','grade_encoded']]
corr = numeric.corr()['grade_encoded'].drop('grade_encoded').abs().sort_values(ascending=False)
for col, val in corr.items():
    bar = '█' * int(val * 30)
    print(f'  {col:<30} {val:.4f}  {bar}')

print('\n' + '='*60)
print('PREPROCESSING DONE!')
print(f'Raw: 50,000 rows → Clean: {len(df_final):,} rows')
print(f'Raw: 18 columns  → Clean: {len(df_final.columns)} columns')
print('='*60)