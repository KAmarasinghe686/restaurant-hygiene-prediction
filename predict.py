import joblib
import pandas as pd

# ══════════════════════════════════════════
# LOAD SAVED MODEL
# ══════════════════════════════════════════
print("Loading saved model...")
model        = joblib.load('best_model.pkl')
feature_cols = joblib.load('feature_cols.pkl')

grade_map = {0:'A', 1:'B', 2:'C'}
labels    = {
    'A': 'Excellent — passed inspection',
    'B': 'Needs improvement — some violations',
    'C': 'Failed — serious violations',
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
# ENTER YOUR OWN RESTAURANT DETAILS HERE
# ══════════════════════════════════════════

# ── Change these values to test ──
my_restaurant = {
    # Violation code (0-58)
    # Common codes:
    # 40 = violation 10F (minor, vermin proofing)
    # 5  = violation 02G (critical, food handling)
    # 10 = violation 08A (facility not maintained)
    'violation_code_encoded'  : 34,

    # How many food complaints in that zipcode (1-631)
    'food_complaints_zipcode' : 501,

    # Month of inspection (1-12)
    'inspection_month'        :8,

    # Year of inspection
    'inspection_year'         : 2025,

    # Is this a reinspection? (1=Yes, 0=No)
    'is_reinspection'         : 1,

    # Is this a reopening after closure? (1=Yes, 0=No)
    'is_reopening'            : 0,

    # Is this an initial inspection? (1=Yes, 0=No)
    'is_initial'              : 0,

    # Was violation critical? (1=Yes, 0=No)
    'is_critical'             : 1,

    # How many violations found in this inspection
    'violation_count'         : 14,

    # Is violation severe? codes starting 0 = severe
    # (1=Yes, 0=No)
    'is_severe_violation'     : 1,
}

# ══════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════
grade, conf, pa, pb, pc = predict_grade(my_restaurant)

print("\n" + "="*55)
print("MY RESTAURANT PREDICTION")
print("="*55)
print(f"\nInputs:")
for key, val in my_restaurant.items():
    print(f"  {key:<35} = {val}")

print(f"\nResult:")
print(f"  Predicted Grade : {grade}")
print(f"  Meaning         : {labels[grade]}")
print(f"  Confidence      : {conf}%")
print(f"\nProbability breakdown:")
print(f"  Grade A : {pa}%")
print(f"  Grade B : {pb}%")
print(f"  Grade C : {pc}%")
print("="*55)