# Restaurant Hygiene Grade Prediction Using NYC Inspection Data

**Course:** CMIS 3253 — Data Mining  
**Domain:** Public Health Analytics  
**Location:** Queens, New York City  
**Period:** 2023–2025  

---

## 1. Project Overview

This project predicts restaurant hygiene grades (A, B, C) using NYC restaurant inspection records collected from two public APIs. The target variable is the hygiene grade assigned by NYC health inspectors during restaurant visits.

---

## 2. Data Sources

| Source | API | Data Collected |
|--------|-----|----------------|
| NYC Open Data | data.cityofnewyork.us/resource/43nn-pn8j.json | Inspection records, violation codes, critical flags, cuisine types |
| NYC 311 | data.cityofnewyork.us/resource/erm2-nwe9.json | Food complaint counts per zipcode |

---

## 3. Dataset

| Factor | Value |
|--------|-------|
| Raw rows | 50,000 |
| Clean rows | 22,654 |
| Features | 16 columns |
| Null values | 0 |
| Date range | May 2023 — Dec 2025 |

### Grade Distribution
| Grade | Count | Percentage |
|-------|-------|-----------|
| A | 15,468 | 68.3% |
| B | 3,931 | 17.4% |
| C | 3,255 | 14.4% |

---

## 4. Preprocessing

### Columns Dropped
| Column | Reason |
|--------|--------|
| score | Leakage — directly determines grade |
| action | Leakage — reveals grade |
| camis | Just an ID number |
| dba | Restaurant name — no predictive value |
| boro | Single value (Queens only) |
| pct_poverty | 100% null |
| median_income | 100% null |
| latitude | Redundant with zipcode |
| longitude | Redundant with zipcode |
| violation_description | Redundant with violation_code |

### Features Created
| Feature | From | Reason |
|---------|------|--------|
| inspection_month | inspection_date | Seasonal patterns |
| inspection_year | inspection_date | Year trend |
| is_reinspection | inspection_type | Strong predictor (0.57) |
| is_reopening | inspection_type | Always Grade C |
| is_initial | inspection_type | Always Grade A |
| is_critical | critical_flag | Violation severity |
| violation_count | grouped rows | How many violations found |
| is_severe_violation | violation_code | Codes starting with 0 = severe |

---

## 5. Features Used In Model

| # | Feature | Correlation | Importance |
|---|---------|-------------|-----------|
| 1 | is_initial | -0.58 | 0.3538 |
| 2 | violation_count | — | 0.2928 |
| 3 | food_complaints_zipcode | +0.12 | 0.1110 |
| 4 | is_reinspection | +0.57 | 0.0887 |
| 5 | inspection_month | +0.02 | 0.0626 |
| 6 | violation_code_encoded | -0.20 | 0.0538 |
| 7 | inspection_year | +0.05 | 0.0216 |
| 8 | is_reopening | +0.10 | 0.0152 |
| 9 | is_severe_violation | — | 0.0003 |
| 10 | is_critical | +0.17 | 0.0002 |

---

## 6. EDA Key Findings

- Grade A dominates at 68.3% — class imbalance exists
- Most common cuisine: Chinese
- Most common violation: 10F (vermin proofing)
- 50.6% of inspections had critical violations
- Initial inspection → 100% Grade A
- Reopening inspection → 100% Grade C
- Strongest correlations: is_initial (-0.58), is_reinspection (+0.57)
- Multicollinearity: violation_code ↔ is_critical = -0.81

---

## 7. Pattern Discovery Key Findings

| Pattern | Finding |
|---------|---------|
| Seasonal | August worst month (21.4% Grade C), June best (9.2%) |
| Inspection Type | Initial=100% A, Reopening=100% C |
| Zipcode | 11040 worst area (34.4% Grade C) |
| Critical Flag | Critical violations → 60% still Grade A |
| Year Trend | Grade A declining: 74% (2023) → 66% (2025) |
| Cuisine | Pakistani (40%), Bangladeshi (39%) highest Grade C |
| Violations | A=4.93 avg, B=7.59 avg, C=10.11 avg violations |

---

## 8. Classification Results

### Baseline (8 features, no class weights)
| Model | Accuracy | F1 | Error |
|-------|----------|----|-------|
| Logistic Regression | 63.45% | 0.679 | 36.55% |
| Decision Tree | 64.62% | 0.687 | 35.38% |
| Random Forest | 70.62% | 0.710 | 29.38% |
| Gradient Boosting | 74.69% | 0.738 | 25.31% |
| XGBoost | 74.51% | 0.735 | 25.49% |

### Improved (10 features + class weights)
| Model | Accuracy | F1 | Error | CV |
|-------|----------|----|-------|-----|
| Logistic Regression | 70.07% | 0.733 | 29.93% | 69.17% |
| Decision Tree | 75.17% | 0.773 | 24.83% | 75.53% |
| Random Forest | 80.80% | 0.807 | 19.20% | 80.05% |
| **Gradient Boosting** | **82.28%** | **0.834** | **17.72%** | **82.92%** |
| XGBoost | 81.62% | 0.829 | 18.38% | 82.14% |

### Per-Class Recall Improvement
| Grade | Before | After | Change |
|-------|--------|-------|--------|
| A | 89% | 83% | -6% |
| B | 42% | 79% | +37% |
| C | 45% | 83% | +38% |

### Class Weights Applied
| Grade | Weight |
|-------|--------|
| A | 0.49 |
| B | 1.92 |
| C | 2.32 |

---

## 9. Association Rules Results

| Grade | Best Rule | Confidence | Lift |
|-------|-----------|------------|------|
| A | initial + violation_02G → A | 100% | 3.27 |
| B | violation_08A + reinspection → B | 38% | 5.63 |
| C | reinspection + critical → C | 31% | 2.15 |

**Total meaningful rules found: 141**

---

## 10. Clustering Results

| Cluster | Size | Composition | Meaning |
|---------|------|-------------|---------|
| 0 | 6,021 (26.6%) | 100% Grade A | Initial inspections |
| 1 | 9,015 (39.8%) | 36%A 35%B 29%C | Reinspections (mixed) |
| 2 | 7,618 (33.6%) | 81%A 11%B 8%C | Non-critical violations |

- Algorithm: K-Means, k=3
- Silhouette score: 0.24 (weak — confirms supervised learning needed)
- PCA variance explained: 47.2%

---

## 11. Project Flow

```
EDA → identified inspection type as key factor
  ↓
Pattern Discovery → confirmed seasonal/geographic patterns
  ↓
Correlation → identified is_initial (0.58) as strongest feature
  ↓
Association Rules → violation_08A + reinspection → Grade B (lift=5.63)
  ↓
Classification → 82.28% accuracy, is_initial = 49% importance
  ↓
Clustering → Cluster 0 = 100% Grade A (validates classification)
```

---

## 12. Limitations

- Queens data only (one of five NYC boroughs)
- Only 2 data sources
- Features collected during inspection (not pre-inspection)
- No hyperparameter tuning applied
- Weak clustering silhouette (0.24)
- is_critical near-zero importance despite inclusion

---

## 13. Key Conclusions

1. **Inspection type is the strongest predictor** — initial inspections always pass, reopening always fail
2. **violation_count is the most useful engineered feature** — average violations: A=4.93, B=7.59, C=10.11
3. **Summer increases failures** — August Grade C rate 21.4% vs June 9.2%
4. **Geographic variation exists** — zipcode 11040 has 34.4% Grade C rate
5. **Grade A is declining** — 74% (2023) to 66% (2025)
6. **Supervised learning outperforms clustering** — silhouette=0.24 confirms classification is needed
7. **Grade B is hardest to predict** — sits between A and C in feature space

---

## 14. Tools Used

```
Python, pandas, numpy, scikit-learn,
matplotlib, seaborn, mlxtend, xgboost, tkinter
```
