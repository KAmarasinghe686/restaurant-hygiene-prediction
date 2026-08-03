import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════
df = pd.read_csv('restaurant_hygiene_clean.csv')
df['inspection_date'] = pd.to_datetime(df['inspection_date'])
print("=" * 60)
print("K-MEANS CLUSTERING")
print("=" * 60)
print(f"Dataset shape : {df.shape}")

# ══════════════════════════════════════════
# ADD violation_count + is_severe_violation
# Same features used in classification
# ══════════════════════════════════════════
df['violation_count'] = df.groupby(
    ['inspection_date','zipcode','is_reinspection']
)['violation_code_encoded'].transform('count')

df['is_severe_violation'] = df['violation_code'].str.startswith('0').astype(int)

# ══════════════════════════════════════════
# PREPARE FEATURES — NO GRADE LABEL
# Unsupervised = no target variable
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

print(f"\nFeatures used : {len(feature_cols)}")
print(f"No grade label used — unsupervised ✅")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ══════════════════════════════════════════
# FIND OPTIMAL K — Elbow Method
# ══════════════════════════════════════════
print("\nFinding optimal K (elbow method)...")

inertias    = []
silhouettes = []
k_range     = range(2, 8)

for k in k_range:
    km  = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, lbl, sample_size=5000))
    print(f"  k={k}: inertia={km.inertia_:.0f}, silhouette={silhouettes[-1]:.4f}")

# ── PLOT 1: Elbow + Silhouette ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(k_range, inertias, marker='o',
             color='#3498db', linewidth=2.5, markersize=8)
axes[0].axvline(x=3, color='red', linestyle='--',
                alpha=0.7, label='k=3 (chosen)')
axes[0].set_title('Elbow Method — Optimal K',
                   fontsize=13, fontweight='bold')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(k_range, silhouettes, marker='o',
             color='#2ecc71', linewidth=2.5, markersize=8)
axes[1].axvline(x=3, color='red', linestyle='--',
                alpha=0.7, label='k=3 (chosen)')
axes[1].set_title('Silhouette Score — Optimal K',
                   fontsize=13, fontweight='bold')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cluster_plot1_elbow_silhouette.png', dpi=150)
plt.show()
print("Saved: cluster_plot1_elbow_silhouette.png")

# ══════════════════════════════════════════
# RUN K-MEANS WITH K=3
# k=3 because we have 3 grades (A, B, C)
# ══════════════════════════════════════════
print("\nRunning K-Means with k=3...")

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

print(f"\nCluster distribution:")
for c, cnt in df['cluster'].value_counts().sort_index().items():
    print(f"  Cluster {c}: {cnt:,} ({cnt/len(df)*100:.1f}%)")

# ══════════════════════════════════════════
# COMPARE CLUSTERS TO ACTUAL GRADES
# ══════════════════════════════════════════
print("\nComparing clusters to actual grades...")

cross_tab = pd.crosstab(df['cluster'], df['grade'],
                         normalize='index') * 100
print("\nGrade distribution per cluster (%):")
print(cross_tab.round(1).to_string())

# Name clusters based on dominant grade
cluster_names = {}
for c in range(3):
    dominant_grade = cross_tab.loc[c].idxmax()
    cluster_names[c] = f"Cluster {c} → mostly Grade {dominant_grade}"
    print(f"  Cluster {c}: dominant grade = {dominant_grade}")

# ── PLOT 2: Cluster vs Grade Heatmap ──
plt.figure(figsize=(8, 5))
sns.heatmap(cross_tab, annot=True, fmt='.1f',
            cmap='YlOrRd', linewidths=0.5,
            annot_kws={'size': 11})
plt.title('Cluster vs Actual Grade Distribution (%)',
          fontsize=13, fontweight='bold')
plt.xlabel('Actual Grade')
plt.ylabel('Cluster')
plt.tight_layout()
plt.savefig('cluster_plot2_cluster_grade_heatmap.png', dpi=150)
plt.show()
print("Saved: cluster_plot2_cluster_grade_heatmap.png")

# ── PLOT 3: Cluster vs Grade Bar Chart ──
fig, ax = plt.subplots(figsize=(10, 5))
colors_grade = {'A': '#2ecc71', 'B': '#f39c12', 'C': '#e74c3c'}
cross_tab.plot(kind='bar',
               color=[colors_grade[g] for g in cross_tab.columns],
               edgecolor='white', ax=ax)
ax.set_title('Grade Distribution Per Cluster',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Cluster')
ax.set_ylabel('Percentage (%)')
ax.set_xticklabels([f'Cluster {i}' for i in range(3)],
                    rotation=0)
ax.legend(title='Grade')
plt.tight_layout()
plt.savefig('cluster_plot3_cluster_grade_bar.png', dpi=150)
plt.show()
print("Saved: cluster_plot3_cluster_grade_bar.png")

# ══════════════════════════════════════════
# PCA — Visualize Clusters In 2D
# ══════════════════════════════════════════
print("\nRunning PCA for visualization...")

pca     = PCA(n_components=2, random_state=42)
X_pca   = pca.fit_transform(X_scaled)
var_exp = pca.explained_variance_ratio_

print(f"PCA variance explained:")
print(f"  PC1: {var_exp[0]*100:.1f}%")
print(f"  PC2: {var_exp[1]*100:.1f}%")
print(f"  Total: {sum(var_exp)*100:.1f}%")

# ── PLOT 4: PCA Scatter — Clusters ──
plt.figure(figsize=(10, 7))
cluster_colors = ['#3498db', '#e74c3c', '#2ecc71']
for c in range(3):
    mask = df['cluster'] == c
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=cluster_colors[c], alpha=0.4,
                s=10, label=f'Cluster {c}')
plt.title(f'K-Means Clusters — PCA Visualization\n'
          f'(PC1={var_exp[0]*100:.1f}%, PC2={var_exp[1]*100:.1f}% variance)',
          fontsize=13, fontweight='bold')
plt.xlabel(f'PC1 ({var_exp[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({var_exp[1]*100:.1f}% variance)')
plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig('cluster_plot4_pca_clusters.png', dpi=150)
plt.show()
print("Saved: cluster_plot4_pca_clusters.png")

# ── PLOT 5: PCA Scatter — Actual Grades ──
plt.figure(figsize=(10, 7))
grade_colors = {'A': '#2ecc71', 'B': '#f39c12', 'C': '#e74c3c'}
for grade in ['A', 'B', 'C']:
    mask = df['grade'] == grade
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=grade_colors[grade], alpha=0.4,
                s=10, label=f'Grade {grade}')
plt.title('Actual Grades — PCA Visualization\n'
          '(compare with cluster plot above)',
          fontsize=13, fontweight='bold')
plt.xlabel(f'PC1 ({var_exp[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({var_exp[1]*100:.1f}% variance)')
plt.legend(title='Grade')
plt.tight_layout()
plt.savefig('cluster_plot5_pca_grades.png', dpi=150)
plt.show()
print("Saved: cluster_plot5_pca_grades.png")

# ══════════════════════════════════════════
# CLUSTER FEATURE ANALYSIS
# What makes each cluster different?
# ══════════════════════════════════════════
print("\nCluster feature means:")
cluster_means = df.groupby('cluster')[feature_cols].mean().round(3)
print(cluster_means.to_string())

# ── PLOT 6: Feature Means Per Cluster ──
fig, ax = plt.subplots(figsize=(12, 6))
cluster_means_norm = (cluster_means - cluster_means.min()) / \
                     (cluster_means.max() - cluster_means.min())
cluster_means_norm.T.plot(kind='bar',
                           color=cluster_colors,
                           edgecolor='white', ax=ax)
ax.set_title('Normalised Feature Means Per Cluster',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Feature')
ax.set_ylabel('Normalised Value')
ax.set_xticklabels(feature_cols, rotation=30, ha='right')
ax.legend(title='Cluster',
          labels=[f'Cluster {i}' for i in range(3)])
plt.tight_layout()
plt.savefig('cluster_plot6_feature_means.png', dpi=150)
plt.show()
print("Saved: cluster_plot6_feature_means.png")

# ══════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════
sil = silhouette_score(X_scaled, df['cluster'], sample_size=5000)
print("\n" + "="*60)
print("CLUSTERING SUMMARY")
print("="*60)
print(f"Algorithm     : K-Means")
print(f"K chosen      : 3 (matches number of grades)")
print(f"Silhouette    : {sil:.4f}")
print(f"  (0=random, 1=perfect, >0.5=good)")
print()
print("Cluster composition:")
for c in range(3):
    row = cross_tab.loc[c]
    dominant = row.idxmax()
    print(f"  Cluster {c}: "
          f"A={row.get('A',0):.1f}% "
          f"B={row.get('B',0):.1f}% "
          f"C={row.get('C',0):.1f}% "
          f"← mostly Grade {dominant}")
print()
print("Cluster feature highlights:")
for c in range(3):
    row = cluster_means.loc[c]
    print(f"  Cluster {c}:")
    print(f"    violation_count : {row['violation_count']:.2f}")
    print(f"    is_reinspection : {row['is_reinspection']:.2f}")
    print(f"    is_initial      : {row['is_initial']:.2f}")
    print(f"    is_critical     : {row['is_critical']:.2f}")
print("="*60)
print("\nDONE! 6 clustering plots saved.")