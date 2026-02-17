import pandas as pd
import numpy as np

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# CONFIG
# =============================================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

NROWS = 5000
UNDERSAMPLE_RATIO = 0.5  # minority/majority after undersampling

data_path = r"data\census-bureau.data"
cols_path = r"data\census-bureau.columns"

# =============================================================================
# 1) LOAD RAW DATA
# =============================================================================
with open(cols_path, "r") as f:
    cols = [line.strip() for line in f if line.strip()]

df = pd.read_csv(
    data_path,
    header=None,
    names=cols,
    na_values="?",
    nrows=NROWS
)

# =============================================================================
# 2) MISSING HANDLING (AS YOU DID)
# =============================================================================
low_missing_features = [
    "hispanic origin",
    "state of previous residence",
    "country of birth self",
    "country of birth mother",
    "country of birth father"
]
for col in low_missing_features:
    if col in df.columns:
        df[col] = df[col].astype("object").fillna("Unknown")

migration_cols = [
    "migration code-change in msa",
    "migration code-change in reg",
    "migration code-move within reg",
    "migration prev res in sunbelt"
]
for c in migration_cols:
    if c in df.columns:
        df[c] = df[c].astype("object").fillna("Not Applicable")

# =============================================================================
# 3) ORDINAL ENCODING FOR EDUCATION
# =============================================================================
education_alias = {
    "Some college but no degree": "Some college",
    "Bachelors degree(BA AB BS)": "Bachelor's degree",
    "Masters degree(MA MS MEng MEd MSW MBA)": "Master's degree",
    "Associates degree-academic program": "Associate's (academic)",
    "Associates degree-occup /vocational": "Associate's (vocational)",
    "Prof school degree (MD DDS DVM LLB JD)": "Professional degree",
    "Doctorate degree(PhD EdD)": "Doctorate degree",
    "7th and 8th grade": "7th-8th grade",
    "5th or 6th grade": "5th-6th grade",
    "1st 2nd 3rd or 4th grade": "1st-4th grade",
}
df["education_clean"] = df["education"].replace(education_alias)

education_map = {
    "Children": 0,
    "Less than 1st grade": 1,
    "1st-4th grade": 2,
    "5th-6th grade": 3,
    "7th-8th grade": 4,
    "9th grade": 5,
    "10th grade": 6,
    "11th grade": 7,
    "12th grade no diploma": 8,
    "High school graduate": 9,
    "Some college": 10,
    "Associate's (vocational)": 11,
    "Associate's (academic)": 12,
    "Bachelor's degree": 13,
    "Master's degree": 14,
    "Professional degree": 15,
    "Doctorate degree": 16
}
df["education_ord"] = df["education_clean"].map(education_map)

unknown_edu = df.loc[df["education_ord"].isna(), "education"].unique()
if len(unknown_edu) > 0:
    print("Still unmapped education categories:", unknown_edu)

df = df.drop(columns=["education", "education_clean"])

# =============================================================================
# 4) SEPARATE TARGET + WEIGHT
# =============================================================================
y_raw = df["label"].copy()
df = df.drop(columns=["label"])

weights = df["weight"].copy()  # not used in models below, but kept
df = df.drop(columns=["weight"])

y = (y_raw == "50000+.").astype(int)

# =============================================================================
# 5) ENCODING (SEX BINARY, LOW/MED OHE, HIGH FREQ)
# =============================================================================
# 5.1 sex binary
sex_map = {"Female": 0, "Male": 1}
df["sex_encoded"] = df["sex"].map(sex_map).astype(int)
df = df.drop(columns=["sex"])

# 5.2 low-card one-hot (3-9)
low_card_features = [
    "enroll in edu inst last wk",
    "live in this house 1 year ago",
    "member of a labor union",
    "migration prev res in sunbelt",
    "fill inc questionnaire for veteran's admin",
    "own business or self employed",
    "veterans benefits",
    "citizenship",
    "race",
    "family members under 18",
    "reason for unemployment",
    "region of previous residence",
    "tax filer stat",
    "marital stat",
    "detailed household summary in household",
    "full or part time employment stat",
    "migration code-change in reg",
    "class of worker",
    "hispanic origin",
    "migration code-change in msa",
    "migration code-move within reg"
]
for col in ["own business or self employed", "veterans benefits"]:
    df[col] = df[col].astype(str)

ohe_low = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
X_low = ohe_low.fit_transform(df[low_card_features])
low_names = ohe_low.get_feature_names_out(low_card_features)
df_low = pd.DataFrame(X_low, columns=low_names, index=df.index)
df = df.drop(columns=low_card_features)

# 5.3 medium-card one-hot (10-20)
medium_card_features = ["major occupation code"]
ohe_med = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
X_med = ohe_med.fit_transform(df[medium_card_features])
med_names = ohe_med.get_feature_names_out(medium_card_features)
df_med = pd.DataFrame(X_med, columns=med_names, index=df.index)
df = df.drop(columns=medium_card_features)

# 5.4 high-card frequency (21+)
high_card_features = [
    "major industry code",
    "detailed household and family stat",
    "country of birth self"
    "country of birth father",
    "country of birth mother",
    "detailed occupation recode",
    "detailed industry recode",
    "state of previous residence"
]
for col in ["detailed occupation recode", "detailed industry recode"]:
    df[col] = df[col].astype(str)

df_freq = pd.DataFrame(index=df.index)
for col in high_card_features:
    freq_map = df[col].value_counts(normalize=True).to_dict()
    df_freq[col + "_freq"] = df[col].map(freq_map).fillna(0.0)

df = df.drop(columns=high_card_features)

# 5.5 numeric keep
numeric_features_keep = [
    "age",
    "wage per hour",
    "capital gains",
    "capital losses",
    "dividends from stocks",
    "num persons worked for employer",
    "weeks worked in year",
    "year",
    "education_ord",
    "sex_encoded"
]
df_num = df[numeric_features_keep].copy()

# 5.6 final matrix
X = pd.concat([df_num, df_low, df_med, df_freq], axis=1)

# final missing check
if X.isna().sum().sum() != 0:
    print("Missing exists after preprocessing:", X.isna().sum()[X.isna().sum() > 0])

print("=" * 80)
print("DATA READY")
print("=" * 80)
print("X shape:", X.shape)
print("y distribution:", y.value_counts().to_dict())

# =============================================================================
# 6) TRAIN / TEST SPLIT (TEST UNTOUCHED)
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
#rus = RandomUnderSampler(sampling_strategy=UNDERSAMPLE_RATIO, random_state=RANDOM_STATE)
rus = RandomOverSampler(sampling_strategy='auto', random_state=RANDOM_STATE)
results = {}

# =============================================================================
# 7) MODELS (PRECISION-OPTIMIZED)
# =============================================================================

# --- Logistic Regression (RUS inside CV) ---
lr_pipe = ImbPipeline(steps=[
    ("rus", rus),
    ("scaler", RobustScaler()),
    ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))
])

lr_grid = {
    "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
    "model__solver": ["liblinear"],
    "model__penalty": ["l2"],
    "model__class_weight": [None]
}

lr_search = GridSearchCV(lr_pipe, lr_grid, cv=cv, scoring="precision", n_jobs=-1, verbose=1)
lr_search.fit(X_train, y_train)
lr_best = lr_search.best_estimator_
yp = lr_best.predict(X_test)

results["LR_RUS"] = {
    "best_params": lr_search.best_params_,
    "accuracy": accuracy_score(y_test, yp),
    "precision": precision_score(y_test, yp, zero_division=0),
    "recall": recall_score(y_test, yp, zero_division=0),
    "cm": confusion_matrix(y_test, yp)
}

# --- Random Forest (RUS inside CV) ---
rf_pipe = ImbPipeline(steps=[
    ("rus", rus),
    ("model", RandomForestClassifier(random_state=RANDOM_STATE))
])

rf_grid = {
    "model__n_estimators": [100, 200, 500],
    "model__max_depth": [10, 20, 30, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2"],
    "model__class_weight": [None]
}

rf_search = GridSearchCV(rf_pipe, rf_grid, cv=cv, scoring="precision", n_jobs=-1, verbose=1)
rf_search.fit(X_train, y_train)
rf_best = rf_search.best_estimator_
yp = rf_best.predict(X_test)

results["RF_RUS"] = {
    "best_params": rf_search.best_params_,
    "accuracy": accuracy_score(y_test, yp),
    "precision": precision_score(y_test, yp, zero_division=0),
    "recall": recall_score(y_test, yp, zero_division=0),
    "cm": confusion_matrix(y_test, yp)
}

# --- SVM (RUS inside CV) ---
svm_pipe = ImbPipeline(steps=[
    ("rus", rus),
    ("scaler", RobustScaler()),
    ("model", SVC(probability=False, random_state=RANDOM_STATE))
])

svm_grid = {
    "model__C": [0.1, 1, 10, 100],
    "model__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
    "model__kernel": ["rbf"],
    "model__class_weight": [None]
}

svm_search = GridSearchCV(svm_pipe, svm_grid, cv=cv, scoring="precision", n_jobs=-1, verbose=1)
svm_search.fit(X_train, y_train)
svm_best = svm_search.best_estimator_
yp = svm_best.predict(X_test)

results["SVM_RUS"] = {
    "best_params": svm_search.best_params_,
    "accuracy": accuracy_score(y_test, yp),
    "precision": precision_score(y_test, yp, zero_division=0),
    "recall": recall_score(y_test, yp, zero_division=0),
    "cm": confusion_matrix(y_test, yp)
}

# =============================================================================
# 8) PRINT RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("TEST RESULTS (PRECISION-OPTIMIZED)")
print("=" * 80)

for k, v in results.items():
    print("\n", k)
    print("best_params:", v["best_params"])
    print(f"accuracy : {v['accuracy']:.4f}")
    print(f"precision: {v['precision']:.4f}")
    print(f"recall   : {v['recall']:.4f}")
    print("cm:\n", v["cm"])

# =============================================================================
# STEP 1: FEATURE SELECTION FOR SEGMENTATION
# =============================================================================
print("\n--- Step 1: Feature Selection for Segmentation ---")

# Select meaningful features for marketing segmentation
# Focus on: Demographics, Income, Employment, Household

segmentation_features = [
    # Demographics
    'age',
    'education_ord',
    'sex_encoded',

    # Income & Wealth
    'capital gains',
    'capital losses',
    'dividends from stocks',

    # Employment
    'weeks worked in year',
    'num persons worked for employer',
]

X_seg_base = X[segmentation_features].copy()

# Add aggregated categorical features
# Marital status (married vs not married)
marital_cols = [col for col in X.columns if col.startswith('marital stat_')]
married_cols = [col for col in marital_cols if 'Married' in col]
X_seg_base['is_married'] = X[married_cols].max(axis=1) if married_cols else 0

# Employment status (full-time vs other)
employment_cols = [col for col in X.columns if col.startswith('full or part time employment stat_')]
fulltime_cols = [col for col in employment_cols if 'Full-time' in col]
X_seg_base['is_employed_full_time'] = X[fulltime_cols].max(axis=1) if fulltime_cols else 0

# Derived features
X_seg_base['has_capital_income'] = ((X['capital gains'] > 0) |
                                    (X['dividends from stocks'] > 0)).astype(int)
X_seg_base['high_education'] = (X['education_ord'] >= 13).astype(int)  # Bachelor's+

print(f"Selected {X_seg_base.shape[1]} features for segmentation")
print(f"Features: {list(X_seg_base.columns)}")

# Now standardize only these selected features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_seg = scaler.fit_transform(X_seg_base)

print("Features standardized for clustering")

# =============================================================================
# STEP 2: DETERMINE OPTIMAL NUMBER OF CLUSTERS
# =============================================================================
print("\n--- Step 2: Determining Optimal Number of Clusters ---")

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

# Test different numbers of clusters (K)
K_range = range(2, 11)  # Test K from 2 to 10
inertias = []
silhouettes = []
davies_bouldins = []

print("Testing different values of K...")
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X_seg)

    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_seg, labels))
    davies_bouldins.append(davies_bouldin_score(X_seg, labels))

    print(
        f"K={k}: Inertia={kmeans.inertia_:.0f}, Silhouette={silhouettes[-1]:.3f}, Davies-Bouldin={davies_bouldins[-1]:.3f}")

# Find optimal K based on different metrics
optimal_k_silhouette = K_range[np.argmax(silhouettes)]
optimal_k_db = K_range[np.argmin(davies_bouldins)]

print(f"\nOptimal K based on Silhouette Score: {optimal_k_silhouette}")
print(f"Optimal K based on Davies-Bouldin Index: {optimal_k_db}")

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Elbow Method (Inertia)
axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[0].set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(K_range)

# Plot 2: Silhouette Score (higher is better)
axes[1].plot(K_range, silhouettes, 'go-', linewidth=2, markersize=8)
axes[1].axvline(x=optimal_k_silhouette, color='red', linestyle='--', label=f'Optimal K={optimal_k_silhouette}')
axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[1].set_ylabel('Silhouette Score (higher is better)', fontsize=12)
axes[1].set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(K_range)
axes[1].legend()

# Plot 3: Davies-Bouldin Index (lower is better)
axes[2].plot(K_range, davies_bouldins, 'ro-', linewidth=2, markersize=8)
axes[2].axvline(x=optimal_k_db, color='green', linestyle='--', label=f'Optimal K={optimal_k_db}')
axes[2].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[2].set_ylabel('Davies-Bouldin Index (lower is better)', fontsize=12)
axes[2].set_title('Davies-Bouldin Analysis', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3)
axes[2].set_xticks(K_range)
axes[2].legend()

plt.tight_layout()

plt.close()

# Choose final K (you can adjust this based on business needs)
# For marketing, 4-6 segments are typically manageable
FINAL_K = 5  # You can change this based on the plots

print(f"\nSelected K={FINAL_K} for final segmentation")
print("(Balancing statistical quality with business manageability)")
# =============================================================================
# STEP 3: CREATE FINAL SEGMENTS
# =============================================================================
print("\n--- Step 3: Creating Final Customer Segments ---")

# Apply K-Means with final K
FINAL_K = 4
kmeans_final = KMeans(n_clusters=FINAL_K, random_state=RANDOM_STATE, n_init=20)
clusters = kmeans_final.fit_predict(X_seg)

print(f"\nCreated {FINAL_K} customer segments")
print(f"\nSegment Distribution:")
for i in range(FINAL_K):
    count = (clusters == i).sum()
    pct = count / len(clusters) * 100
    print(f"  Segment {i}: {count:,} customers ({pct:.1f}%)")

# =============================================================================
# STEP 4: SEGMENT PROFILING (Map back to original features)
# =============================================================================
print("\n--- Step 4: Segment Profiling ---")

# Create profiling dataset with original features + segment labels
X_profiling = X_seg_base.copy()  # Use the un-scaled features for interpretation
X_profiling['Segment'] = clusters
X_profiling['Income_50K+'] = y.values

# Calculate profile statistics for each segment
segment_profiles = []

for seg in range(FINAL_K):
    seg_data = X_profiling[X_profiling['Segment'] == seg]

    profile = {
        'Segment': f'Segment {seg}',
        'Size': len(seg_data),
        'Size_Pct': len(seg_data) / len(X_profiling) * 100,
        'Avg_Age': seg_data['age'].mean(),
        'Avg_Education': seg_data['education_ord'].mean(),
        'Pct_Male': seg_data['sex_encoded'].mean() * 100,
        'Pct_Married': seg_data['is_married'].mean() * 100,
        'Pct_FullTime': seg_data['is_employed_full_time'].mean() * 100,
        'Pct_HighEd': seg_data['high_education'].mean() * 100,
        'Avg_CapitalGains': seg_data['capital gains'].mean(),
        'Avg_Dividends': seg_data['dividends from stocks'].mean(),
        'Pct_HasCapital': seg_data['has_capital_income'].mean() * 100,
        'Avg_WeeksWorked': seg_data['weeks worked in year'].mean(),
        'Avg_NumEmployers': seg_data['num persons worked for employer'].mean(),
        'Pct_Income50K': seg_data['Income_50K+'].mean() * 100
    }

    segment_profiles.append(profile)

# Create DataFrame
df_profiles = pd.DataFrame(segment_profiles)

print("\n" + "=" * 100)
print("SEGMENT PROFILES")
print("=" * 100)
print(df_profiles.round(1).to_string(index=False))

# Save profiles


# =============================================================================
# STEP 5: SEGMENT CHARACTERIZATION (Give descriptive names)
# =============================================================================
print("\n--- Step 5: Segment Characterization ---")
print("\n" + "=" * 100)

segment_names = []
segment_descriptions = []

for i in range(FINAL_K):
    profile = segment_profiles[i]

    # Determine age group
    age = profile['Avg_Age']
    if age < 25:
        age_group = "Young"
    elif age < 45:
        age_group = "Mid-Career"
    elif age < 65:
        age_group = "Established"
    else:
        age_group = "Senior"

    # Determine income/wealth level
    high_income_pct = profile['Pct_Income50K']
    capital_pct = profile['Pct_HasCapital']

    if high_income_pct > 15 and capital_pct > 15:
        wealth = "Affluent"
    elif high_income_pct > 10 or capital_pct > 5:
        wealth = "Moderate-Income"
    else:
        wealth = "Mass-Market"

    # Determine education level
    if profile['Pct_HighEd'] > 30:
        education = "Highly-Educated"
    elif profile['Pct_HighEd'] > 15:
        education = "Educated"
    else:
        education = "General-Education"

    # Combine characteristics
    name = f"{age_group} {wealth} {education}"
    segment_names.append(name)

    # Create detailed description
    description = f"""
SEGMENT {i}: {name}
{'=' * 100}
Size: {profile['Size']:,} customers ({profile['Size_Pct']:.1f}% of total)

Demographics:
  - Average Age: {profile['Avg_Age']:.0f} years
  - Gender: {profile['Pct_Male']:.0f}% Male, {100 - profile['Pct_Male']:.0f}% Female
  - Marital Status: {profile['Pct_Married']:.0f}% Married
  - Education: {profile['Pct_HighEd']:.0f}% have Bachelor's degree or higher

Economic Characteristics:
  - Income: {profile['Pct_Income50K']:.1f}% earn over $50,000 annually
  - Capital Income: {profile['Pct_HasCapital']:.0f}% have investment income
  - Average Capital Gains: ${profile['Avg_CapitalGains']:,.0f}
  - Average Dividends: ${profile['Avg_Dividends']:,.0f}

Employment:
  - Full-time Employment: {profile['Pct_FullTime']:.0f}%
  - Average Weeks Worked: {profile['Avg_WeeksWorked']:.1f} weeks/year
  - Average Employers: {profile['Avg_NumEmployers']:.1f}
"""

    segment_descriptions.append(description)
    print(description)


