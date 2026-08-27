# Holistic Data Preparer: Customer Credit Risk Prediction

**Project Type:** Final Assessment | **Duration:** 8 Days (1 hour per day) | **Difficulty:** Advanced

**Status:** Production-Grade Data Engineering Project for ML Modeling

---

## 📋 Project Overview

This is a **comprehensive end-to-end data preprocessing and feature engineering project** for a fintech company building a **credit risk prediction model**. You'll work with a multi-source customer credit risk dataset and transform it into a production-ready dataset optimized for default prediction.

As a Junior Data Scientist, you'll perform the complete ML pipeline's data preparation phase—the foundation that determines downstream model performance.

**Key Insight:** Poor preprocessing ≈ Poor models. This project teaches you why 80% of ML work is data engineering.

---

## 🎯 Project Objective

Build a **fully cleaned, engineered, and scaled dataset** that:

- ✅ Handles missing values using domain-appropriate strategies
- ✅ Detects and treats outliers without data loss
- ✅ Encodes categorical variables intelligently
- ✅ Engineers domain-specific financial features
- ✅ Scales numerical features for ML readiness
- ✅ Produces reproducible, documented transformations

**Primary Deliverable:** `final_cleaned_credit_risk_dataset.csv` (ML-ready)

---

## 📊 Dataset Overview

### Input Data Sources (Multi-Source Integration)

| Source | Format | Content | Volume |
|--------|--------|---------|--------|
| **Transactions Dataset** | CSV | Main customer demographics & financial records | ~50K rows |
| **Customer Metadata** | JSON | Extended customer attributes & behavioral data | ~50K records |
| **Loan Repayment History** | SQL | Historical repayment patterns & missed payments | ~50K rows |
| **External Economic Indicators** | REST API | Macro economic data by region | Dynamic |

### Dataset Features: 15 Dimensions + 1 Target

#### **Demographic Features**
```
customer_id (String/Int)      → Unique customer identifier [No nulls]
age (Integer)                 → Years [INJECTED MISSING VALUES for imputation tasks]
gender (Categorical)          → {Male, Female, Other} [Missing + imbalance]
region (Categorical)          → {North, South, East, West} [One-Hot candidate]
education_level (Ordinal)     → {Primary, Secondary, Graduate, Post-Grad} [Preserve order]
employment_type (Categorical) → {Salaried, Self-Employed, Unemployed} [Categorical nulls]
```

#### **Financial Features**
```
annual_income (Float)         → Annual income in ₹ [Outliers + missing + skewed]
loan_amount (Float)           → Requested loan in ₹ [Outliers + skewed]
loan_purpose (Categorical)    → {Home, Car, Education, Business, Other}
credit_score (Float)          → CIBIL score [300-850 range] [Outliers + nulls]
repayment_history (Integer)   → Missed payments (12 months) [For binning]
```

#### **Behavioral Features**
```
transaction_count (Integer)   → Transactions in last 6 months [For K-Means binning]
spending_ratio (Float)        → Spending-to-Income ratio (%) [Skewed: test transformations]
join_date (Date)              → Bank membership date [Extract temporal features]
```

#### **Target Variable**
```
default_flag (Binary Int)     → 0 = No Default, 1 = Default [ML Target]
```

---

## 🛠️ Technology Stack

```
Python 3.8+
pandas                        - Data manipulation & ETL
numpy                         - Numerical operations
scikit-learn                  - ML pipelines & preprocessing
scipy                         - Statistical methods (Box-Cox, Yeo-Johnson)
matplotlib / seaborn          - EDA & visualization
pandas-profiling              - Automated data profiling (bonus)
```

---

## 📅 Project Timeline & Phases

### **Day 1: Conceptual Foundation**

**Goal:** Build domain knowledge before touching code

**Tasks:**

1. **Write Short Notes On:**
   - What is Data Analysis? (Types, applications, lifecycle)
   - How to Plan a Data Science Project? (Scoping, timeline, resources)
   - How to Frame a Machine Learning Problem? (Objective, metrics, constraints)

2. **Explain Tensors with NumPy Examples**
   - 0D (Scalars): `np.array(5)` → single value
   - 1D (Vectors): `np.array([1,2,3])` → 1D array
   - 2D (Matrices): `np.array([[1,2],[3,4]])` → 2D array
   - 3D+ (Tensors): Multi-dimensional arrays for images, sequences, etc.
   - Matrix operations: transpose, multiplication, reshaping

**Why This Matters:**
> Understanding tensor algebra is fundamental to working with ML frameworks (TensorFlow, PyTorch). Data itself is a tensor; transformations are tensor operations.

---

### **Day 2: Data Acquisition & Understanding**

**Goal:** Load data from multiple sources and profile it

**Part B: Data Acquisition**

3. **Import Datasets from Multiple Sources:**

   ```python
   import pandas as pd
   import json
   import sqlite3
   import requests
   
   # Load CSV (Main transactions)
   df_transactions = pd.read_csv('credit_risk_main.csv')
   
   # Parse JSON (Customer metadata)
   with open('customer_metadata.json', 'r') as f:
       customer_meta = json.load(f)
   df_meta = pd.json_normalize(customer_meta)
   
   # Fetch from SQL (Repayment history)
   conn = sqlite3.connect('loan_history.db')
   df_repayment = pd.read_sql_query(
       "SELECT * FROM repayment_history", conn
   )
   
   # Call REST API (Economic indicators)
   response = requests.get('https://api.economic-data.com/indicators')
   df_economic = pd.DataFrame(response.json())
   ```

4. **Merge Datasets on Key Identifiers**
   - Left join on `customer_id`
   - Handle mismatches and duplicates
   - Verify referential integrity

**Part C: Data Understanding & Cleaning**

4. **Exploratory Data Analysis**

   ```python
   # Data shape & types
   df.info()                    # Data types, null counts
   df.describe()                # Statistical summary
   df.shape                     # Dimensions
   
   # Null value analysis
   df.isnull().sum()            # Missing counts per column
   df.isnull().sum() / len(df) * 100  # Missing percentages
   
   # Duplicate detection
   df.duplicated().sum()        # Duplicate row count
   df.duplicated(subset=['customer_id']).sum()  # By specific column
   ```

5. **Pandas Profiling (Auto-generated Report)**
   ```python
   from pandas_profiling import ProfileReport
   
   profile = ProfileReport(df, title="Credit Risk Data Profile")
   profile.to_file("data_profile.html")
   ```
   - Generates detailed HTML report with:
     - Variable types & correlations
     - Missing value patterns
     - Sample values & frequency distributions
     - Interaction matrix

---

### **Day 3-4: Missing Value Imputation**

**Goal:** Handle missing data using 6 different strategies

**Part C: Missing Value Strategies**

6. **Strategy-Based Imputation:**

   ```python
   from sklearn.impute import SimpleImputer, KNNImputer
   from sklearn.experimental import enable_iterative_imputer
   from sklearn.impute import IterativeImputer
   import numpy as np
   
   # Strategy 1: Simple Imputer (Numeric - Mean/Median)
   imputer_mean = SimpleImputer(strategy='mean')
   df['age'] = imputer_mean.fit_transform(df[['age']])
   
   # Strategy 2: Simple Imputer (Categorical - Most Frequent)
   imputer_freq = SimpleImputer(strategy='most_frequent')
   df['gender'] = imputer_freq.fit_transform(df[['gender']])
   
   # Strategy 3: Most Frequent Category Imputation (explicit)
   df['employment_type'].fillna(df['employment_type'].mode()[0], inplace=True)
   
   # Strategy 4: Missing Indicator + Random Sample
   # Add binary indicator for missing + fill with random values from observed
   df['annual_income_missing_flag'] = df['annual_income'].isnull().astype(int)
   observed_incomes = df['annual_income'].dropna()
   missing_mask = df['annual_income'].isnull()
   df.loc[missing_mask, 'annual_income'] = \
       np.random.choice(observed_incomes, size=missing_mask.sum())
   
   # Strategy 5: KNN Imputer (Multivariate)
   # Uses K-nearest neighbors for imputation
   knn_imputer = KNNImputer(n_neighbors=5)
   numeric_cols = ['annual_income', 'loan_amount', 'credit_score']
   df[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])
   
   # Strategy 6: MICE (Multivariate Imputation by Chained Equations)
   mice_imputer = IterativeImputer(max_iter=10, random_state=42)
   df[numeric_cols] = mice_imputer.fit_transform(df[numeric_cols])
   
   # Strategy 7: Complete Case Analysis (Drop)
   df_complete = df.dropna()
   ```

**When to Use Each:**

| Strategy | Use Case | Pros | Cons |
|----------|----------|------|------|
| **Mean Imputation** | Normal distributions | Fast, simple | Biased for skewed data |
| **Median Imputation** | Skewed distributions | Robust to outliers | Ignores relationships |
| **Most Frequent** | Categorical data | Preserves majority | Loses minority patterns |
| **Missing Indicator** | MCAR (data quality signal) | Captures missingness pattern | Adds feature |
| **KNN Imputer** | Relationships matter | Uses local similarity | Computationally expensive |
| **MICE** | Complex patterns | Best for MNAR | Slower, complex hyperparameters |

**Documentation Template:**
```
IMPUTATION LOG:
├─ age: SimpleImputer(mean) → filled 847 values
├─ gender: Mode imputation → filled 234 values
├─ annual_income: KNN(k=5) → filled 1,203 values
├─ credit_score: MICE(10 iterations) → filled 556 values
└─ employment_type: Most Frequent → filled 92 values

Total records before: 50,000
Total records after: 50,000
Records with ANY imputation: 2,932 (5.9%)
```

---

### **Day 5: Outlier Detection & Treatment**

**Goal:** Handle anomalous values using 4 statistical methods

**Part D: Outlier Handling**

7. **Detect & Treat Outliers**

   ```python
   import numpy as np
   
   # Method 1: Z-Score Detection
   from scipy import stats
   
   z_scores = np.abs(stats.zscore(df['annual_income']))
   outliers_zscore = z_scores > 3  # Typically |z| > 3
   
   print(f"Outliers detected (Z-score): {outliers_zscore.sum()}")
   df_no_outliers_z = df[~outliers_zscore]
   
   # Method 2: IQR (Interquartile Range)
   Q1 = df['loan_amount'].quantile(0.25)
   Q3 = df['loan_amount'].quantile(0.75)
   IQR = Q3 - Q1
   
   lower_bound = Q1 - 1.5 * IQR
   upper_bound = Q3 + 1.5 * IQR
   
   outliers_iqr = (df['loan_amount'] < lower_bound) | \
                  (df['loan_amount'] > upper_bound)
   print(f"Outliers detected (IQR): {outliers_iqr.sum()}")
   
   # Method 3: Percentile Method
   lower_percentile = df['credit_score'].quantile(0.01)
   upper_percentile = df['credit_score'].quantile(0.99)
   
   outliers_percentile = (df['credit_score'] < lower_percentile) | \
                         (df['credit_score'] > upper_percentile)
   
   # Method 4: Winsorization (Cap Extreme Values)
   # Instead of removing, cap at percentiles
   
   def winsorize_column(series, lower=0.01, upper=0.99):
       lower_val = series.quantile(lower)
       upper_val = series.quantile(upper)
       return series.clip(lower_val, upper_val)
   
   df['annual_income_winsorized'] = \
       winsorize_column(df['annual_income'])
   df['credit_score_winsorized'] = \
       winsorize_column(df['credit_score'], lower=0.05, upper=0.95)
   ```

**Treatment Decision Matrix:**

```
Outlier Severity | Decision | Logic
─────────────────────────────────────────────────────────────
Extreme (|z|>4) | Remove | Data entry error / impossible value
High (|z|>3)    | Winsorize | Cap to 99th percentile
Moderate (IQR)  | Cap | Preserve but reduce impact
Mild (Percentile)| Keep | Natural variation in data
```

**Before/After Comparison:**
```
Feature              Before    After (Winsorized)  Removed
annual_income
  Mean             ₹4,50,000   ₹4,42,000          -
  Std              ₹2,10,000   ₹1,65,000          -
  Max              ₹80,00,000  ₹12,50,000         -
  Outliers         847         0                   -

credit_score
  Mean             650         655                -
  Min              120         300                -
  Max              950         850                -
  Outliers         234         0                  -
```

---

### **Day 6: Feature Engineering (Categorical)**

**Goal:** Transform categorical variables intelligently

**Part E: Encoding Strategies**

8. **Handle Variable Types**

   ```python
   # Mixed Variables (Numeric + Categorical)
   # Some features need cleaning before encoding
   
   # Date & Time Variables
   df['join_date'] = pd.to_datetime(df['join_date'])
   
   # Extract temporal features
   df['join_year'] = df['join_date'].dt.year
   df['join_month'] = df['join_date'].dt.month
   df['join_day_of_week'] = df['join_date'].dt.dayofweek  # 0=Mon, 6=Sun
   df['days_customer'] = (pd.Timestamp.now() - df['join_date']).dt.days
   ```

9. **Categorical Encoding**

   ```python
   # Strategy 1: Ordinal Encoding (Preserves Order)
   # For ordered categories: Primary < Secondary < Graduate < Post-Grad
   
   education_mapping = {
       'Primary': 1,
       'Secondary': 2,
       'Graduate': 3,
       'Post-Graduate': 4
   }
   df['education_level_encoded'] = df['education_level'].map(education_mapping)
   
   # Strategy 2: Label Encoding (Binary Features)
   # For binary categories: Male (0) / Female (1)
   df['gender_encoded'] = (df['gender'] == 'Female').astype(int)
   
   # Strategy 3: One-Hot Encoding (Nominal Categorical)
   # For regions: North, South, East, West (no inherent order)
   df_region_encoded = pd.get_dummies(df['region'], prefix='region', drop_first=True)
   df = pd.concat([df, df_region_encoded], axis=1)
   
   # Strategy 4: One-Hot for Loan Purpose
   df_purpose = pd.get_dummies(df['loan_purpose'], prefix='purpose', drop_first=True)
   df = pd.concat([df, df_purpose], axis=1)
   ```

**Encoding Decision Framework:**

| Variable | Type | Strategy | Reason |
|----------|------|----------|--------|
| `education_level` | Ordinal | Ordinal Encode | Preserves ranking (Primary→Post-Grad) |
| `gender` | Binary | Label Encode | Only 2 categories, interpretable |
| `region` | Nominal | One-Hot | No inherent order; prevents rank assumption |
| `loan_purpose` | Nominal | One-Hot | 5 categories; captures independent effects |
| `employment_type` | Nominal | One-Hot | 3 categories; no ordering |

---

### **Day 7: Numerical Encoding & Scaling**

**Goal:** Transform numerical features for ML algorithms

**Part F: Numerical Feature Engineering**

10. **Binning & Discretization**

    ```python
    # Binning: Convert continuous → discrete categories
    
    # Equal-Width Binning (income groups)
    df['income_bin_equal'] = pd.cut(
        df['annual_income'],
        bins=5,
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    # Equal-Frequency (Quantile) Binning
    df['income_bin_quantile'] = pd.qcut(
        df['annual_income'],
        q=4,
        labels=['Q1', 'Q2', 'Q3', 'Q4'],
        duplicates='drop'
    )
    
    # Custom Binning (Domain-Driven)
    income_bins = [0, 200000, 500000, 1000000, np.inf]
    income_labels = ['Poor', 'Lower Middle', 'Upper Middle', 'Rich']
    df['income_category'] = pd.cut(
        df['annual_income'],
        bins=income_bins,
        labels=income_labels
    )
    
    # Binarization: Create flag for threshold
    df['high_credit_score'] = (df['credit_score'] > 700).astype(int)
    
    # K-Means Binning (for transaction count)
    from sklearn.cluster import KMeans
    
    X = df[['transaction_count']].values
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['transaction_cluster'] = kmeans.fit_predict(X)
    ```

11. **Feature Scaling**

    ```python
    from sklearn.preprocessing import (
        StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
    )
    
    numeric_features = [
        'annual_income', 'loan_amount', 'credit_score', 'spending_ratio'
    ]
    
    # Method 1: Standardization (Z-score scaling)
    # Formula: X_scaled = (X - mean) / std
    # Use for: Normal distributions, algorithms using distance metrics
    
    scaler_standard = StandardScaler()
    df_scaled_standard = scaler_standard.fit_transform(df[numeric_features])
    
    # Method 2: Min-Max Normalization
    # Formula: X_scaled = (X - min) / (max - min) → [0, 1]
    # Use for: Bounded features, neural networks
    
    scaler_minmax = MinMaxScaler()
    df_scaled_minmax = scaler_minmax.fit_transform(df[numeric_features])
    
    # Method 3: Robust Scaling
    # Formula: X_scaled = (X - median) / IQR
    # Use for: Data with outliers (less sensitive than Z-score)
    
    scaler_robust = RobustScaler()
    df_scaled_robust = scaler_robust.fit_transform(df[numeric_features])
    
    # Method 4: Max-Abs Scaling
    # Formula: X_scaled = X / max(|X|) → [-1, 1]
    # Use for: Sparse data, tree-based models
    
    scaler_maxabs = MaxAbsScaler()
    df_scaled_maxabs = scaler_maxabs.fit_transform(df[numeric_features])
    
    # Document the scaling method used
    print("Scaling Method: StandardScaler (Z-score)")
    print(f"Mean (should ≈ 0): {df_scaled_standard.mean(axis=0)}")
    print(f"Std (should ≈ 1): {df_scaled_standard.std(axis=0)}")
    ```

**Scaling Comparison Table:**

| Scaler | Formula | Range | Best For | Sensitive to Outliers |
|--------|---------|-------|----------|----------------------|
| **StandardScaler** | (X - μ) / σ | (-∞, ∞) | Normal data, regression | YES |
| **MinMaxScaler** | (X - min) / (max - min) | [0, 1] | Neural networks | YES |
| **RobustScaler** | (X - median) / IQR | (-∞, ∞) | Outliers present | NO |
| **MaxAbsScaler** | X / max(\|X\|) | [-1, 1] | Sparse data | NO |

---

### **Day 8: Transformations & Final Assembly**

**Goal:** Apply mathematical transformations and assemble final dataset

**Part G: Transformations & Feature Construction**

12. **Apply Transformations**

    ```python
    from sklearn.preprocessing import FunctionTransformer, PowerTransformer
    
    # Log Transformation (right-skewed distributions)
    # spending_ratio is skewed; log normalizes it
    df['spending_ratio_log'] = np.log1p(df['spending_ratio'])
    
    # Reciprocal Transformation
    df['income_reciprocal'] = 1 / (df['annual_income'] + 1)
    
    # Square Root Transformation
    df['spending_ratio_sqrt'] = np.sqrt(df['spending_ratio'])
    
    # Box-Cox Transformation (finds optimal λ)
    # Automatically normalizes skewed data
    df['loan_amount_boxcox'], lambda_param = \
        stats.boxcox(df['loan_amount'] + 1)  # Add 1 to handle zeros
    
    # Yeo-Johnson Transformation (handles zeros and negatives)
    pt = PowerTransformer(method='yeo-johnson')
    df['annual_income_yeojohnson'] = \
        pt.fit_transform(df[['annual_income']])
    
    # Using sklearn Pipeline for automation
    from sklearn.pipeline import Pipeline
    
    def log_transformer(X):
        return np.log1p(X)
    
    transformer = FunctionTransformer(log_transformer)
    X_transformed = transformer.fit_transform(df[['spending_ratio']])
    ```

13. **Construct New Features**

    ```python
    # Feature 1: Debt-to-Income Ratio
    df['debt_to_income_ratio'] = df['loan_amount'] / (df['annual_income'] + 1)
    
    # Feature 2: Average Monthly Transactions
    df['avg_monthly_transactions'] = df['transaction_count'] / 6
    
    # Feature 3: Spending-to-Income Ratio (already exists)
    # df['spending_ratio'] exists; verify it's meaningful
    
    # Feature 4: Credit Score Health Flag
    df['credit_score_healthy'] = (df['credit_score'] >= 700).astype(int)
    
    # Feature 5: Age Group Binning
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 25, 35, 50, 100],
        labels=['Young', 'Prime', 'Mid-Career', 'Senior']
    )
    
    # Feature 6: Repayment Consistency (binary)
    df['good_repayment_history'] = (df['repayment_history'] <= 2).astype(int)
    
    # Feature 7: Loan-to-Income Ratio
    df['loan_to_income_ratio'] = df['loan_amount'] / (df['annual_income'] + 1)
    ```

**Part H: Final Dataset Assembly**

14. **Create Production-Ready Dataset**

    ```python
    # Step 1: Select final features
    feature_columns = [
        # Original features (cleaned)
        'customer_id', 'age', 'gender_encoded', 'education_level_encoded',
        'region_South', 'region_East', 'region_West',  # One-hot encoded
        'annual_income', 'loan_amount', 'credit_score', 
        'spending_ratio', 'repayment_history', 'transaction_count',
        # Engineered features
        'debt_to_income_ratio', 'loan_to_income_ratio',
        'avg_monthly_transactions', 'credit_score_healthy',
        'good_repayment_history', 'days_customer',
        # Scaled features
        'annual_income_scaled', 'loan_amount_scaled',
        # Target
        'default_flag'
    ]
    
    df_final = df[feature_columns].copy()
    
    # Step 2: Validate final dataset
    print("═" * 50)
    print("FINAL DATASET VALIDATION")
    print("═" * 50)
    print(f"Shape: {df_final.shape}")
    print(f"Duplicates: {df_final.duplicated().sum()}")
    print(f"Missing values:\n{df_final.isnull().sum()}")
    print(f"Data types:\n{df_final.dtypes}")
    print(f"Memory usage: {df_final.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Step 3: Export to CSV
    df_final.to_csv('final_cleaned_credit_risk_dataset.csv', index=False)
    print("\n✅ Dataset saved: final_cleaned_credit_risk_dataset.csv")
    ```

15. **Write Comprehensive Report**

    ```
    SUMMARY REPORT STRUCTURE:
    
    1. EXECUTIVE SUMMARY
       - Dataset overview & stats
       - Total records processed: 50,000
       - Final records: 49,156 (after cleaning)
       
    2. MISSING VALUE STRATEGIES & RESULTS
       - Strategy breakdown with percentages
       - Before/after fill rates
       - Impact on downstream features
       
    3. OUTLIER HANDLING RESULTS
       - Detection method comparison
       - Outliers found and treated
       - Statistical impact (mean/std change)
       
    4. ENCODING METHODS APPLIED
       - Categorical features and choices
       - One-hot expanded features
       - Ordinal mapping documentation
       
    5. SCALING & TRANSFORMATIONS
       - Method chosen and rationale
       - Feature statistics (before/after)
       - Normality checks (post-transform)
       
    6. NEWLY ENGINEERED FEATURES
       - All 7 new features with formulas
       - Business interpretation
       - Correlation with target
       
    7. FINAL DATASET READINESS
       - Shape and composition
       - Feature count breakdown
       - Ready for ML modeling ✓
    ```

---

## 🎯 Expected Outcomes

By completing this project, you'll:

✅ **Understand:** Complete data preprocessing workflow for credit risk modeling  
✅ **Implement:** 6+ missing value strategies with decision framework  
✅ **Execute:** Outlier detection using 4 statistical methods  
✅ **Engineer:** Domain-specific financial features (debt ratios, score health, etc.)  
✅ **Transform:** Skewed data using Box-Cox and Yeo-Johnson  
✅ **Scale:** Numerical features using 4 different scaling methods  
✅ **Document:** Production-grade data engineering work  

---

## 📦 Deliverables Checklist

- [ ] **Jupyter Notebook** (`credit_risk_preprocessing.ipynb`)
  - All 8 days of work documented
  - Code with markdown explanations
  - Visualizations for before/after
  - No hard-coded paths

- [ ] **Final CSV Dataset** (`final_cleaned_credit_risk_dataset.csv`)
  - 0 null values in critical columns
  - All features encoded & scaled
  - ~49K-50K clean records

- [ ] **Comprehensive Report** (PDF, 2-3 pages)
  - Theory + observations
  - Strategy justifications
  - Results & metrics
  - Recommendations

## 📂 Project Structure

```text
Credit-Risk-Preprocessing-Project/
│
├── 📂 charts/                         # Analysis & visualization charts
├── 📓 Credit_Risk_Preprocessing_Project.ipynb
├── 📄 README.md                       # Project documentation
├── 📕 Theory_Concepts_Reference.pdf   # Theory reference
├── 📊 credit_risk_main.csv            # Main raw dataset
├── 🧾 customer_metadata.json          # Customer metadata
├── 🌐 economic_indicators_api.json    # Economic indicators data
├── 📊 final_processed_dataset.csv     # Final processed dataset
├── 🗄️ loan_repayment_sql.csv          # Loan repayment data
└── 🎥 video/                          # Project demonstration



## 💡 Pro Tips from 5+ Years Experience

### 1. **Imputation is an Art, Not Science**
- Don't blindly use mean for all numeric columns
- Check distribution shape before choosing strategy
- Document assumptions and their impact
- ⚠️ AVOID: Imputing target variable (causes leakage)

### 2. **Outliers ≠ Always Bad**
- In credit risk, high earners (~outliers) are IMPORTANT
- Remove data entry errors, keep business anomalies
- Winsorization often better than removal
- Preserve 90%+ of data when possible

### 3. **Feature Scaling is Algorithm-Specific**
- **Distance-based (KNN, K-Means):** Must scale → StandardScaler
- **Tree-based (XGBoost, RF):** Don't scale → invariant
- **Neural Networks:** Scale to [0,1] or [-1,1]
- **Linear Models:** Scale important for regularization

### 4. **Categorical Encoding Matters**
- One-Hot → explodes dimensionality; use for <5 categories
- Ordinal → only if true ordering exists
- Target Encoding → advanced; can cause overfitting
- Label Encoding → risky; model might learn false order

### 5. **Always Document Transformations**
- Future you will ask "why did I do this?"
- Keep mapping dictionaries for production
- Save scaler parameters for live data
- Version control your pipelines

### 6. **The 80/20 Rule**
- 80% of ML problems are solved by 20% of feature engineering
- Start simple, iterate
- Don't over-engineer with 100 features; 15-20 is usually optimal
- Feature selection comes after engineering

### 7. **Common Pitfalls to AVOID**

```
❌ Fitting scaler on ALL data (data leakage!)
   ✅ Fit on train → transform on test

❌ Imputing before understanding data
   ✅ Analyze missing patterns → then impute

❌ Using mean for outlier-heavy distributions
   ✅ Use median or IQR-based methods

❌ One-hot encoding high-cardinality features
   ✅ Group rare categories or use target encoding

❌ Forgetting to inverse-transform for interpretation
   ✅ Keep original scale for business reporting

❌ Not validating transformations
   ✅ Check distributions post-transform
```

---

## 🔧 Quick Reference: When to Use What

### Missing Values
```
Normal distribution     → Mean imputation
Skewed distribution    → Median imputation
Categorical data       → Mode imputation
Relationships matter   → KNN or MICE
Data quality signal    → Missing indicator + fill
```

### Outliers
```
|Z-score| > 3         → Likely error, remove
IQR bounds breach     → Assess domain validity
Extreme cases         → Winsorize to percentile
```

### Encoding
```
Ordinal categorical    → Ordinal Encode (preserve order)
Binary categorical     → Label Encode (0/1)
Nominal categorical    → One-Hot Encode
High cardinality       → Target Encode or grouping
```

### Scaling
```
Linear models          → StandardScaler
Neural networks        → MinMaxScaler
Distance-based ML      → StandardScaler
Robust to outliers     → RobustScaler
```

### Transformations
```
Right skew (income)    → Log transform
Left skew              → Square root
Optimize normality     → Box-Cox
Handle zeros/negatives → Yeo-Johnson
```

---

## 📊 Dataset Statistics Template

```
BEFORE PREPROCESSING:
─────────────────────
Total Records:     50,000
Features:          14
Missing %:         8.2%
Outliers:          1,247
Duplicates:        23
Default Rate:      12.5%

AFTER PREPROCESSING:
────────────────────
Total Records:     49,156 (-844)
Features:          21 (7 engineered)
Missing %:         0.0% ✓
Outliers:          0 (winsorized)
Duplicates:        0 ✓
Default Rate:      12.4% (stable)

Quality Score:     98.7% ✓
```

---

## 🚀 Success Criteria

- ✅ 0 null values in final dataset
- ✅ All numerical features scaled to [-3, 3] range
- ✅ All categorical features encoded
- ✅ At least 7 engineered features with business meaning
- ✅ Before/after statistics documented
- ✅ Code is reproducible and well-commented
- ✅ Report explains every decision
- ✅ Dataset ready for ML model training

---

## 📚 References

- Scikit-learn Preprocessing: https://scikit-learn.org/stable/modules/preprocessing.html
- Pandas Documentation: https://pandas.pydata.org/docs/
- "Feature Engineering for Machine Learning" - Alice Zheng
- "Approaching (Almost) Any Machine Learning Problem" - Abhishek Thakur

---

## ⚠️ Important Notes

**This is NOT:**
- A feature selection project (we engineer first, select later)
- A model building project (just data prep)
- An EDA deep-dive (we explore → clean → engineer)

**This IS:**
- Industrial-grade data engineering
- Production-ready pipeline setup
- Foundation for ML model success
- Career-building practical project

---

**Remember: "Garbage in = Garbage out"**  
Your preprocessing quality determines your model's ceiling. Invest time here! 🎯

---

**Good luck! You're building the most critical part of the ML pipeline.** 🚀
