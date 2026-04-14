# 🥗 Nutrition Intelligence System
## NutriScore & Macro-Predictor
### Predicting the NutriScore label A-E & Predicting Protein, Fat and Sugar Content of Food Products

---

## 👥 Students

- **Aline Aloulou Mahjoub** : NutriScore
- **Rahma Essaiem** : Macro-Predctor

---

## 📌 Project Overview
In this project we want to explore how machine learning can be used to analyze nutritional information and help people better understand the food they consume.It focuses on building a nutrition intelligence system composed of two machine learning models
# NutriScore
A classification model that predicts the NutriScore label (A, B, C, D, or E) of a food product based on its nutritional values.
# Macro-Predictor
A multi-output regression model that predictes Protein, Fat and Sugar content per 100g using micronutrient profiles and food category These two models together 

---

## 🎯 Problem Statement
Nowadays, many people don’t pay enough attention to their health because of their busy lives. They often eat fast food or follow unhealthy eating habits without thinking about what their body really needs.

At the same time, some people want to improve their diet but don’t consult nutrition experts. Instead, they follow advice from social media, which is not always reliable or suitable for everyone.

Another problem is that not all food products have clear nutritional labels, making it hard to know how healthy they are or what nutrients they contain.

That’s why we created this **Nutrition Intelligence System**, a tool that helps analyze food products and estimate their nutritional value so users can make better and healthier choices
---

## 📦 Dataset
 ### **1. Nutriscore**

| Property | Detail |
|----------|--------|
| **Source** | [Open Food Facts](https://world.openfoodfacts.org/data) |
| **Full size** | ~3,000,000 products x 209 columns |
| **Download** | `https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz` |

Open Food Facts is a collaborative, crowd-sourced database. This means the data is highly heterogeneous it has many missing values, physically impossible entries, and inconsistent formats. This makes it a realistic and challenging dataset for a Machine Learning project

### **2. Macro-Predictor**

| Property | Detail |
|----------|--------|
| **Source** | USDA National Nutrient Database for Standard Reference — SR Legacy (April 2018) |
| **Full size** | 7,793 food items × up to 474 nutrients |
| **Download** | `https://fdc.nal.usda.gov/download-datasets/` |

**Files used:**
| File | Description | Rows |
|---|---|---|
| food.csv | Food identifiers and category IDs | 7,793 |
| food_nutrient.csv | Nutrient amounts per food | 644,125 |
| nutrient.csv | Nutrient definitions and units | 474 |
| food_category.csv | Food category names | 25 |

## ⚠️ Important
Due to size limitations, datasets are not included in this repository.

You can download them here:

🔗 Nutrition Intelligence System Dataset (Google Drive):
https://drive.google.com/drive/folders/11cHKFzFWUQdnXSP3dMuMZ4D2RpzTqURH?usp=sharing

### How to use:
1. Download and extract the folder
2. Place files inside:
   - `nutriscore/data/`
   - `macro_predictor/data/`
3. Run notebooks or Streamlit app
---

## 🗂️ Repository Structure

```
Nutrition-Intelligence-System/
│
├── nutriscore/
│   ├── data/
│   │   ├── en.openfoodfacts.org.products.csv.gz
│   │   ├── openfoodfacts_sample.csv
│   │   ├── sampled_data.csv
│   │   └── data_cleaned.csv
│   │
│   ├── notebooks/
│   │   ├── 01_EDA.ipynb
│   │   └── 02_Modeling.ipynb
│   │
│   ├── models/
│   │   ├── nutriscore_model.pkl
│   │   └── feature_names.pkl
│   │
│   ├── requirements.txt
│   └── plots/
│  
│
├── macro_predictor/
│   ├── data/
│   │   ├── food.csv
│   │   ├── food_nutrient.csv
│   │   ├── nutrient.csv
│   │   ├── food_category.csv
│   │   ├── original_food_dataset.csv
│   │   ├── final_original_food_dataset.csv
│   │   └── final_corrupted_food_dataset.csv
│   │
│   ├── notebooks/
│   │   ├── notebook1_preprocessing_original.ipynb
│   │   ├── notebook2_preprocessing_corrupted.ipynb
│   │   ├── notebook3_EDA.ipynb
│   │   ├── notebook4_model_training.ipynb
│   │   └── final_model.ipynb
│   │
│   ├── models/
│   │   ├── final_model.pkl
│   │   └── selected_features.pkl
│   └── requirements.txt  
│
├── nutrionalApp/
│   ├── app.py
│   ├── pages/
│   │   ├── Macro_Predictor.py
│   │   └── NutriScore.py
│   └── requirements.txt
├── requirements.txt
├── README.md
├── Presentation.pdf
└── .gitignore  
```

---

## ⚙️ Installation & Setup

### Option 1 — pip

```bash
# 1. Clone the repository
git clone https://github.com/alinealouloumahjoub/Nutrition-Intelligence-System.git
cd Nutrition-Intelligence-System

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook notebooks/
```

### Option 2 — Conda

```bash
conda create -n nutriscore python=3.11
conda activate nutriscore
pip install -r requirements.txt
jupyter notebook notebooks/
```

### Run the Web Application
```bash
streamlit run app.py
```

Make sure pkl files are are in the same folder as `app.py`.

### Input Features Required
The app requires nutritional values per 100g of food plus a food category selection. Quick food presets are available in the sidebar for testing (Chicken Breast, Orange, Dark Chocolate, Olive Oil)

---
## **1. Nutriscore** 

### Notebook 1 — EDA (`01_EDA.ipynb`)

| Section | What we analyze |
|---------|-----------------|
| Dataset Overview | Shape, data types, memory usage |
| Missing Values | Distribution of NaN across 209 columns |
| Target Variable | NutriScore grade distribution, class imbalance |
| Feature Analysis | Numeric vs categorical, column families |
| Correlation Analysis | Top 20 features most correlated with the target |
| Feature Distributions | Histograms of top nutritional features |
| Correlation Heatmap | Inter-feature correlations |
| Outlier Detection | IQR method on top 20 features |

### Notebook 2 — Modeling (`02_Modeling.ipynb`)

| Section | Description |
|---------|-------------|
| Data Loading | Chunk-based loading + stratified sampling (5,000/grade) |
| Preprocessing | Feature selection, imputation, impossible value removal |
| Feature Engineering | 15 engineered features across 4 families |
| Train/Test Split | 80/20 stratified split, scaler fitted on train only |
| Modeling | Logistic Regression, Random Forest, XGBoost |
| Hyperparameter Tuning | RandomizedSearchCV : 30 trials, 5-fold CV |
| Error Analysis | Confusion matrices, per-class error rates, learning curves |
| SHAP Interpretability | Global importance, beeswarm, waterfall plots |

---

### 📊 Results

| Model | Accuracy | F1-Macro |
|-------|----------|----------|
| Logistic Regression | 72.9% | 0.730 |
| Random Forest | 87.8% | 0.878 |
| XGBoost | 87.8% | 0.878 |
| **Tuned XGBoost** | **89.1%** | **0.892** |

**5-Fold CV (Tuned XGBoost):** 0.880 ± 0.008  
**CV / Test gap:** 0.012 =>  No overfitting

### Per-class recall (Tuned XGBoost)

| Grade | Recall |
|-------|--------|
| A | 88.4% |
| B | 89.6% |
| C | 87.2% |
| D | 91.9% |
| E | 93.7% |

---

### 🧠 Key Technical Decisions

**Why stratified sampling (5,000/grade)?**
The original dataset is heavily skewed — grade E appears 3× more than grade B. We sampled 5,000 products per grade to build a perfectly balanced training set, preventing the model from systematically favoring majority classes.

**Why no IQR capping?**
We tested IQR outlier capping and our F1-Macro dropped from 0.87 to 0.33. Nutritional data has naturally wide distributions — a biscuit legitimately has 50g of sugar per 100g. That asymmetry is the signal, not noise.

**Why global median imputation instead of KNN?**
KNN imputation on 25,000 rows took over 60 minutes. Per-grade imputation caused data leakage (F1=1.0 — the model was detecting imputation patterns, not learning nutrition). Global median is fast, clean, and leak-free.

**Why `approx_nutriscore` instead of `nutriscore_score`?**
`nutriscore_score` is the answer directly computed by the official algorithm. Using it would be data leakage. We reconstructed the formula from scratch using raw nutrient values — this is legitimate feature engineering with a correlation of 0.79 with the target.

**Why RandomizedSearchCV over GridSearchCV?**
With 8 hyperparameters and 3-4 values each, GridSearch would require thousands of model fits. RandomizedSearch samples 30 random combinations across the full parameter space, finding near-optimal solutions in a fraction of the time.

---

---
## **2. Macro-Predictor** 
### Notebook 1 :Data Preprocessing (Original Data)
Loads the USDA SR Legacy dataset from 4 CSV files and following steps to build the final dataset used for modeling:
- Column selection and copy creation
- Removal of leakage nutrients (sub-components of targets), archived and irrelevant nutrients 
- keeping only nutrients that exists in food_nutrient file
- Merging food data with category descriptions
- Pivoting food_nutrient from long format to wide format
- Final merge into one complete dataframe
- Outlier removal using threshold (100g per 100g)
- encoding of food categories
-removing food id fdc_id since it's no more needed
- Export of final dataset

### Notebook 2 : Data Preprocessing (Corrupted Data)
- Using the non encoded data from the notebook 1
- Discovery of data problems (nulls, duplicates, wrong dtypes, negative values, string pollution)
- cleaning  and following the same logic as notebook 1


### Notebook 3 : Exploratory Data Analysis
Performs EDA on both datasets before encoding to understand data structure and feature-target relationships:
- Descriptive statistics
- Target variable distributions
- Correlation heatmap
- Scatter plots of targets vs most correlated features
- Univariate F-tests for statistical feature validation
- Boxplots by food category
- Skewness analysis
- Comparison between clean and corrupted datasets

### Notebook 4 : Model Training
Trains, evaluates and compares 5 regression models on both datasets:
- Random Forest (good for non linear and complex realtionship)
- Gradient Boosting (builds trees sequentially, where each new tree corrects the errors of the previous ones)
- XGBoost (an optimized version of Gradient Boosting )
- SVR with RBF kernel (captures complex non-linear relationships by mapping data into higher dimensions using kernel trick)
- Linear Regression (used to test linear relationships between features and targets)

For each model: training, evaluation (R², MAE, RMSE), overfitting check (train vs test gap).
Additional analysis: cross validation, hyperparameter tuning with Randomized Search, feature importance, actual vs predicted plots, residual plots, clean vs corrupted comparison.

**Note** :for corrupted cleand data only random forest, gradient boost and xgboost were used

### Final Model Notebook
Trains the final selected model (Gradient Boosting with tuned hyperparameters and selected features) and saves it as a `.pkl` file for deployment

---

### 📊 Results

| Model               | Avg R² | Avg MAE | Avg RMSE |
|--------------------|--------|---------|----------|
| **Gradient Boosting**  | **0.8790** | **2.2689**  | **4.4508**   |
| XGBoost            | 0.8776 | 2.2779  | 4.4967   |
| Random Forest      | 0.8739 | 1.9006  | 4.5337   |
| SVR RBF            | 0.8558 | 2.3904  | 5.0510   |
| Linear Regression  | 0.7451 | 3.5830  | 6.6684   |

***Model:** Gradient Boosting (tuned) with 24 selected features

**Protein**
- R²  : 0.924  
- MAE : 1.608  
- RMSE: 3.090  

**Total lipid (fat)**
- R²  : 0.963  
- MAE : 1.787  
- RMSE: 3.234  

**Sugars, Total**
- R²  : 0.767  
- MAE : 3.105  
- RMSE: 6.652  

**Key findings:**
- Water content is the strongest predictor of Fat (importance 0.55)
- Phosphorus is the strongest predictor of Protein (importance 0.40)
- Carbohydrate is the strongest predictor of Sugar (importance 0.41)
- Sugar is the hardest target due to complex non-linear relationships
- models performance mainly boosters is very close
- the model is stable across folds
Final Model Performance (Clean corrupted Dataset)
 #### **XGBoost Results**

**Protein**
- R²  : 0.846  
- MAE : 2.491  
- RMSE: 3.951  

**Total lipid (fat)**
- R²  : 0.765  
- MAE : 3.050  
- RMSE: 5.317  

**Sugars, Total**
- R²  : 0.236  
- MAE : 2.396  
- RMSE: 5.614  

**Key findings:**
- Data corruption caused a big performance drop confirming data quality and preprocessing step is very important

---

### 🧠 Key Technical Decisions

**multi-output regression** Predicting all 3 targets at once instead of using separate models helps the model learn relationships between the targets and avoids redundancy.

**random forest** rf show strong performance very similar to boosters but train-test gap makes boosters more reliable on unseen data.

**using threshold not IQR** Since all values are per 100g of food, any value above 100g is physically impossible 

**feature units** : because features have different scales and some nutrients naturally have much higher ranges, scaling is necessary for models like SVR to ensure all variables contribute fairly(changing units shrink values and make patterns harder to learn that's why scaling is a better choice)

**median imputation** Skewness analysis revealed values up to 78 for some nutrients mean imputation would be distorted by extreme values. Median is robust to outliers.(still for sugar most values are either very high or low near 0 replacing with median every corrupted sugar value gets replaced with approximatly 0g regardless of whether the real value was 0g or high)

---



---

