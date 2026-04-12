from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches




st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
}

.main-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    color: #1a1a1a;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}

.subtitle {
    font-size: 1.1rem;
    color: #666;
    font-weight: 300;
    margin-bottom: 2rem;
}

.grade-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 120px;
    height: 120px;
    border-radius: 50%;
    font-size: 3.5rem;
    font-weight: 700;
    font-family: 'DM Serif Display', serif;
    color: white;
    margin: 1rem auto;
}

.metric-card {
    background: #f8f8f6;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    border-left: 4px solid #038141;
    margin-bottom: 0.8rem;
}

.metric-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #888;
    font-weight: 500;
}

.metric-value {
    font-size: 1.6rem;
    font-weight: 600;
    color: #1a1a1a;
    font-family: 'DM Serif Display', serif;
}

.info-box {
    background: #f0f7f0;
    border: 1px solid #c3dfc3;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: #2d5a2d;
}

.warning-box {
    background: #fff8f0;
    border: 1px solid #ffd199;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: #7a4a00;
}

.stSlider > div > div > div {
    background: #038141 !important;
}

div[data-testid="metric-container"] {
    background: #f8f8f6;
    border-radius: 12px;
    padding: 1rem;
    border: 1px solid #eee;
}

.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #1a1a1a;
    border-bottom: 2px solid #038141;
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem 0;
}
</style>
""", unsafe_allow_html=True)


GRADE_COLORS = {
    0: '#038141', 1: '#85BB2F', 2: '#FECB02', 3: '#EE8100', 4: '#E63E11'
}
GRADE_LABELS  = ['A', 'B', 'C', 'D', 'E']
GRADE_DESCRIPTIONS = {
    0: ('Excellent nutritional quality', 'Very low in bad nutrients, rich in fiber and protein.'),
    1: ('Good nutritional quality',      'Good balance — slightly higher in sugar or fat.'),
    2: ('Average nutritional quality',   'Moderate — some nutrients to watch.'),
    3: ('Poor nutritional quality',      'High in bad nutrients. Consume occasionally.'),
    4: ('Very poor nutritional quality', 'Very high in sugar, salt or saturated fat. Limit consumption.')
}

ENGINEERED_FEATURES = [
    'sugar_to_protein_ratio', 'fat_to_protein_ratio', 'fiber_sugar_balance',
    'saturated_fat_proportion', 'fat_carb_ratio', 'carb_sugar_ratio',
    'bad_nutrients_score', 'good_nutrients_score', 'health_balance',
    'total_macros', 'negative_points', 'positive_points', 'approx_nutriscore',
    'fat_category', 'salt_category'
]
REPO_ROOT = Path(__file__).resolve().parents[2]
NUTRISCORE_MODELS_DIR = REPO_ROOT / 'nutriscore' / 'models'


@st.cache_resource
def load_model():
    model = joblib.load(NUTRISCORE_MODELS_DIR / 'nutriscore_model.pkl')
    feature_names = joblib.load(NUTRISCORE_MODELS_DIR / 'feature_names.pkl')
    return model, feature_names

def build_features(inputs: dict) -> pd.DataFrame:
    """Build the full feature vector from raw nutritional inputs."""
    d   = inputs
    eps = 1e-6

    row = {
        'nova_group'              : d['nova_group'],
        'additives_n'             : d['additives_n'],
        'energy_100g'             : d['energy_100g'],
        'fat_100g'                : d['fat_100g'],
        'saturated-fat_100g'      : d['sat_fat_100g'],
        'carbohydrates_100g'      : d['carbs_100g'],
        'sugars_100g'             : d['sugars_100g'],
        'fiber_100g'              : d['fiber_100g'],
        'proteins_100g'           : d['proteins_100g'],
        'salt_100g'               : d['salt_100g'],
        # ratio features
        'sugar_to_protein_ratio'  : d['sugars_100g']    / (d['proteins_100g'] + eps),
        'fat_to_protein_ratio'    : d['fat_100g']       / (d['proteins_100g'] + eps),
        'fiber_sugar_balance'     : d['fiber_100g']     / (d['sugars_100g']   + eps),
        'saturated_fat_proportion': min(d['sat_fat_100g'] / (d['fat_100g'] + eps), 1.0),
        'fat_carb_ratio'          : d['fat_100g']       / (d['carbs_100g']   + eps),
        'carb_sugar_ratio'        : d['sugars_100g']    / (d['carbs_100g']   + eps),
        # composite scores
        'bad_nutrients_score'     : d['sugars_100g'] + d['fat_100g'] + d['salt_100g'] + d['sat_fat_100g'],
        'good_nutrients_score'    : d['proteins_100g'] + d['fiber_100g'],
        'health_balance'          : (d['proteins_100g'] + d['fiber_100g']) - (d['sugars_100g'] + d['fat_100g'] + d['salt_100g'] + d['sat_fat_100g']),
        'total_macros'            : d['fat_100g'] + d['proteins_100g'] + d['carbs_100g'],
        # NutriScore approximation
        'neg_energy_pts'          : min(d['energy_100g'] / 335, 10),
        'neg_satfat_pts'          : min(d['sat_fat_100g'] / 1.0, 10),
        'neg_sugar_pts'           : min(d['sugars_100g'] / 4.5, 10),
        'neg_salt_pts'            : min(d['salt_100g'] / 0.45, 10),
        'negative_points'         : min(d['energy_100g']/335,10) + min(d['sat_fat_100g']/1.0,10) + min(d['sugars_100g']/4.5,10) + min(d['salt_100g']/0.45,10),
        'pos_protein_pts'         : min(d['proteins_100g'] / 1.6, 5),
        'pos_fiber_pts'           : min(d['fiber_100g'] / 0.9, 5),
        'positive_points'         : min(d['proteins_100g']/1.6, 5) + min(d['fiber_100g']/0.9, 5),
        'approx_nutriscore'       : (min(d['energy_100g']/335,10) + min(d['sat_fat_100g']/1.0,10) + min(d['sugars_100g']/4.5,10) + min(d['salt_100g']/0.45,10)) - (min(d['proteins_100g']/1.6,5) + min(d['fiber_100g']/0.9,5)),
        # bins
        'fat_category'            : 0 if d['fat_100g'] < 3 else (1 if d['fat_100g'] < 17.5 else 2),
        'salt_category'           : 0 if d['salt_100g'] < 0.3 else (1 if d['salt_100g'] < 1.5 else 2),
        # additional nutrients
        'fruits-vegetables-legumes_100g': d['fruits_veg_leg'],
        'lactose_100g'            : d['lactose_100g'],
        'phosphorus_100g'         : d['phosphorus_100g'],
        'magnesium_100g'          : d['magnesium_100g'],
    }
    return pd.DataFrame([row])


def get_shap_explanation(model, X_input, feature_names):
    """Compute SHAP values for a single prediction."""
    try:
        explainer   = shap.TreeExplainer(model)
        shap_vals   = explainer(X_input)
        pred_class  = model.predict(X_input)[0]
        return shap_vals, pred_class
    except Exception:
        return None, model.predict(X_input)[0]


col_title, col_logo = st.columns([3, 1])
with col_title:
    st.markdown('<p class="main-title">🥗 NutriScore<br>Decoder</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Enter the nutritional values of any food product and our ML model will predict its NutriScore grade (A → E).</p>', unsafe_allow_html=True)

st.divider()


try:
    model,  feature_names = load_model()
    st.sidebar.success('✅ Model loaded successfully')
    st.sidebar.caption(f'Features : {len(feature_names)}')
except FileNotFoundError:
    st.error('❌ Model files not found. Make sure `models/` folder contains `nutriscore_model.pkl`, `scaler.pkl` and `feature_names.pkl`.')
    st.stop()


st.sidebar.markdown('## 🥄 Quick Presets')
st.sidebar.caption('Load a food example to test the model quickly.')

presets = {
    '🥗 Fresh salad '   : dict(energy_100g=50,   fat_100g=0.5,  sat_fat_100g=0.1, carbs_100g=5,   sugars_100g=2.0, fiber_100g=3.0, proteins_100g=2.0, salt_100g=0.05, nova_group=1, additives_n=0,  fruits_veg_leg=60.0,  lactose_100g=0.0, phosphorus_100g=30.0,  magnesium_100g=15.0),
    '🍌 Banana '        : dict(energy_100g=371,  fat_100g=0.3,  sat_fat_100g=0.1, carbs_100g=23,  sugars_100g=12,  fiber_100g=2.6, proteins_100g=1.1, salt_100g=0.0,  nova_group=1, additives_n=0,  fruits_veg_leg=100.0, lactose_100g=0.0, phosphorus_100g=22.0,  magnesium_100g=27.0),
    '🥛 Whole milk '    : dict(energy_100g=264,  fat_100g=3.5,  sat_fat_100g=2.1, carbs_100g=4.8, sugars_100g=4.8, fiber_100g=0.0, proteins_100g=3.2, salt_100g=0.1,  nova_group=1, additives_n=0,  fruits_veg_leg=0.0,  lactose_100g=4.7, phosphorus_100g=95.0,  magnesium_100g=11.0),
    '🍕 Pepperoni pizza': dict(energy_100g=1100, fat_100g=14,   sat_fat_100g=5.5, carbs_100g=31,  sugars_100g=3.0, fiber_100g=2.0, proteins_100g=12,  salt_100g=1.5,  nova_group=4, additives_n=5,  fruits_veg_leg=5.0,  lactose_100g=0.5, phosphorus_100g=180.0, magnesium_100g=20.0),
    '🍫 Chocolate bar ' : dict(energy_100g=2200, fat_100g=35,   sat_fat_100g=20,  carbs_100g=56,  sugars_100g=48,  fiber_100g=3.0, proteins_100g=5.0, salt_100g=0.1,  nova_group=4, additives_n=3,  fruits_veg_leg=0.0,  lactose_100g=0.0, phosphorus_100g=130.0, magnesium_100g=45.0),
    '🍟 French fries '     : dict(energy_100g=1400, fat_100g=15.0, sat_fat_100g=2.0, carbs_100g=41.0, sugars_100g=0.3,  fiber_100g=3.8, proteins_100g=3.5,  salt_100g=0.5,  nova_group=3, additives_n=2,  fruits_veg_leg=0.0,  lactose_100g=0.0,  phosphorus_100g=85.0,  magnesium_100g=23.0),
    '🥤 Cola soda '        : dict(energy_100g=180,  fat_100g=0.0,  sat_fat_100g=0.0, carbs_100g=10.6, sugars_100g=10.6, fiber_100g=0.0, proteins_100g=0.0,  salt_100g=0.0,  nova_group=4, additives_n=7,  fruits_veg_leg=0.0,  lactose_100g=0.0,  phosphorus_100g=0.0,   magnesium_100g=0.0),
}
selected_preset = st.sidebar.radio('Select a preset', list(presets.keys()), index=None, label_visibility='collapsed')

left_col, right_col = st.columns([1, 1], gap='large')

with left_col:
    st.markdown('<p class="section-header">📋 Nutritional Values (per 100g)</p>', unsafe_allow_html=True)

    # load preset if selected
    defaults = presets[selected_preset] if selected_preset else dict(
        energy_100g=500, fat_100g=5.0, sat_fat_100g=2.0, carbs_100g=30,
        sugars_100g=10, fiber_100g=3.0, proteins_100g=8.0, salt_100g=0.5,
        nova_group=2, additives_n=2, fruits_veg_leg=0.0, lactose_100g=0.0,
        phosphorus_100g=0.0, magnesium_100g=0.0
    )

    c1, c2 = st.columns(2)
    with c1:
        energy   = st.number_input('⚡ Energy (kJ)',          min_value=0.0, max_value=3800.0, value=float(defaults['energy_100g']),  step=10.0)
        fat      = st.number_input('🧈 Fat (g)',              min_value=0.0, max_value=100.0,  value=float(defaults['fat_100g']),     step=0.5)
        sat_fat  = st.number_input('🔴 Saturated fat (g)',    min_value=0.0, max_value=100.0,  value=float(defaults['sat_fat_100g']), step=0.5)
        carbs    = st.number_input('🌾 Carbohydrates (g)',    min_value=0.0, max_value=100.0,  value=float(defaults['carbs_100g']),   step=0.5)
    with c2:
        sugars   = st.number_input('🍬 Sugars (g)',           min_value=0.0, max_value=100.0,  value=float(defaults['sugars_100g']),  step=0.5)
        fiber    = st.number_input('🌿 Fiber (g)',             min_value=0.0, max_value=100.0,  value=float(defaults['fiber_100g']),   step=0.5)
        proteins = st.number_input('💪 Proteins (g)',          min_value=0.0, max_value=100.0,  value=float(defaults['proteins_100g']),step=0.5)
        salt     = st.number_input('🧂 Salt (g)',             min_value=0.0, max_value=100.0,  value=float(defaults['salt_100g']),    step=0.1)

    st.markdown('<p class="section-header">🥦 Additional Nutrients (per 100g)</p>', unsafe_allow_html=True)
    c1b, c2b, c3b, c4b = st.columns(4)
    with c1b:
        fruits_veg_leg = st.number_input('🥦 Fruits/Veg/Leg (g)', min_value=0.0, max_value=100.0,  value=float(defaults['fruits_veg_leg']),  step=1.0, help='Fruits, vegetables and legumes content')
    with c2b:
        lactose        = st.number_input('🥛 Lactose (g)',         min_value=0.0, max_value=100.0,  value=float(defaults['lactose_100g']),    step=0.1)
    with c3b:
        phosphorus     = st.number_input('🔵 Phosphorus (mg)',     min_value=0.0, max_value=2000.0, value=float(defaults['phosphorus_100g']), step=1.0)
    with c4b:
        magnesium      = st.number_input('🟢 Magnesium (mg)',      min_value=0.0, max_value=1000.0, value=float(defaults['magnesium_100g']),  step=1.0)

    st.markdown('<p class="section-header">🏷️ Product Info</p>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        nova_group  = st.selectbox('NOVA group (processing level)',
                                   [1, 2, 3, 4],
                                   index=[1,2,3,4].index(defaults['nova_group']),
                                   help='1=Unprocessed, 2=Processed ingredients, 3=Processed, 4=Ultra-processed')
    with c4:
        additives_n = st.number_input('Number of additives', min_value=0, max_value=50,
                                      value=int(defaults['additives_n']), step=1)

    # validation
    if sat_fat > fat:
        st.markdown('<div class="warning-box">⚠️ Saturated fat cannot exceed total fat.</div>', unsafe_allow_html=True)
    if sugars > carbs:
        st.markdown('<div class="warning-box">⚠️ Sugars cannot exceed total carbohydrates.</div>', unsafe_allow_html=True)

    predict_btn = st.button('🔍 Predict NutriScore', type='primary', width='stretch')


with right_col:
    st.markdown('<p class="section-header">🎯 Prediction Result</p>', unsafe_allow_html=True)

    if predict_btn or selected_preset:
        inputs = dict(
            energy_100g=energy, fat_100g=fat, sat_fat_100g=sat_fat,
            carbs_100g=carbs, sugars_100g=sugars, fiber_100g=fiber,
            proteins_100g=proteins, salt_100g=salt,
            nova_group=nova_group, additives_n=additives_n,
            fruits_veg_leg=fruits_veg_leg, lactose_100g=lactose,
            phosphorus_100g=phosphorus, magnesium_100g=magnesium
        )

        X_input = build_features(inputs)
        # align columns with model
        for col in feature_names:
            if col not in X_input.columns:
                X_input[col] = 0
        X_input = X_input[feature_names]

        # predict
        pred_class   = model.predict(X_input)[0]
        pred_proba   = model.predict_proba(X_input)[0]
        grade_letter = GRADE_LABELS[pred_class]
        grade_color  = GRADE_COLORS[pred_class]
        desc_title, desc_body = GRADE_DESCRIPTIONS[pred_class]

        # grade badge
        st.markdown(f"""
        <div style="text-align:center; margin: 1rem 0;">
            <div class="grade-badge" style="background:{grade_color}; display:inline-flex;">
                {grade_letter}
            </div>
            <h3 style="margin:0.5rem 0 0.2rem; font-family: DM Serif Display, serif;">{desc_title}</h3>
            <p style="color:#666; font-size:0.95rem;">{desc_body}</p>
        </div>
        """, unsafe_allow_html=True)

        # probability bars
        st.markdown('**Confidence per grade:**')
        for i, (label, prob) in enumerate(zip(GRADE_LABELS, pred_proba)):
            col_label, col_bar = st.columns([1, 5])
            with col_label:
                st.markdown(f'<span style="background:{GRADE_COLORS[i]};color:white;padding:2px 8px;border-radius:4px;font-weight:600;">{label}</span>', unsafe_allow_html=True)
            with col_bar:
                st.progress(float(prob), text=f'{prob*100:.1f}%')

        st.divider()

      
        st.markdown('**Key computed values:**')
        approx_score = inputs['energy_100g']/335 + inputs['sat_fat_100g'] + inputs['sugars_100g']/4.5 + inputs['salt_100g']/0.45 - min(inputs['proteins_100g']/1.6, 5) - min(inputs['fiber_100g']/0.9, 5)

        m1, m2, m3 = st.columns(3)
        m1.metric('Approx. NutriScore', f'{approx_score:.1f}', help='Reconstructed from official formula. Higher = worse quality.')
        m2.metric('Bad nutrients', f'{inputs["sugars_100g"] + inputs["fat_100g"] + inputs["salt_100g"] + inputs["sat_fat_100g"]:.1f}g')
        m3.metric('Good nutrients', f'{inputs["proteins_100g"] + inputs["fiber_100g"]:.1f}g')

        st.divider()

        
        with st.expander('🔬 Why this grade? — SHAP Explanation', expanded=False):
            try:
                explainer  = shap.TreeExplainer(model)
                shap_vals  = explainer(X_input)

               
                fig, ax = plt.subplots(figsize=(8, 5))
                shap_single = shap.Explanation(
                    values        = shap_vals.values[0, :, pred_class],
                    base_values   = shap_vals.base_values[0, pred_class],
                    data          = shap_vals.data[0],
                    feature_names = feature_names
                )
                shap.plots.waterfall(shap_single, max_display=10, show=False)
                plt.title(f'Why Grade {grade_letter}? — Feature contributions', fontsize=11)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                st.caption('Red bars push toward higher grades (worse quality). Blue bars push toward lower grades (better quality).')
            except Exception as e:
                st.info(f'SHAP explanation not available: {e}')

    else:
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem; color: #aaa;">
            <div style="font-size:4rem;">🥗</div>
            <p style="font-size:1.1rem; margin-top:1rem;">Enter nutritional values on the left<br>and click <strong>Predict NutriScore</strong></p>
            <p style="font-size:0.9rem;">Or pick a preset from the sidebar →</p>
        </div>
        """, unsafe_allow_html=True)


st.divider()
st.markdown("""
<div style="text-align:center; color:#aaa; font-size:0.8rem; padding: 1rem 0;">
    NutriScore Decoder — Machine Learning Project | Open Food Facts dataset<br>
    Model: Tuned XGBoost | F1-Macro: 0.881 | 25,000 products | 5 classes
</div>
""", unsafe_allow_html=True)
