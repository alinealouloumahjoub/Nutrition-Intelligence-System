import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif;
}

.main-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    color: #1a1a2e;
    line-height: 1.15;
    margin-bottom: 0.3rem;
}

.subtitle {
    font-size: 1.05rem;
    color: #666;
    font-weight: 300;
    margin-bottom: 1.5rem;
}

.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.25rem;
    color: #1a1a2e;
    border-bottom: 2px solid #2ecc71;
    padding-bottom: 0.3rem;
    margin: 1.5rem 0 1rem 0;
}

.result-card {
    background: linear-gradient(135deg, #f8fffe 0%, #e8f8f0 100%);
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid #c3e6cb;
    margin-bottom: 1rem;
    text-align: center;
}

.result-value {
    font-family: 'Playfair Display', serif;
    font-size: 2.5rem;
    font-weight: 700;
    color: #1a1a2e;
}

.result-label {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #888;
    font-weight: 500;
}

.result-unit {
    font-size: 0.9rem;
    color: #555;
    font-weight: 400;
}

.warning-box {
    background: #fff8f0;
    border: 1px solid #ffd199;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: #7a4a00;
}

.info-box {
    background: #f0f7ff;
    border: 1px solid #bee3f8;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: #2b6cb0;
}

.preset-info {
    background: #f0fff4;
    border-left: 3px solid #2ecc71;
    padding: 0.6rem 1rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.85rem;
    color: #276749;
    margin: 0.5rem 0;
}

div[data-testid="metric-container"] {
    background: #f8fffe;
    border-radius: 14px;
    padding: 0.6rem;   /* smaller padding */
    border: 1px solid #c3e6cb;
}


[data-testid="stMetricLabel"] {
    font-size: 14px !important;
    margin-bottom: 2px;
}


[data-testid="stMetricValue"] {
    font-size: 14px !important;
    font-weight: 600;
}

/* Optional: reduce spacing */
div[data-testid="metric-container"] > div {
    gap: 2px;
}
.stButton > button {
    background: linear-gradient(135deg, #2ecc71, #27ae60);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 1rem;
    padding: 0.7rem 2rem;
    transition: all 0.2s;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #27ae60, #219a52);
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(46,204,113,0.3);
}

.footer-text {
    text-align: center;
    color: #aaa;
    font-size: 0.8rem;
    padding: 1rem 0;
}

.quality-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-left: 8px;
}
</style>
""", unsafe_allow_html=True)


REPO_ROOT = Path(__file__).resolve().parents[2]
MACRO_MODELS_DIR = REPO_ROOT / 'macro_predictor' / 'models'


@st.cache_resource
def load_model():
    model = joblib.load(MACRO_MODELS_DIR / 'final_model.pkl')
    selected_features = joblib.load(MACRO_MODELS_DIR / 'selected_features.pkl')
    return model, selected_features

try:
    model, selected_features = load_model()
    st.sidebar.success('✅ Model loaded successfully')
    st.sidebar.caption(f'Features used: {len(selected_features)}')
    st.sidebar.caption('Model: Gradient Boosting')
    st.sidebar.caption('Dataset: USDA SR Legacy')
except FileNotFoundError:
    st.error('❌ Model files not found. Make sure `final_model.pkl` and `selected_features.pkl` are in the same folder.')
    st.stop()

numeric_features = [f for f in selected_features if not f.startswith('cat_')]
category_features = [f for f in selected_features if f.startswith('cat_')]
all_categories = [f.replace('cat_', '') for f in category_features] + ['Other']


st.sidebar.markdown('---')
st.sidebar.markdown('## 🍽️ Quick Food Presets')
st.sidebar.caption('Load a real food example to test quickly.')

presets = {
    '🍗 Chicken Breast': {
        'Phosphorus, P': 220.0, 'Carbohydrate, by difference': 0.0,
        'Zinc, Zn': 1.0, 'Cholesterol': 85.0, 'Selenium, Se': 27.0,
        'Water': 74.0, 'Magnesium, Mg': 28.0, 'Iron, Fe': 1.0,
        'Niacin': 14.0, 'Potassium, K': 256.0, 'Ash': 1.1, 'Sodium, Na': 74.0,
        'Vitamin E (alpha-tocopherol)': 0.3, 'Manganese, Mn': 0.02,
        'Choline, total': 85.0, 'Theobromine': 0.0, 'Vitamin B-12': 0.3,
        'Vitamin K (phylloquinone)': 0.0, 'Fiber, total dietary': 0.0,
        'Vitamin D (D2 + D3)': 0.1, 'category': 'Poultry Products'
    },
    '🍊 Orange': {
        'Phosphorus, P': 14.0, 'Carbohydrate, by difference': 12.0,
        'Zinc, Zn': 0.1, 'Cholesterol': 0.0, 'Selenium, Se': 0.5,
        'Water': 87.0, 'Magnesium, Mg': 10.0, 'Iron, Fe': 0.1,
        'Niacin': 0.3, 'Potassium, K': 181.0, 'Ash': 0.4, 'Sodium, Na': 0.0,
        'Vitamin E (alpha-tocopherol)': 0.2, 'Manganese, Mn': 0.03,
        'Choline, total': 8.4, 'Theobromine': 0.0, 'Vitamin B-12': 0.0,
        'Vitamin K (phylloquinone)': 0.0, 'Fiber, total dietary': 2.4,
        'Vitamin D (D2 + D3)': 0.0, 'category': 'Fruits and Fruit Juices'
    },
    '🍫 Dark Chocolate': {
        'Phosphorus, P': 132.0, 'Carbohydrate, by difference': 60.0,
        'Zinc, Zn': 1.4, 'Cholesterol': 8.0, 'Selenium, Se': 4.2,
        'Water': 1.4, 'Magnesium, Mg': 65.0, 'Iron, Fe': 3.1,
        'Niacin': 0.4, 'Potassium, K': 365.0, 'Ash': 1.5, 'Sodium, Na': 11.0,
        'Vitamin E (alpha-tocopherol)': 0.5, 'Manganese, Mn': 1.1,
        'Choline, total': 23.0, 'Theobromine': 400.0, 'Vitamin B-12': 0.2,
        'Vitamin K (phylloquinone)': 3.6, 'Fiber, total dietary': 7.0,
        'Vitamin D (D2 + D3)': 0.0, 'category': 'Sweets'
    },
    '🫒 Olive Oil': {
        'Phosphorus, P': 0.0, 'Carbohydrate, by difference': 0.0,
        'Zinc, Zn': 0.0, 'Cholesterol': 0.0, 'Selenium, Se': 0.0,
        'Water': 0.2, 'Magnesium, Mg': 0.0, 'Iron, Fe': 0.6,
        'Niacin': 0.0, 'Potassium, K': 1.0, 'Ash': 0.0, 'Sodium, Na': 2.0,
        'Vitamin E (alpha-tocopherol)': 14.0, 'Manganese, Mn': 0.0,
        'Choline, total': 0.3, 'Theobromine': 0.0, 'Vitamin B-12': 0.0,
        'Vitamin K (phylloquinone)': 60.0, 'Fiber, total dietary': 0.0,
        'Vitamin D (D2 + D3)': 0.0, 'category': 'Fats and Oils'
    },
}

selected_preset = st.sidebar.radio('Select a preset', list(presets.keys()), 
                                    index=None, label_visibility='collapsed')

if selected_preset:
    preset_data = presets[selected_preset]
    st.sidebar.markdown(f'<div class="preset-info">✅ Loaded: <strong>{selected_preset}</strong><br>Category: {preset_data["category"]}</div>', unsafe_allow_html=True)


col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown('<p class="main-title">🥗 Nutrition Intelligence<br>System</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Enter nutritional values of any food and our Gradient Boosting model will predict its Protein, Fat and Sugar content per 100g.</p>', unsafe_allow_html=True)



st.divider()


left_col, right_col = st.columns([1.1, 1], gap='large')

with left_col:
    st.markdown('<p class="section-header">📋 Nutritional Values (per 100g)</p>', unsafe_allow_html=True)

    def get_preset_val(feat, default=0.0):
        if selected_preset and feat in presets[selected_preset]:
            return float(presets[selected_preset][feat])
        return default

    inputs = {}

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**🔬 Minerals & Macros**")
        inputs['Phosphorus, P']                 = st.number_input('Phosphorus, P (mg)',          min_value=0.0, max_value=10000.0, value=get_preset_val('Phosphorus, P'),          step=1.0)
        inputs['Carbohydrate, by difference']   = st.number_input('Carbohydrate (g)',            min_value=0.0, max_value=100.0,   value=get_preset_val('Carbohydrate, by difference'), step=0.1)
        inputs['Zinc, Zn']                      = st.number_input('Zinc, Zn (mg)',              min_value=0.0, max_value=100.0,   value=get_preset_val('Zinc, Zn'),                step=0.1)
        inputs['Cholesterol']                   = st.number_input('Cholesterol (mg)',            min_value=0.0, max_value=3000.0,  value=get_preset_val('Cholesterol'),             step=1.0)
        inputs['Selenium, Se']                  = st.number_input('Selenium, Se (µg)',           min_value=0.0, max_value=2000.0,  value=get_preset_val('Selenium, Se'),            step=0.1)
        inputs['Water']                         = st.number_input('Water (g)',                   min_value=0.0, max_value=100.0,   value=get_preset_val('Water'),                   step=0.1)
        inputs['Magnesium, Mg']                 = st.number_input('Magnesium, Mg (mg)',          min_value=0.0, max_value=1000.0,  value=get_preset_val('Magnesium, Mg'),           step=1.0)
        inputs['Iron, Fe']                      = st.number_input('Iron, Fe (mg)',              min_value=0.0, max_value=200.0,   value=get_preset_val('Iron, Fe'),                step=0.1)
        inputs['Niacin']                        = st.number_input('Niacin (mg)',                min_value=0.0, max_value=200.0,   value=get_preset_val('Niacin'),                  step=0.1)
        inputs['Potassium, K']                  = st.number_input('Potassium, K (mg)',          min_value=0.0, max_value=20000.0, value=get_preset_val('Potassium, K'),            step=1.0)

    with c2:
        st.markdown("**💊 Vitamins & Others**")
        inputs['Ash']                           = st.number_input('Ash (g)',                    min_value=0.0, max_value=100.0,   value=get_preset_val('Ash'),                     step=0.1)
        inputs['Sodium, Na']                    = st.number_input('Sodium, Na (mg)',            min_value=0.0, max_value=40000.0, value=get_preset_val('Sodium, Na'),              step=1.0)
        inputs['Vitamin E (alpha-tocopherol)']  = st.number_input('Vitamin E (mg)',             min_value=0.0, max_value=200.0,   value=get_preset_val('Vitamin E (alpha-tocopherol)'), step=0.1)
        inputs['Manganese, Mn']                 = st.number_input('Manganese, Mn (mg)',         min_value=0.0, max_value=300.0,   value=get_preset_val('Manganese, Mn'),           step=0.01)
        inputs['Choline, total']                = st.number_input('Choline, total (mg)',        min_value=0.0, max_value=3000.0,  value=get_preset_val('Choline, total'),          step=0.1)
        inputs['Theobromine']                   = st.number_input('Theobromine (mg)',           min_value=0.0, max_value=3000.0,  value=get_preset_val('Theobromine'),             step=1.0)
        inputs['Vitamin B-12']                  = st.number_input('Vitamin B-12 (µg)',          min_value=0.0, max_value=100.0,   value=get_preset_val('Vitamin B-12'),            step=0.01)
        inputs['Vitamin K (phylloquinone)']     = st.number_input('Vitamin K (µg)',             min_value=0.0, max_value=2000.0,  value=get_preset_val('Vitamin K (phylloquinone)'), step=0.1)
        inputs['Fiber, total dietary']          = st.number_input('Fiber, total dietary (g)',   min_value=0.0, max_value=100.0,   value=get_preset_val('Fiber, total dietary'),    step=0.1)
        inputs['Vitamin D (D2 + D3)']           = st.number_input('Vitamin D (µg)',             min_value=0.0, max_value=300.0,   value=get_preset_val('Vitamin D (D2 + D3)'),     step=0.1)

    st.markdown('<p class="section-header">🏷️ Food Category</p>', unsafe_allow_html=True)

    preset_cat = presets[selected_preset]['category'] if selected_preset else 'Other'
    cat_index = all_categories.index(preset_cat) if preset_cat in all_categories else len(all_categories) - 1
    selected_category = st.selectbox('Select the food category', options=all_categories, index=cat_index)

    for cat_feat in category_features:
        cat_name = cat_feat.replace('cat_', '')
        inputs[cat_feat] = 1.0 if cat_name == selected_category else 0.0

    st.markdown('')
    predict_btn = st.button('🔍 Predict Macronutrients', width='stretch')


with right_col:
    st.markdown('<p class="section-header">🎯 Prediction Results</p>', unsafe_allow_html=True)

    if predict_btn or selected_preset:
        input_df = pd.DataFrame([inputs])[selected_features]
        prediction = model.predict(input_df)[0]
        prediction = np.clip(prediction, 0, 100)

        protein, fat, sugar = prediction[0], prediction[1], prediction[2]

        # result cards
        st.markdown(f"""
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; margin-bottom:1.5rem;">
            <div class="result-card">
                <div style="font-size:2rem; margin-bottom:0.3rem;">🥩</div>
                <div class="result-value">{protein:.1f}</div>
                <div class="result-unit">g per 100g</div>
                <div class="result-label" style="margin-top:0.3rem;">Protein</div>
            </div>
            <div class="result-card">
                <div style="font-size:2rem; margin-bottom:0.3rem;">🧈</div>
                <div class="result-value">{fat:.1f}</div>
                <div class="result-unit">g per 100g</div>
                <div class="result-label" style="margin-top:0.3rem;">Fat</div>
            </div>
            <div class="result-card">
                <div style="font-size:2rem; margin-bottom:0.3rem;">🍬</div>
                <div class="result-value">{sugar:.1f}</div>
                <div class="result-unit">g per 100g</div>
                <div class="result-label" style="margin-top:0.3rem;">Sugar</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

      
        st.markdown('<p class="section-header">📊 Nutritional Profile</p>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        nutrients = ['Protein', 'Fat', 'Sugar']
        values = [protein, fat, sugar]
        colors = ['#2ecc71', '#e67e22', '#e74c3c']
        bars = ax.barh(nutrients, values, color=colors, height=0.5, edgecolor='none')
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}g', va='center', fontsize=11, fontweight='600', color='#1a1a2e')
        ax.set_xlim(0, max(values) * 1.25 + 5)
        ax.set_xlabel('g per 100g', fontsize=10, color='#888')
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='x', colors='#aaa')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_facecolor('#fafffe')
        fig.patch.set_facecolor('#fafffe')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

     
        st.markdown('<p class="section-header">📈 Summary</p>', unsafe_allow_html=True)

        total = protein + fat + sugar
        dominant = nutrients[np.argmax(values)]
        emoji = {'Protein':'🥩','Fat':'🧈','Sugar':'🍬'}

        m1, m2, m3 = st.columns(3)
        m1.metric("Total macros", f"{total:.1f}g")
        m2.metric("Dominant", f"{emoji[dominant]} {dominant}")
        m3.metric("Category", selected_category[:12] + "..." 
          if len(selected_category) > 12 
          else selected_category)

        # quality hint
        if protein > 15:
            st.markdown('<div class="info-box">💪 <strong>High protein food</strong> — good for muscle building and satiety.</div>', unsafe_allow_html=True)
        if sugar > 20:
            st.markdown('<div class="warning-box">⚠️ <strong>High sugar content</strong> — consume in moderation.</div>', unsafe_allow_html=True)
        if fat > 30:
            st.markdown('<div class="warning-box">⚠️ <strong>High fat content</strong> — check saturated fat intake.</div>', unsafe_allow_html=True)
        if protein > 10 and sugar < 5 and fat < 10:
            st.markdown('<div class="info-box">✅ <strong>Balanced profile</strong> — good nutritional composition.</div>', unsafe_allow_html=True)

       
        with st.expander("🔬 Show full input values used"):
            display_df = input_df[numeric_features].T.rename(columns={0: 'Value'})
            display_df['Value'] = display_df['Value'].round(2)
            st.dataframe(display_df, width='stretch')

    else:
        st.markdown("""
        <div style="text-align:center; padding:4rem 2rem; color:#aaa;">
            <div style="font-size:4rem;">🥗</div>
            <p style="font-size:1.1rem; margin-top:1rem; color:#888;">
                Enter nutritional values on the left<br>
                and click <strong style="color:#27ae60;">Predict Macronutrients</strong>
            </p>
            <p style="font-size:0.9rem; color:#aaa;">
                Or pick a preset from the sidebar →
            </p>
        </div>
        """, unsafe_allow_html=True)
