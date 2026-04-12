import streamlit as st

st.set_page_config(
    page_title="Nutrition Intelligence System",
    page_icon="🥗",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Playfair Display', serif; }

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.5rem;
    color: #1a1a2e;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}
.hero-subtitle {
    font-size: 1.15rem;
    color: #666;
    font-weight: 300;
    max-width: 600px;
    margin-bottom: 3rem;
}
.card {
    background: white;
    border-radius: 20px;
    padding: 2.5rem 2rem;
    border: 1px solid #eee;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    height: 100%;
}
.card:hover {
    border-color: #2ecc71;
    box-shadow: 0 8px 30px rgba(46,204,113,0.15);
    transform: translateY(-3px);
}
.card-icon { font-size: 3.5rem; margin-bottom: 1rem; }
.card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    color: #1a1a2e;
    margin-bottom: 0.5rem;
}
.card-desc { color: #888; font-size: 0.95rem; line-height: 1.6; }
.card-tag {
    display: inline-block;
    background: #f0fff4;
    color: #276749;
    border: 1px solid #c3e6cb;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.8rem;
    font-weight: 500;
    margin-top: 1rem;
}
.stat-box {
    background: #f8fffe;
    border-radius: 12px;
    padding: 1.2rem;
    border: 1px solid #c3e6cb;
    text-align: center;
}
.stat-value {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    color: #1a1a2e;
    font-weight: 700;
}
.stat-label {
    font-size: 0.8rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
</style>
""", unsafe_allow_html=True)

# Hero section
st.markdown('<p class="hero-title">🥗 Nutrition<br>Intelligence System</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">A machine learning platform for food nutrition analysis. Predict NutriScore grades and macronutrient content from nutritional data.</p>', unsafe_allow_html=True)

st.divider()

# Clickable cards
st.markdown("### 🚀 Choose a Tool")
st.markdown("")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="card">
        <div class="card-icon">🏷️</div>
        <div class="card-title">NutriScore Decoder</div>
        <div class="card-desc">
            Enter a food product's nutritional values and our XGBoost model 
            predicts its NutriScore grade from A (best) to E (worst).<br><br>
            Includes confidence scores and SHAP explanations showing 
            why the model assigned that grade.
        </div>
        
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")
    if st.button("🏷️ Open NutriScore Decoder", use_container_width=True, key="btn_nutri"):
        st.switch_page("pages/NutriScore.py")

with col2:
    st.markdown("""
    <div class="card">
        <div class="card-icon">🔬</div>
        <div class="card-title">Macro Predictor</div>
        <div class="card-desc">
            Enter micronutrient values of any USDA food and our 
            Gradient Boosting model predicts its Protein, Fat and Sugar 
            content per 100g.<br><br>
            Useful for understanding nutrient relationships 
            in raw and unlabeled food ingredients.
        </div>
        
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")
    if st.button("🔬 Open Macro Predictor", use_container_width=True, key="btn_macro"):
        st.switch_page("pages/Macro_Predictor.py")

st.divider()
