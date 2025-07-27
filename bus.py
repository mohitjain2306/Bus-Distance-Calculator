import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, .stApp, .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        min-height: 100vh !important;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        background-attachment: fixed !important;
    }
    
    .main {
        padding: 0 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        min-height: 100vh !important;
        background-attachment: fixed !important;
    }
    
    .main .block-container {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    .main-header {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0 2rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .main-title {
        font-family: 'Inter', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        color: white;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .main-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.9);
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2);
    }
    
    .input-section {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .section-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .section-header-white {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: white;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    
    .result-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(79, 172, 254, 0.3);
        transition: all 0.3s ease;
    }
    
    .result-card:hover {
        transform: scale(1.02);
        box-shadow: 0 12px 35px rgba(79, 172, 254, 0.4);
    }
    
    .result-title {
        font-size: 1.2rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
        opacity: 0.9;
    }
    
    .result-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border-left: 4px solid;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateX(5px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
    }
    
    .metric-excellent { border-left-color: #27ae60; }
    .metric-good { border-left-color: #f39c12; }
    .metric-average { border-left-color: #e67e22; }
    .metric-poor { border-left-color: #e74c3c; }
    
    .metric-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #ffffff;
    }
    
    .metric-value {
        font-size: 0.95rem;
        color: #7f8c8d;
        margin: 0.2rem 0;
    }
    
    .stDataFrame {
        background: white !important;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    
    .stDataFrame > div {
        background: white !important;
    }
    
    .progress-container {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        height: 8px;
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease-in-out;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e1e8ed;
        transition: all 0.3s ease;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .stSelectbox > div > div > select {
        border-radius: 10px;
        border: 2px solid #e1e8ed;
    }
    
    .fade-in {
        animation: fadeIn 0.8s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .slide-in-left {
        animation: slideInLeft 0.8s ease-out;
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .slide-in-right {
        animation: slideInRight 0.8s ease-out;
    }
    
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .alert-success {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        color: #ffffff;
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        border: none;
        box-shadow: 0 4px 20px rgba(132, 250, 176, 0.3);
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #ffffff;
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        border: none;
        box-shadow: 0 4px 20px rgba(252, 182, 159, 0.3);
    }
    
    .alert-info {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #ffffff;
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        border: none;
        box-shadow: 0 4px 20px rgba(168, 237, 234, 0.3);
    }
    
    .training-section {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
    }
    
    .stApp > div:first-child {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .element-container:empty {
        display: none !important;
    }
    
    .stMarkdown:empty {
        display: none !important;
    }
    
    div[data-testid="element-container"]:empty {
        display: none !important;
    }
    
    .stContainer:empty {
        display: none !important;
    }
    
    div:empty {
        display: none !important;
    }
    
    .stContainer {
        background: transparent !important;
    }
    
    .stColumn {
        background: transparent !important;
    }
    
    .stColumn > div {
        background: transparent !important;
    }
    
    .main > div {
        background: transparent !important;
    }
    
    .block-container {
        background: transparent !important;
        padding-top: 1rem !important;
    }
    
    div[data-testid="stVerticalBlock"] {
        background: transparent !important;
    }
    
    div[data-testid="stHorizontalBlock"] {
        background: transparent !important;
    }
    
    .streamlit-container {
        background: transparent !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: white;
    }
    
    .stNumberInput label, .stSelectbox label {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    .stMarkdown p {
        color: #ffffff;
        font-weight: 600;
    }
    
    .glass-card .stMarkdown p {
        color: white !important;
    }
    
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }
        
        .glass-card {
            padding: 1rem;
        }
        
        .input-section {
            padding: 1rem;
        }
    }
    
    .stSuccess {
        background: rgba(255, 255, 255, 0.15) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 15px !important;
        backdrop-filter: blur(20px) !important;
    }

    .stSuccess * {
        color: white !important;
    }

    div[data-testid="stAlert"] {
        background: rgba(255, 255, 255, 0.15) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 15px !important;
        backdrop-filter: blur(20px) !important;
    }

    div[data-testid="stAlert"] * {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = Path('results/best_model.pkl')
    if model_path.exists():
        return joblib.load(model_path)
    else:
        st.warning("🤔 Hmm, I can't find a pre-trained model. Let's train one together first!")
        return None

def train_model(data):
    try:
        required_cols = ['Fuel_Level_Percentage', 'Vehicle_Load_kg', 
                        'Speed_kmph', 'Temperature_C', 'Route_Type', 'distance']
        if not all(col in data.columns for col in required_cols):
            missing = [col for col in required_cols if col not in data.columns]
            return None, f"Oops! We're missing some important data: {', '.join(missing)}"
        
        route_mapping = {'Highway': 0, 'Urban': 1, 'Rural': 2}
        data['Route_Type'] = data['Route_Type'].map(route_mapping)
        
        X = data[['Fuel_Level_Percentage', 'Vehicle_Load_kg', 
                 'Speed_kmph', 'Temperature_C', 'Route_Type']]
        y = data['distance']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        model_path = Path('results/best_model.pkl')
        model_path.parent.mkdir(exist_ok=True)
        joblib.dump({
            'model': model,
            'route_mapping': route_mapping
        }, model_path)
        
        return model, f"🎉 Great success! Your AI is now trained and ready! Accuracy score: {model.score(X_test, y_test):.2f}"
    except Exception as e:
        return None, f"Oops, something went wrong during training: {str(e)}"

def get_efficiency_category(efficiency):
    if efficiency > 2.0:
        return "🌟 Excellent! You're a fuel-saving superstar!", "🟢", "metric-excellent", "#27ae60"
    elif efficiency > 1.5:
        return "👍 Pretty good! You're doing well!", "🟡", "metric-good", "#f39c12"
    elif efficiency > 1.0:
        return "🤔 Average performance - room for improvement", "🟠", "metric-average", "#e67e22"
    else:
        return "😬 Let's work on this together", "🔴", "metric-poor", "#e74c3c"

def get_load_category(load_per_km):
    if load_per_km < 10:
        return "🪶 Light as a feather - efficient load!", "🟢", "metric-excellent", "#27ae60"
    elif load_per_km < 20:
        return "⚖️ Well-balanced load", "🟡", "metric-good", "#f39c12"
    else:
        return "🏋️ Heavy load - consider optimization", "🔴", "metric-poor", "#e74c3c"

def get_temp_impact(temp):
    if 15 <= temp <= 25:
        return "🌡️ Perfect weather for efficiency!", "🟢", "metric-excellent", "#27ae60"
    elif 10 <= temp < 15 or 25 < temp <= 30:
        return "🌤️ Temperature is okay, slight impact", "🟡", "metric-good", "#f39c12"
    else:
        return "🌡️ Extreme temperature - efficiency affected", "🔴", "metric-poor", "#e74c3c"

def get_route_impact(route_type):
    if route_type == "Highway":
        return "🛣️ Smooth sailing - highways are efficient!", "🟢", "metric-excellent", "#27ae60"
    elif route_type == "Urban":
        return "🏙️ City driving - lots of stop-and-go", "🟡", "metric-good", "#f39c12"
    elif route_type == "Rural":
        return "🌄 Country roads - hills and curves ahead", "🟠", "metric-average", "#e67e22"
    else:
        return "❓ Unknown route type", "❓", "metric-poor", "#95a5a6"

def get_speed_impact(speed):
    if 50 <= speed <= 80:
        return "🎯 Sweet spot! Perfect speed for efficiency", "🟢", "metric-excellent", "#27ae60"
    elif 30 <= speed < 50 or 80 < speed <= 100:
        return "⚡ Decent speed, could be optimized", "🟡", "metric-good", "#f39c12"
    else:
        return "🚀 Too fast or too slow for optimal efficiency", "🔴", "metric-poor", "#e74c3c"

def create_gauge_chart(value, title, max_value=100, color_scheme="blues"):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 16, 'family': 'Inter'}},
        delta = {'reference': max_value/2},
        gauge = {
            'axis': {'range': [None, max_value], 'tickfont': {'size': 12}},
            'bar': {'color': "rgba(79, 172, 254, 0.8)"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, max_value/3], 'color': 'rgba(231, 76, 60, 0.2)'},
                {'range': [max_value/3, 2*max_value/3], 'color': 'rgba(243, 156, 18, 0.2)'},
                {'range': [2*max_value/3, max_value], 'color': 'rgba(39, 174, 96, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.8
            }
        }
    ))
    
    fig.update_layout(
        font={'color': "#ffffff", 'family': 'Inter'},
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_metrics_visualization(metrics_data):
    fig = go.Figure()
    
    categories = list(metrics_data.keys())
    values = [metrics_data[cat]['score'] for cat in categories]
    colors = [metrics_data[cat]['color'] for cat in categories]
    
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f"{v:.1f}" for v in values],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Score: %{y:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Your Fleet Performance at a Glance",
        title_font={'size': 18, 'family': 'Inter', 'color': 'white'},
        xaxis_title="Performance Areas",
        yaxis_title="Your Score",
        font={'color': 'white', 'family': 'Inter'},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=20, r=20, t=60, b=40)
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', showgrid=True)
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', showgrid=True)
    
    return fig

def display_metric_card(title, value, description, category_class, icon, color):
    st.markdown(f"""
    <div class="metric-card {category_class} fade-in">
        <div class="metric-title">{icon} {title}</div>
        <div class="metric-value"><strong>{value}</strong></div>
        <div class="metric-value">{description}</div>
    </div>
    """, unsafe_allow_html=True)

def process_csv_data(csv_data, model_data):
    model = model_data['model']
    route_mapping = model_data['route_mapping']
    
    results = []
    
    for idx, row in csv_data.iterrows():
        try:
            route_mapped = route_mapping.get(str(row['Route_Type']).strip(), 0)
            
            input_data = pd.DataFrame({
                'Fuel_Level_Percentage': [float(row['Fuel_Level_Percentage'])],
                'Vehicle_Load_kg': [float(row['Vehicle_Load_kg'])],
                'Speed_kmph': [float(row['Speed_kmph'])],
                'Temperature_C': [float(row['Temperature_C'])],
                'Route_Type': [route_mapped]
            })
            
            distance = model.predict(input_data)[0]
            fuel_efficiency = distance / row['Fuel_Level_Percentage'] if row['Fuel_Level_Percentage'] > 0 else 0
            load_per_km = row['Vehicle_Load_kg'] / distance if distance > 0 else 0
            
            results.append({
                'Record': idx + 1,
                'Fuel_Level': float(row['Fuel_Level_Percentage']),
                'Vehicle_Load': float(row['Vehicle_Load_kg']),
                'Speed': float(row['Speed_kmph']),
                'Temperature': float(row['Temperature_C']),
                'Route_Type': str(row['Route_Type']),
                'Predicted_Distance': round(float(distance), 2),
                'Fuel_Efficiency': round(float(fuel_efficiency), 2),
                'Load_per_km': round(float(load_per_km), 2)
            })
            
        except Exception as e:
            st.error(f"❌ Had trouble with row {idx + 1}: {str(e)}")
            continue
    
    return pd.DataFrame(results)

def main():
    st.set_page_config(
        layout="wide", 
        page_title="🚌 Bus Efficiency Dashboard",
        page_icon="🚌",
        initial_sidebar_state="collapsed"
    )
    
    load_custom_css()
    
    hide_streamlit_style = """
    <style>
    .stApp, .stApp > div, .main, .block-container, html, body {
        background:linear-gradient(135deg, #667eea 0%, #764ba2 33%, #f093fb 66%, #f5576c 100%) !important;
        background-attachment: fixed !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    div[data-testid="stToolbar"] {visibility: hidden;}
    
    .stApp, .stApp > div, .main, .block-container {
        background-color: transparent !important;
    }
    
    div[class*="css-"]:empty {
        display: none !important;
    }
    
    div[class^="css-"]:not([class*="glass-card"]):not([class*="input-section"]):not([class*="result-card"]):not([class*="metric-card"]) {
        background-color: transparent !important;
        background-image: none !important;
    }
    
    .reportview-container .main .block-container {
        max-width: 100%;
        padding: 1rem;
        background: transparent !important;
    }
    
    .stVerticalBlock > div:empty {
        display: none !important;
    }
    
    .stHorizontalBlock > div:empty {
        display: none !important;
    }
    
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    [data-testid="stHeader"] {
        background: transparent !important;
    }
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-header fade-in">
        <h1 class="main-title">🚌 Bus Efficiency Dashboard</h1>
        <p class="main-subtitle">Machine Learning Model for Bus Distance Predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="training-section slide-in-left">', unsafe_allow_html=True)
    with st.expander("🧠 Train Your AI Assistant", expanded=False):
            st.markdown('<div class="section-header-white">📊 Upload Your Bus Data</div>', unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Drop your CSV file here - I'll learn from your historical data!", 
                type="csv",
                help="Upload your bus performance history and I'll become smarter at predicting efficiency!"
            )
            
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.markdown("**👀 Here's a peek at your data:**", unsafe_allow_html=True)
                        st.dataframe(data.head(), use_container_width=True)
                    
                    with col2:
                        st.markdown("**📊 Quick data summary:**", unsafe_allow_html=True)
                        st.markdown(f"**📋 Total Records:** {len(data)} (Great dataset!)", unsafe_allow_html=True)
                        st.markdown(f"**📁 Data Columns:** {len(data.columns)}", unsafe_allow_html=True)
                        missing_count = data.isnull().sum().sum()
                        if missing_count == 0:
                            st.markdown("**✅ Data Quality:** Perfect! No missing values", unsafe_allow_html=True)
                        else:
                            st.markdown(f"**⚠️ Missing Values:** {missing_count} (we can handle this!)", unsafe_allow_html=True)
                    
                    if st.button("🚀 Let's Train Your AI!", use_container_width=True):
                        with st.spinner("🤖 Teaching your AI about bus efficiency... This is exciting!"):
                            progress_bar = st.progress(0)
                            progress_messages = [
                                "📖 Reading your data...",
                                "🔍 Analyzing patterns...", 
                                "🧠 Learning from examples...",
                                "⚡ Optimizing predictions...",
                                "✨ Almost ready!"
                            ]
                            
                            for i in range(100):
                                if i % 20 == 0 and i < 100:
                                    st.info(progress_messages[i//20])
                                time.sleep(0.01)
                                progress_bar.progress(i + 1)
                            
                            model, message = train_model(data)
                            if model is not None:
                                st.success(f"🎉 {message}")
                                st.session_state.model = model
                                st.balloons()
                            else:
                                st.error(f"😅 {message}")
                except Exception as e:
                    st.error(f"🤔 Hmm, I had trouble reading your file: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if 'model' in st.session_state:
        model = st.session_state.model
    else:
        model = load_model()
    
    if model is None and 'model' not in st.session_state:
        st.markdown("""
        <div class="alert-warning">
            🤖 <strong>Your AI Assistant Needs Training!</strong><br>
            Don't worry - just upload your bus data above and I'll learn to help you optimize efficiency!
        </div>
        """, unsafe_allow_html=True)
        return
    
    col1, col2 = st.columns([1, 2], gap="large")
    
    st.markdown('<div style="clear: both; height: 0; overflow: hidden;"></div>', unsafe_allow_html=True)
    
    with col1:
        st.markdown('<div class="input-section slide-in-left">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🚌 Tell Me About Your Bus</div>', unsafe_allow_html=True)
        
        input_method = st.radio(
            "How would you like to input your data?",
            ["✋ Enter details manually", "📊 Upload a CSV file"],
            horizontal=True,
            help="Choose manual entry for one bus or CSV upload to analyze your entire fleet!"
        )
        
        if input_method == "✋ Enter details manually":
            with st.form("prediction_form"):
                st.markdown("**⛽ How much fuel do you have?**")
                fuel_level = st.number_input(
                    "Current fuel percentage (0-100%)", 
                    min_value=0.0, max_value=100.0, value=50.0, step=0.1, 
                    format="%.1f", label_visibility="collapsed",
                    help="Tell me your current fuel level - I'll factor this into efficiency calculations!"
                )
                
                st.markdown("**📦 What's your passenger/cargo load?**")
                bus_load = st.number_input(
                    "Total load in kilograms", 
                    min_value=0.0, max_value=5000.0, value=1000.0, step=0.1, 
                    format="%.1f", label_visibility="collapsed",
                    help="Heavier loads mean more fuel consumption - let's see how this affects your efficiency!"
                )
                
                st.markdown("**🚗 How fast are you planning to go?**")
                speed = st.number_input(
                    "Average speed in km/h", 
                    min_value=0.0, max_value=200.0, value=60.0, step=0.1, 
                    format="%.1f", label_visibility="collapsed",
                    help="Sweet spot is usually 50-80 km/h for optimal fuel efficiency!"
                )
                
                st.markdown("**🌡️ What's the weather like?**")
                temperature = st.number_input(
                    "Outside temperature in °C", 
                    min_value=-30.0, max_value=50.0, value=20.0, step=0.1, 
                    format="%.1f", label_visibility="collapsed",
                    help="Extreme temperatures affect engine efficiency - I'll account for this!"
                )
                
                st.markdown("**🛣️ What kind of roads will you take?**")
                route_type = st.selectbox(
                    "Select your primary route type", 
                    ["Highway", "Urban", "Rural"],
                    label_visibility="collapsed",
                    help="Highways are most efficient, urban routes have lots of stops, rural roads vary!"
                )
                
                submitted = st.form_submit_button("🎯 Calculate My Bus Efficiency!", use_container_width=True)
        
        else:
            st.markdown("**📊 Upload Your Fleet Data**")
            uploaded_csv = st.file_uploader(
                "Drop your CSV file here and I'll analyze your entire fleet!",
                type="csv",
                help="Your CSV should include: Fuel_Level_Percentage, Vehicle_Load_kg, Speed_kmph, Temperature_C, Route_Type",
                key="csv_uploader"
            )
            
            if uploaded_csv is not None:
                try:
                    csv_data = pd.read_csv(uploaded_csv)
                    
                    required_cols = ['Fuel_Level_Percentage', 'Vehicle_Load_kg', 'Speed_kmph', 'Temperature_C', 'Route_Type']
                    missing_cols = [col for col in required_cols if col not in csv_data.columns]
                    
                    if missing_cols:
                        st.error(f"😅 I need these columns to help you: {', '.join(missing_cols)}")
                        st.info("💡 Make sure your CSV has: Fuel_Level_Percentage, Vehicle_Load_kg, Speed_kmph, Temperature_C, Route_Type")
                        submitted = False
                    else:
                        st.success(f"🎉 Perfect! I found {len(csv_data)} bus records to analyze!")
                        
                        st.session_state.csv_data_raw = csv_data
                        
                        submitted = st.button("🚀 Analyze My Entire Fleet!", use_container_width=True, key="process_csv")
                        
                        if submitted:
                            st.session_state.csv_submitted = True
                
                except Exception as e:
                    st.error(f"🤔 I had trouble reading your CSV: {str(e)}")
                    submitted = False
            else:
                submitted = False
        
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if st.session_state.get('csv_submitted', False) and 'csv_data_raw' in st.session_state:
            try:
                csv_data = st.session_state.csv_data_raw
                
                if isinstance(model, dict):
                    model_data = model
                else:
                    model_data = joblib.load('results/best_model.pkl')
                
                st.markdown(f"""
                <div class="result-card slide-in-right">
                    <div class="result-title">🔄 Analyzing Your Fleet Data</div>
                    <div class="result-value">{len(csv_data)} Buses Being Analyzed</div>
                </div>
                """, unsafe_allow_html=True)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.info("🤖 Your AI is crunching the numbers...")
                
                results_df = process_csv_data(csv_data, model_data)
                
                progress_bar.progress(100)
                status_text.success("✅ Analysis Complete! Here are your insights...")
                
                st.markdown(f"""
                <div class="result-card slide-in-right">
                    <div class="result-title">🎊 Fleet Analysis Complete!</div>
                    <div class="result-value">{len(results_df)} Buses Analyzed Successfully</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="glass-card slide-in-right">', unsafe_allow_html=True)
                st.markdown('<div class="section-header-white">📈 Your Fleet Performance Summary</div>', unsafe_allow_html=True)
                
                if not results_df.empty:
                    col1_summary, col2_summary = st.columns(2)
                    
                    with col1_summary:
                        avg_distance = results_df['Predicted_Distance'].mean()
                        max_distance = results_df['Predicted_Distance'].max()
                        min_distance = results_df['Predicted_Distance'].min()
                        
                        st.markdown(f"""
                        <div class="metric-card metric-excellent fade-in">
                            <div class="metric-title">📏 Distance Capabilities</div>
                            <div class="metric-value"><strong>Average Range: {avg_distance:.1f} km</strong></div>
                            <div class="metric-value">Best Performer: {max_distance:.1f} km</div>
                            <div class="metric-value">Needs Attention: {min_distance:.1f} km</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2_summary:
                        avg_efficiency = results_df['Fuel_Efficiency'].mean()
                        good_efficiency = len(results_df[results_df['Fuel_Efficiency'] > 1.5])
                        poor_efficiency = len(results_df[results_df['Fuel_Efficiency'] <= 1.0])
                        
                        efficiency_message = "Great job!" if avg_efficiency > 1.5 else "Room for improvement" if avg_efficiency > 1.0 else "Let's optimize together!"
                        
                        st.markdown(f"""
                        <div class="metric-card metric-good fade-in">
                            <div class="metric-title">⛽ Fleet Efficiency Overview</div>
                            <div class="metric-value"><strong>Average: {avg_efficiency:.2f} ({efficiency_message})</strong></div>
                            <div class="metric-value">High Performers: {good_efficiency} buses</div>
                            <div class="metric-value">Optimization Opportunities: {poor_efficiency} buses</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.markdown("### 📥 Take Your Results With You!")
                    csv_download = results_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download Complete Fleet Analysis",
                        data=csv_download,
                        file_name=f"fleet_efficiency_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        type="primary",
                        help="Get a detailed CSV with all efficiency metrics for every bus in your fleet!"
                    )
                
                else:
                    st.error("😅 No results to show. Let's check your data format and try again!")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                if st.button("🔄 Analyze Different Data", use_container_width=True):
                    if 'csv_submitted' in st.session_state:
                        del st.session_state.csv_submitted
                    if 'csv_data_raw' in st.session_state:
                        del st.session_state.csv_data_raw
                    st.rerun()
                
            except Exception as e:
                st.markdown(f"""
                <div class="alert-warning">
                    😅 <strong>Oops! Something went wrong with the analysis</strong><br>
                    Let's check your data format and try again. I'm here to help!<br>
                    <small>Technical details: {str(e)}</small>
                </div>
                """, unsafe_allow_html=True)
        
        elif submitted and not st.session_state.get('csv_submitted', False):
            try:
                if isinstance(model, dict):
                    model_data = model
                else:
                    model_data = joblib.load('results/best_model.pkl')
                
                model_obj = model_data['model']
                route_mapping = model_data['route_mapping']
                
                input_data = pd.DataFrame({
                    'Fuel_Level_Percentage': [float(fuel_level)],
                    'Vehicle_Load_kg': [float(bus_load)],
                    'Speed_kmph': [float(speed)],
                    'Temperature_C': [float(temperature)],
                    'Route_Type': [route_mapping[route_type]]
                })
                
                distance = model_obj.predict(input_data)[0]
                
                fuel_efficiency = distance / fuel_level if fuel_level > 0 else 0
                load_per_km = bus_load / distance if distance > 0 else 0
                
                st.markdown(f"""
                <div class="result-card slide-in-right">
                    <div class="result-title">🎯 Your Bus Can Travel</div>
                    <div class="result-value">{distance:.1f} km</div>
                </div>
                """, unsafe_allow_html=True)
                
                gauge_col1, gauge_col2 = st.columns(2)
                
                with gauge_col1:
                    efficiency_gauge = create_gauge_chart(
                        fuel_efficiency, "Fuel Efficiency (km per % fuel)", 5
                    )
                    st.plotly_chart(efficiency_gauge, use_container_width=True)
                
                with gauge_col2:
                    load_gauge = create_gauge_chart(
                        min(load_per_km, 50), "Load Efficiency (kg per km)", 50
                    )
                    st.plotly_chart(load_gauge, use_container_width=True)
                
                st.markdown('<div class="glass-card slide-in-right">', unsafe_allow_html=True)
                st.markdown('<div class="section-header-white">🔍 Detailed Performance Analysis</div>', unsafe_allow_html=True)
                
                eff_title, eff_icon, eff_class, eff_color = get_efficiency_category(fuel_efficiency)
                load_title, load_icon, load_class, load_color = get_load_category(load_per_km)
                temp_title, temp_icon, temp_class, temp_color = get_temp_impact(temperature)
                route_title, route_icon, route_class, route_color = get_route_impact(route_type)
                speed_title, speed_icon, speed_class, speed_color = get_speed_impact(speed)
                
                metric_col1, metric_col2 = st.columns(2)
                
                with metric_col1:
                    display_metric_card(
                        "Fuel Efficiency", f"{fuel_efficiency:.2f} km per % fuel", 
                        eff_title, eff_class, eff_icon, eff_color
                    )
                    display_metric_card(
                        "Weather Impact", f"{temperature:.1f} °C", 
                        temp_title, temp_class, temp_icon, temp_color
                    )
                    display_metric_card(
                        "Speed Analysis", f"{speed:.1f} km/h", 
                        speed_title, speed_class, speed_icon, speed_color
                    )
                
                with metric_col2:
                    display_metric_card(
                        "Load Impact", f"{load_per_km:.2f} kg per km", 
                        load_title, load_class, load_icon, load_color
                    )
                    display_metric_card(
                        "Route Analysis", route_type, 
                        route_title, route_class, route_icon, route_color
                    )
                
                st.markdown("---")
                
                if "superstar" in eff_title.lower():
                    st.markdown("""
                    <div class="alert-success">
                        🌟 <strong>Absolutely Amazing!</strong><br>
                        You're operating at peak efficiency! Your bus is a fuel-saving champion. Keep up the fantastic work!
                    </div>
                    """, unsafe_allow_html=True)
                elif "doing well" in eff_title.lower():
                    st.markdown("""
                    <div class="alert-info">
                        👍 <strong>Great Job!</strong><br>
                        Your efficiency is solid! A few small tweaks to speed and route planning could make you even better.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="alert-warning">
                        💪 <strong>Let's Optimize Together!</strong><br>
                        There's room for improvement, and I'm here to help! Consider adjusting speed, optimizing routes, or managing loads better.
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f"""
                <div class="alert-warning">
                    😅 <strong>Oops! I hit a snag</strong><br>
                    Let's double-check those inputs and try again. I'm here to help you succeed!<br>
                    <small>What happened: {str(e)}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card slide-in-right">
                <div style="text-align: center; color: white; padding: 2rem;">
                    <h2 style="margin-bottom: 1rem;">🎯 Welcome to Your Smart Fleet Assistant!</h2>
                    <p style="font-size: 1.1rem; opacity: 0.9; margin-bottom: 2rem;">
                        I'm here to help you optimize your bus efficiency! Just fill in your details on the left, and I'll give you personalized insights.
                    </p>
                    <div style="display: flex; justify-content: space-around; margin: 2rem 0;">
                        <div style="text-align: center;">
                            <div style="font-size: 2rem; margin-bottom: 0.5rem;">⛽</div>
                            <div style="font-weight: 600;">Save Fuel Costs</div>
                            <div style="font-size: 0.9rem; opacity: 0.8;">Smart optimization tips</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
                            <div style="font-weight: 600;">Real-time Insights</div>
                            <div style="font-size: 0.9rem; opacity: 0.8;">Instant performance analysis</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎯</div>
                            <div style="font-weight: 600;">Actionable Advice</div>
                            <div style="font-size: 0.9rem; opacity: 0.8;">Practical recommendations</div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
           
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="margin-top: 3rem; text-align: center; color: rgba(255,255,255,0.7); padding: 2rem;">
        <hr style="border: 1px solid rgba(255,255,255,0.1); margin: 2rem 0;">
        <p style="margin: 0;">🚌 Bus Efficiency Assistant</p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Machine Learning Model for Bus Distance Predictions</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()