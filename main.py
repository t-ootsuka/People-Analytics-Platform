import streamlit as st
from eda_app import eda_app
from analytics_app import analytics_app
from timeline_app import timeline_app
from text_analytics_app import text_analytics_app
from advanced_analytics import advanced_analytics_app
from causal_discovery import causal_discovery_app

# 1. Page Config (Title bar of the browser)
st.set_page_config(
    page_title="People Logic | Platform",
    page_icon="🧬",
    layout="wide"
)

# 2. UI Styling (Navy/White Minimalist)
st.markdown("""
    <style>
    .stApp { background-color: #ffffff !important; color: #000000 !important; }
    h1, h2, h3, h4, h5, h6 { color: #1a2a6c !important; font-weight: 700 !important; border-bottom: 1px solid #f0f0f0; }
    [data-testid="stSidebar"] { background-color: #f8f9fa !important; border-right: 1px solid #dddddd !important; }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: #1a2a6c !important; font-weight: bold !important; }
    .stButton>button {
        background-color: #ffffff !important;
        color: #1a2a6c !important;
        border: 2px solid #1a2a6c !important;
        border-radius: 4px !important;
        font-weight: bold !important;
        width: 100% !important;
    }
    .stButton>button:hover { background-color: #1a2a6c !important; color: #ffffff !important; }
    span[data-baseweb="tag"] { background-color: #1a2a6c !important; border-radius: 4px !important; }
    span[data-baseweb="tag"] span { color: #ffffff !important; font-weight: bold !important; }
    div[data-baseweb="select"] > div { border: 1px solid #1a2a6c !important; }
    </style>
    """, unsafe_allow_html=True)

def main():
    # 💡 Sidebar Branding (No AI, Pure People Logic)
    st.sidebar.title("🧬 PEOPLE LOGIC")
    st.sidebar.caption("People Analytics Platform")
    st.sidebar.markdown("---")
    
    # 💡 辞書形式にすることで、メニュー名を変えても壊れない設計にしました
    menu_map = {
        "🔍 EDA": eda_app,
        "📊 Drivers": analytics_app,
        "⏳ Trends": timeline_app,
        "✍️ Voices": text_analytics_app,
        "⚡ Intelligence": advanced_analytics_app,
        "🕸️ Network": causal_discovery_app
    }
    
    # 選択肢を表示（辞書のキーを取得）
    choice = st.sidebar.radio("STRATEGIC MENU", list(menu_map.keys()))
    
    st.sidebar.markdown("---")
    
    # 💡 System Status (Operational)
    st.sidebar.markdown(
        '<div style="background-color: #f8f9fa; border-left: 5px solid #1a2a6c; padding: 12px; color: #1a2a6c;">'
        '<span style="font-size: 0.75rem; font-weight: bold;">SYSTEM OPERATIONAL</span><br>'
        '<span style="font-size: 0.65rem; color: #666;">Core Engine: Hybrid Causal Logic</span>'
        '</div>', 
        unsafe_allow_html=True
    )

    # 💡 ここで選択された関数を実行します（if文を並べる必要がなくなります）
    menu_map[choice]()

if __name__ == "__main__":
    main()