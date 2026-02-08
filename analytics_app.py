import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import matplotlib.pyplot as plt
import japanize_matplotlib
import re

def analytics_app():
    st.markdown("# 📊 Influence Analysis")
    st.caption("Quantitative driver analysis to identify key organizational levers.")

    if 'df_encoded' not in st.session_state:
        st.warning("⚠️ No data available. Please complete 'Landscape' (EDA) stage first.")
        return

    # --- Data Setup (仕様維持) ---
    df = st.session_state.df_encoded.copy()
    original_cols = df.columns.tolist()
    clean_cols = [re.sub(r'[ \t\n\r\f\v（）() -]', '_', col) for col in original_cols]
    df.columns = clean_cols
    col_mapping = dict(zip(clean_cols, original_cols))

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in df.columns if df[c].dtype == 'object' or df[c].dtype.name == 'category']

    # --- Sidebar ---
    st.sidebar.header("🎯 Model Settings")
    method = st.sidebar.radio("Analysis Engine", ["Linear Regression (OLS)", "Decision Tree Map"])
    
    target = st.sidebar.selectbox("Outcome Variable", num_cols)
    remaining_nums = [c for c in num_cols if c != target]
    features = st.sidebar.multiselect("Drivers (Features)", remaining_nums, default=remaining_nums[:10])
    
    st.subheader("🛠 Analysis Scope")
    filter_cat = st.selectbox("Compare by attribute", ["Global"] + cat_cols)
    st.markdown("---")

    if not features:
        st.info("💡 Please select Drivers from the sidebar to begin analysis.")
        return

    if filter_cat == "Global":
        display_df = df
        label = "Global"
    else:
        unique_vals = sorted(df[filter_cat].dropna().unique().tolist())
        f_val = st.selectbox(f"Select {filter_cat}", unique_vals)
        display_df = df[df[filter_cat] == f_val]
        label = f"{filter_cat}: {f_val}"

    if method == "Linear Regression (OLS)":
        run_regression_with_simulation(display_df, target, features, label, col_mapping)
    else:
        max_depth = st.sidebar.slider("Tree Depth", 2, 5, 3) 
        run_decision_tree_logic(display_df, target, features, label, col_mapping, max_depth)

# --- 1. OLS Regression (詳細サマリー復活版) ---
def run_regression_with_simulation(data, target, features, label, col_mapping):
    if len(data) < len(features) + 1:
        st.error(f"❌ {label}: Insufficient data points for regression.")
        return
    try:
        X = sm.add_constant(data[features])
        y = data[target]
        model = sm.OLS(y, X).fit()
        
        m1, m2, m3 = st.columns(3)
        m1.metric("R-Squared", f"{model.rsquared:.3f}")
        m2.metric("Sample Size", f"{len(data)}")
        sig_label = "信頼性あり" if model.f_pvalue < 0.05 else "参考程度"
        m3.metric("Model Sig.", sig_label)

        coef_df = pd.DataFrame({
            'Factor': [col_mapping.get(c, c) for c in model.params.index],
            'Impact': model.params.values,
            'p_value': model.pvalues.values
        }).query("Factor != 'const'")

        coef_df['Confidence'] = coef_df['p_value'].apply(lambda x: '信頼性あり' if x < 0.05 else '参考程度')
        coef_df = coef_df.sort_values('Impact', ascending=True)

        fig = px.bar(
            coef_df, x='Impact', y='Factor', orientation='h', color='Confidence',
            color_discrete_map={'信頼性あり': '#1a2a6c', '参考程度': '#d3d3d3'},
            title=f"{label}: Impact Ranking", height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # 💡 ここを復活させました：統計詳細レポート
        with st.expander("📋 Detailed Statistical Report"):
            st.text(model.summary())

        st.markdown("#### 📈 Sensitivity Simulation")
        sim_impact = 0
        s_cols = st.columns(2)
        for i, feat in enumerate(features):
            with s_cols[i % 2]:
                boost = st.slider(f"Boost {col_mapping.get(feat, feat)} by", 0.0, 1.0, 0.0, 0.1, key=f"reg_{label}_{feat}")
                sim_impact += boost * model.params[feat]
        st.success(f"💡 Improvement: **+{sim_impact:.2f}**")

    except Exception as e:
        st.error(f"Execution Error: {e}")

# --- 2. Decision Tree + Summary Table ---
def run_decision_tree_logic(data, target, features, label, col_mapping, max_depth):
    if len(data) < 5:
        st.error(f"❌ {label}: Insufficient data.")
        return
    try:
        X = data[features]
        y = data[target]
        model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        model.fit(X, y)

        st.write(f"### 📊 {label}: Key Drivers & Branch Logic")
        
        imp_df = pd.DataFrame({
            'Factor': [col_mapping.get(c, c) for c in features],
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        st.plotly_chart(px.bar(imp_df[imp_df['Importance'] > 0], x='Importance', y='Factor', orientation='h', 
                               height=400, color_discrete_sequence=['#1a2a6c']), use_container_width=True)

        st.markdown("---")
        st.markdown("#### Segmentation Structure")
        fig_tree, ax = plt.subplots(figsize=(16, 8)) 
        tree.plot_tree(model, feature_names=[col_mapping.get(c, c) for c in features], 
                       filled=True, rounded=True, fontsize=10, ax=ax, precision=2)
        st.pyplot(fig_tree)

        st.markdown("---")
        st.markdown("#### 📋 Node Analysis Summary")
        st.caption("予測スコアが高い順にセグメントを整理しています。")
        
        from sklearn.tree import _tree
        def get_rules(tree, feature_names):
            tree_ = tree.tree_
            feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]
            rules = []
            def recurse(node, depth, current_rule):
                if tree_.feature[node] != _tree.TREE_UNDEFINED:
                    name = feature_name[node]
                    threshold = tree_.threshold[node]
                    recurse(tree_.children_left[node], depth + 1, current_rule + [f"{name} <= {threshold:.2f}"])
                    recurse(tree_.children_right[node], depth + 1, current_rule + [f"{name} > {threshold:.2f}"])
                else:
                    rules.append({
                        "予測スコア": round(tree_.value[node][0][0], 2),
                        "対象人数": int(tree_.n_node_samples[node]),
                        "分岐ルール": " ➔ ".join(current_rule)
                    })
            recurse(0, 1, [])
            return rules

        rules_list = get_rules(model, [col_mapping.get(c, c) for c in features])
        rules_df = pd.DataFrame(rules_list).sort_values("予測スコア", ascending=False)
        rules_df.index = np.arange(1, len(rules_df) + 1)
        rules_df.index.name = "Rank"
        
        st.table(rules_df) 

    except Exception as e:
        st.error(f"Analysis Error: {e}")








