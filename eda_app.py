import streamlit as st
import pandas as pd
import numpy as np
import chardet
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib

def eda_app():
    st.markdown("## 🔍 Insight-Bridge: Strategic EDA")
    
    # --- 1. CSVアップロード ---
    uploaded_file = st.file_uploader("📂 データソースを選択", type=["csv"], key="eda_file_uploader")
    if not uploaded_file:
        st.info("💡 分析を開始するにはCSVファイルをアップロードしてください。")
        return

    # 文字コード判定
    rawdata = uploaded_file.read()
    result = chardet.detect(rawdata)
    encoding = result["encoding"]
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file, encoding=encoding)
    df.columns = df.columns.str.strip()

    # --- 2. 特徴量生成（年齢・勤続カテゴリ） ---
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="ignore")

    age_cols = [c for c in df.columns if "年齢" in c or "年令" in c]
    if age_cols:
        col_name = age_cols[0]
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        df["年齢カテゴリ"] = pd.cut(
            df[col_name], 
            bins=[0, 29, 39, 49, 59, np.inf], 
            labels=["20代以下", "30代", "40代", "50代", "60代以上"]
        )
    
    tenure_cols = [c for c in df.columns if "勤続" in c]
    if tenure_cols:
        col_name = tenure_cols[0]
        val = pd.to_numeric(df[col_name], errors='coerce')
        years = val if val.max() < 100 else val / 12
        df["勤続年数(計算)"] = years.round(1)
        df["勤続カテゴリ"] = pd.cut(
            years, 
            bins=[0, 1, 3, 5, np.inf], 
            labels=["1年未満", "1〜3年", "3〜5年", "5年以上"]
        )

    # --- 3. 分析設定（サイドバー） ---
    st.sidebar.header("🛠 分析設定")
    all_available_cols = df.columns.tolist()
    selected_cols = st.sidebar.multiselect("分析対象の項目", all_available_cols, default=all_available_cols)
    df = df[selected_cols]
    
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cols_to_encode = st.sidebar.multiselect("ダミー変数化（ワンホット）する項目", options=categorical_cols)

    if st.sidebar.button("🚀 加工データをAnalyticsへ送る"):
        df_encoded = pd.get_dummies(df, columns=cols_to_encode, drop_first=True) if cols_to_encode else df.copy()
        st.session_state.df_encoded = df_encoded
        st.sidebar.success("✅ Analyticsへデータを同期しました")

    with st.expander("👀 データプレビュー（加工後）"):
        st.dataframe(df.head(), use_container_width=True)

    # --- 4. メインコンテンツ（タブ形式） ---
    tab_stat, tab_viz, tab_corr = st.tabs(["📊 統計量", "📈 可視化", "💡 相関分析"])

    with tab_stat:
        st.subheader("グループ別統計")
        c1, c2 = st.columns(2)
        with c1: g_cat1 = st.selectbox("グループ1 (タブ)", ["なし"] + categorical_cols, key="g1")
        with c2: g_cat2 = st.selectbox("グループ2 (比較)", ["なし"] + categorical_cols, key="g2")

        if g_cat1 != "なし":
            vals = sorted(df[g_cat1].dropna().unique().tolist())
            sub_tabs = st.tabs([str(v) for v in vals])
            for i, v in enumerate(vals):
                with sub_tabs[i]:
                    sub = df[df[g_cat1] == v]
                    res = sub.groupby(g_cat2)[num_cols].mean().round(2) if g_cat2 != "なし" else sub[num_cols].describe().T.round(2)
                    st.dataframe(res, use_container_width=True)
        else:
            st.dataframe(df[num_cols].describe().T.round(2), use_container_width=True)

    with tab_viz:
        st.subheader("分布・比較分析")
        v_col1, v_col2 = st.columns([1, 2])
        with v_col1:
            viz_type = st.radio("グラフ種類", ["ヒストグラム", "箱ひげ図", "バイオリン"])
            v_num = st.selectbox("数値軸（縦軸）", num_cols)
        with v_col2:
            v_cat1 = st.selectbox("カテゴリ1 (タブ分割)", ["なし"] + categorical_cols)
            v_cat2 = st.selectbox("カテゴリ2 (比較軸/色分け)", ["なし"] + categorical_cols)

        if v_cat1 != "なし":
            v_vals = sorted(df[v_cat1].dropna().unique().tolist())
            v_sub_tabs = st.tabs([str(v) for v in v_vals])
            for j, vv in enumerate(v_vals):
                with v_sub_tabs[j]:
                    v_sub = df[df[v_cat1] == vv]
                    color = v_cat2 if v_cat2 != "なし" else None
                    if viz_type == "ヒストグラム": fig = px.histogram(v_sub, x=v_num, color=color, barmode="overlay")
                    elif viz_type == "箱ひげ図": fig = px.box(v_sub, x=v_cat2 if v_cat2!="なし" else None, y=v_num, color=color)
                    else: fig = px.violin(v_sub, x=v_cat2 if v_cat2!="なし" else None, y=v_num, color=color, box=True)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            color = v_cat2 if v_cat2 != "なし" else None
            if viz_type == "ヒストグラム": fig = px.histogram(df, x=v_num, color=color)
            elif viz_type == "箱ひげ図": fig = px.box(df, x=v_cat2 if v_cat2!="なし" else None, y=v_num, color=color)
            else: fig = px.violin(df, x=v_cat2 if v_cat2!="なし" else None, y=v_num, color=color, box=True)
            st.plotly_chart(fig, use_container_width=True)

    with tab_corr:
        st.subheader("相関マトリクス")
        corr_cat = st.selectbox("フィルタリング項目 (タブ表示)", ["全体"] + categorical_cols)
        
        def plot_sns(data, title):
            if len(num_cols) < 2:
                st.info("相関を表示するには2つ以上の数値列が必要です。")
                return
            
            corr = data[num_cols].corr()
            num_vars = len(num_cols)
            
            # --- 修正ポイント：1項目あたり1インチを確保し、図を巨大化させる ---
            dynamic_size = max(10, num_vars * 1.0) 
            fig, ax = plt.subplots(figsize=(dynamic_size, dynamic_size * 0.75))
            
            # --- 修正ポイント：数字をハッキリ見えるサイズ(10)で固定し、太字にする ---
            sns.heatmap(
                corr, 
                annot=True, 
                cmap="coolwarm", 
                fmt=".2f", 
                ax=ax,
                annot_kws={"size": 10, "weight": "bold"},
                cbar_kws={"shrink": .7}
            )
            
            plt.xticks(rotation=45, ha="right", fontsize=11)
            plt.yticks(rotation=0, fontsize=11)
            plt.title(title, fontsize=16, pad=20)
            
            # グラフを表示（巨大な図をそのまま出すためにコンテナ幅固定を解除）
            st.pyplot(fig)

        if corr_cat == "全体":
            plot_sns(df, "全体相関")
        else:
            c_vals = sorted(df[corr_cat].dropna().unique().tolist())
            c_sub_tabs = st.tabs([str(v) for v in c_vals])
            for k, cv in enumerate(c_vals):
                with c_sub_tabs[k]:
                    plot_sns(df[df[corr_cat] == cv], f"{cv} の相関")


