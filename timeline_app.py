import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

def timeline_app():
    st.markdown("# ⏳ Insight-Timeline: Strategic Trend Analysis")
    st.caption("現在と過去のデータを比較し、組織の変化を可視化します")

    # 1. 比較用データのアップロード
    with st.expander("📂 比較データのアップロード", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            curr_file = st.file_uploader("今回のデータ (現在)", type=['csv', 'xlsx'], key="curr_up")
        with col2:
            prev_file = st.file_uploader("前回のデータ (過去)", type=['csv', 'xlsx'], key="prev_up")

    if curr_file and prev_file:
        try:
            # データの読み込み
            df_curr_raw = pd.read_csv(curr_file) if curr_file.name.endswith('.csv') else pd.read_excel(curr_file)
            df_prev_raw = pd.read_csv(prev_file) if prev_file.name.endswith('.csv') else pd.read_excel(prev_file)

            # カラム名クリーンアップ
            def clean_cols(df):
                new_cols = [re.sub(r'[ \t\n\r\f\v（）() -]', '_', str(col).strip()) for col in df.columns]
                return df.rename(columns=dict(zip(df.columns, new_cols)))

            df_curr = clean_cols(df_curr_raw)
            df_prev = clean_cols(df_prev_raw)
            inv_map = dict(zip(df_curr.columns, df_curr_raw.columns))

            # --- 【重要】自動カテゴリ化ロジック (EDAの機能をTimelineにも実装) ---
            def auto_binning(df):
                # 年齢の階層化
                age_col = next((c for c in df.columns if '年齢' in inv_map.get(c, c)), None)
                if age_col and pd.api.types.is_numeric_dtype(df[age_col]):
                    bins = [0, 20, 30, 40, 50, 60, 100]
                    labels = ['20歳未満', '20代', '30代', '40代', '50代', '60歳以上']
                    df['年齢層'] = pd.cut(df[age_col], bins=bins, labels=labels, right=False)
                
                # 勤続年数の階層化
                ten_col = next((c for c in df.columns if '勤続' in inv_map.get(c, c)), None)
                if ten_col and pd.api.types.is_numeric_dtype(df[ten_col]):
                    val = df[ten_col]
                    if val.max() > 50: val = val / 12 # 月数なら年数に変換
                    bins = [0, 1, 3, 5, 10, 100]
                    labels = ['1年未満', '1-3年', '3-5年', '5-10年', '10年以上']
                    df['勤続年数層'] = pd.cut(val, bins=bins, labels=labels, right=False)
                return df

            df_curr = auto_binning(df_curr)
            df_prev = auto_binning(df_prev)

            # 数値列と属性列の再抽出
            num_cols_curr = df_curr.select_dtypes(include=np.number).columns.tolist()
            common_nums = [c for c in num_cols_curr if c in df_prev.columns]
            # 新しく作った「層」も含めてカテゴリ列を抽出
            cat_cols = [c for c in df_curr.columns if df_curr[c].dtype == 'object' or df_curr[c].dtype.name == 'category']

            # --- サイドバー設定 ---
            st.sidebar.header("⏳ 時系列比較設定")
            
            selected_features = st.sidebar.multiselect(
                "比較する項目を選択", 
                options=common_nums,
                default=[f for f in common_nums if not any(k in f for k in ["アカウント", "ID", "コード"])][:15],
                format_func=lambda x: inv_map.get(x, x)
            )

            filter_cat = st.sidebar.selectbox(
                "属性で絞り込む", 
                ["全体"] + cat_cols,
                format_func=lambda x: inv_map.get(x, x) if x in inv_map else x # 層などの新設カラムはそのまま表示
            )

            if not selected_features:
                st.info("💡 比較したい項目をサイドバーから選んでください。")
                return

            # --- フィルタリング ---
            if filter_cat == "全体":
                f_curr, f_prev = df_curr, df_prev
                label = "全体"
            else:
                vals = sorted(df_curr[filter_cat].dropna().unique().tolist())
                selected_val = st.sidebar.selectbox(f"{filter_cat}を選択", vals)
                f_curr = df_curr[df_curr[filter_cat] == selected_val]
                f_prev = df_prev[df_prev[filter_cat] == selected_val]
                label = f"{filter_cat}: {selected_val}"

            if f_curr.empty or f_prev.empty:
                st.warning(f"⚠️ {label} の比較データが不足しています。")
                return

            # --- 計算と描画 ---
            curr_means = f_curr[selected_features].mean()
            prev_means = f_prev[selected_features].mean()
            diff_series = curr_means - prev_means
            
            st.subheader(f"📊 前回比スコア変化: {label}")
            
            plot_df = pd.DataFrame({
                '項目': [inv_map.get(c, c) for c in diff_series.index],
                '変化幅': diff_series.values
            }).sort_values('変化幅')

            fig = px.bar(
                plot_df, x='変化幅', y='項目', orientation='h',
                color='変化幅', color_continuous_scale='RdBu',
                height=max(400, len(selected_features) * 35)
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📝 詳細テーブル"):
                table = pd.DataFrame({'今回平均': curr_means, '前回平均': prev_means, '差分': diff_series})
                table.index = [inv_map.get(c, c) for c in table.index]
                st.table(table.style.format("{:.2f}").background_gradient(cmap='RdBu', subset=['差分']))

        except Exception as e:
            st.error(f"解析中にエラーが発生しました: {e}")
    else:
        st.info("💡 「今回」と「前回」の2つのファイルをアップロードして開始してください。")