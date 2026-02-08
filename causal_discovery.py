import streamlit as st
import pandas as pd
import numpy as np
import lingam
import graphviz
import os

def causal_discovery_app():
    # 💡 Windows環境でのGraphviz実行エラー対策（これがないと動かない場合があるため追加）
    graphviz_bin = r'C:\Program Files\Graphviz\bin'
    if os.path.exists(graphviz_bin) and graphviz_bin not in os.environ["PATH"]:
        os.environ["PATH"] += os.pathsep + graphviz_bin

    st.markdown("# 🕸️ 因果探索: LiNGAMモード")
    st.caption("属性ごとに「原因 → 結果」の構造を分析し、組織の力学の違いを可視化します。")

    if 'df_encoded' not in st.session_state:
        st.warning("⚠️ 解析データが見つかりません。先に「EDA」画面でデータを準備してください。")
        return

    df = st.session_state.df_encoded.copy()
    
    # --- 1. 分析範囲の切り替え ---
    st.subheader("🛠 1. 分析範囲の選択")
    cat_cols = [c for c in df.columns if df[c].dtype == 'object' or df[c].dtype.name == 'category']
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        filter_cat = st.selectbox("属性別に分析を比較する", ["全体"] + cat_cols)
    if filter_cat == "全体":
        display_df = df
        label = "全体"
    else:
        unique_vals = sorted(df[filter_cat].dropna().unique().tolist())
        with col_c2:
            f_val = st.selectbox(f"{filter_cat}を選択", unique_vals)
        display_df = df[df[filter_cat] == f_val]
        label = f"{filter_cat}: {f_val}"

    st.info(f"📍 現在の分析対象: **{label}** (サンプル数: {len(display_df)}件)")

    # --- 2. 変数選択と詳細設定 ---
    num_df = display_df.select_dtypes(include=[np.number])
    all_cols = num_df.columns.tolist()
    
    st.subheader("⚙️ 2. 解析の設定")
    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        selected_cols = st.multiselect("項目を選択", all_cols, default=all_cols[:min(8, len(all_cols))])
    with col_s2:
        threshold = st.slider("表示する影響度のしきい値", 0.00, 0.50, 0.10, 0.01)

    if len(selected_cols) < 2:
        st.info("2つ以上の項目を選択してください。")
        return

    # --- 3. 解析実行 ---
    if st.button(f"🚀 {label} の因果構造を推定する"):
        X = num_df[selected_cols].dropna()
        if len(X) < 15:
            st.error("❌ データ数が少なすぎます。")
            return

        with st.spinner("因果関係を計算中..."):
            try:
                # 💡 以前の仕様通り、制約（Prior Knowledge）なしの純粋なLiNGAM
                model = lingam.DirectLiNGAM()
                model.fit(X)
                adj_matrix = model.adjacency_matrix_

                # --- グラフ描画（右クリック保存を可能にする st.image 仕様） ---
                st.subheader(f"📊 {label} の推定因果グラフ")
                dot = graphviz.Digraph(engine='dot', format='png')
                dot.attr(rankdir='LR', overlap='false', splines='true')
                dot.attr('node', shape='box', fontname='MS Gothic', style='filled', color='#E1F5FE', fontcolor='#1a2a6c')

                edges = []
                for i, row in enumerate(adj_matrix):
                    for j, val in enumerate(row):
                        if abs(val) >= threshold:
                            dot.edge(selected_cols[j], selected_cols[i], label=f"{val:.2f}", 
                                     penwidth=str(max(1, abs(val) * 5)), color="#1a2a6c" if val > 0 else "#d62728")
                            edges.append({'原因': selected_cols[j], '結果': selected_cols[i], '影響度': val})

                for col in selected_cols:
                    dot.node(col, col)

                # 💡 st.graphviz_chart ではなく、画像として出力することで右クリック保存を実現
                st.image(dot.pipe(format='png'), caption=f"💡 右クリックで画像を保存できます", use_container_width=True)

                # --- 4. 改善の起点（しきい値考慮の以前のロジック） ---
                st.markdown("---")
                st.subheader("🎯 改善の起点（根本原因）")
                masked_adj = np.where(np.abs(adj_matrix) >= threshold, adj_matrix, 0)
                in_degrees = np.abs(masked_adj).sum(axis=1)
                out_degrees = np.abs(masked_adj).sum(axis=0)
                roots = [selected_cols[i] for i, deg in enumerate(in_degrees) if deg < 0.01 and out_degrees[i] > 0]
                
                if roots:
                    st.success(f"**{', '.join(roots)}** を優先的にケアすることで、効率的な改善が見込めます。")
                else:
                    st.info("現在の設定では、明確な起点は見当たりません。")

                # --- 5. 影響度ランキング ---
                st.markdown("---")
                st.subheader("📈 影響度ランキング")
                if edges:
                    st.dataframe(pd.DataFrame(edges).sort_values('影響度', ascending=False), hide_index=True, use_container_width=True)

            except Exception as e:
                st.error(f"解析エラー: {e}")