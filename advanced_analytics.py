import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import japanize_matplotlib
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap
import re

def advanced_analytics_app():
    st.markdown("# 🤖 AI Deep Diagnosis: Pro Mode")
    st.caption("Optunaで最適化したRandom ForestとSHAPを用いた、高精度な要因特定プラットフォーム")

    if 'df_encoded' not in st.session_state:
        st.warning("⚠️ 解析データが見つかりません。先に「EDA」画面でデータを準備してください。")
        return

    # --- データのクリーンアップ ---
    df = st.session_state.df_encoded.copy()
    original_cols = df.columns.tolist()
    clean_cols = [re.sub(r'[ \t\n\r\f\v（）() -]', '_', col) for col in original_cols]
    df.columns = clean_cols
    col_mapping = dict(zip(clean_cols, original_cols))
    inv_col_mapping = {v: k for k, v in col_mapping.items()}

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in df.columns if df[c].dtype == 'object' or df[c].dtype.name == 'category']

    # --- 1. 分析範囲の切り替え ---
    st.subheader("🛠 分析範囲の切り替え")
    filter_cat = st.selectbox("属性別に分析を比較する", ["全体"] + cat_cols)

    if filter_cat == "全体":
        display_df = df
        label = "全体"
    else:
        unique_vals = sorted(df[filter_cat].dropna().unique().tolist())
        f_val = st.selectbox(f"{filter_cat}を選択", unique_vals)
        display_df = df[df[filter_cat] == f_val]
        label = f"{filter_cat}: {f_val}"

    st.markdown("---")

    # --- 2. サイドバー設定 ---
    st.sidebar.header("🎯 AIモデル設定")
    target = st.sidebar.selectbox("目的変数 (Outcome)", num_cols)
    remaining_nums = [c for c in num_cols if c != target]
    features = st.sidebar.multiselect("説明変数 (Drivers)", remaining_nums, default=remaining_nums[:10])
    n_trials = st.sidebar.slider("Optuna 試行回数", 10, 50, 20)

    if not features:
        st.info("💡 サイドバーから説明変数を選択してください。")
        return

    # --- 3. 解析実行セクション ---
    st.write(f"### 🚀 {label} の精密診断")
    
    if st.button("精密診断を開始する（Optuna + RF）"):
        X = display_df[features].fillna(display_df[features].median())
        y = display_df[target]

        if len(X) < 5:
            st.error("❌ データ数が少なすぎます。最低5件必要です。")
            return

        with st.spinner("AIが最適なモデルを構築中..."):
            try:
                # Optunaによるパラメータ最適化（既存ロジック維持）
                def objective(trial):
                    param = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                        'random_state': 42
                    }
                    model = RandomForestRegressor(**param)
                    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
                    model.fit(X_train, y_train)
                    return model.score(X_val, y_val)

                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=n_trials)

                # ベストモデル作成
                best_model = RandomForestRegressor(**study.best_params, random_state=42)
                best_model.fit(X, y)

                # 予測値と予測誤差の計算
                y_pred = best_model.predict(X)
                display_df['AI予測値'] = y_pred
                display_df['予測誤差'] = y - y_pred

                # SHAP値計算
                explainer = shap.TreeExplainer(best_model)
                shap_values = explainer.shap_values(X)

                # セッション保存
                st.session_state['last_rf_model'] = best_model
                st.session_state['last_rf_shap'] = shap_values
                st.session_state['last_rf_features'] = features
                st.session_state['last_rf_df'] = display_df.copy()
                st.session_state['last_rf_X'] = X
                st.rerun()
            except Exception as e:
                st.error(f"モデル構築中に失敗しました: {e}")

    # --- 4. 結果表示（セッションデータがある場合） ---
    if 'last_rf_model' in st.session_state:
        res_df = st.session_state['last_rf_df']
        best_model = st.session_state['last_rf_model']
        current_features = st.session_state['last_rf_features']
        shap_values = st.session_state['last_rf_shap']
        current_X = st.session_state['last_rf_X']
        
        # ベースライン（全データの平均予測値）
        base_value = res_df['AI予測値'].mean()

        # ① 全体の重要度
        st.markdown("#### ① モデルが重視した項目 (Feature Importance)")
        imp_df = pd.DataFrame({
            '要素': [col_mapping.get(c, c) for c in current_features],
            '重要度': best_model.feature_importances_
        }).sort_values('重要度', ascending=True)
        
        fig_height = max(400, len(imp_df) * 25)
        st.plotly_chart(px.bar(imp_df, x='重要度', y='要素', orientation='h', 
                               height=fig_height, color_discrete_sequence=['#3498db']), use_container_width=True)

        # ② ギャップ分析
        if 'AI予測値' in res_df.columns:
            st.markdown("#### ② AI予測と実績の乖離 (Gap Analysis)")
            try:
                plot_df = res_df.copy().replace([np.inf, -np.inf], np.nan).dropna(subset=['AI予測値', target])
                fig_gap = px.scatter(
                    plot_df, x='AI予測値', y=target, color='予測誤差',
                    color_continuous_scale='RdBu',
                    labels={'AI予測値': 'AIが予測した期待値', target: '本人の回答（実績）'},
                    title="対角線より下：期待より不満が強い / 上：期待より満足が高い"
                )
                fig_gap.add_shape(type="line", x0=plot_df[target].min(), y0=plot_df[target].min(),
                                  x1=plot_df[target].max(), y1=plot_df[target].max(),
                                  line=dict(color="Red", dash="dash"))
                st.plotly_chart(fig_gap, use_container_width=True)
            except Exception as e:
                st.error(f"グラフ描画エラー: {e}")

        st.markdown("---")

        # ③ 個別要因診断 (SHAP)
        st.markdown("### 🔍 個別要因診断 (SHAP Analysis)")
        
        acc_col = inv_col_mapping.get('アカウント', 'アカウント')
        name_col = inv_col_mapping.get('職場氏名', '職場氏名')
        sorted_data = res_df.sort_values(target)

        def get_display_name(idx):
            name_parts = []
            if acc_col in res_df.columns: name_parts.append(str(res_df.loc[idx, acc_col]))
            if name_col in res_df.columns: name_parts.append(str(res_df.loc[idx, name_col]))
            return f'{" | ".join(name_parts) if name_parts else idx} (Score: {res_df.loc[idx, target]:.2f})'

        target_idx = st.selectbox("診断対象を選択 (低スコア順)", sorted_data.index, format_func=get_display_name)

        if target_idx is not None:
            row_pos = current_X.index.get_loc(target_idx)
            person_shap = shap_values[row_pos]
            total_shap = person_shap.sum() # 要因の合計寄与度
            
            diag_df = pd.DataFrame({
                '要素': [col_mapping.get(c, c) for c in current_features],
                '寄与度': person_shap
            }).sort_values('寄与度', ascending=True)

            st.write(f"#### 診断レポート: {get_display_name(target_idx)}")
            
            # --- 🚀 ベースライン・メトリクス表示 ---
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("組織平均", f"{base_value:.2f}")
            with m2:
                st.metric("要因合計(SHAP)", f"{total_shap:+.2f}", delta_color="normal")
            with m3:
                st.metric("AI予測スコア", f"{res_df.loc[target_idx, 'AI予測値']:.2f}")
            with m4:
                st.metric("本人の回答", f"{res_df.loc[target_idx, target]:.2f}")

            st.info(f"💡 解説：この組織の平均点 {base_value:.2f} に対して、あなたの特徴をすべて合算すると {total_shap:+.2f} の影響があり、AIは結果として {res_df.loc[target_idx, 'AI予測値']:.2f} 点と算出しました。")

            l_col, r_col = st.columns([1, 2])
            with l_col:
                if '予測誤差' in res_df.columns:
                    err = res_df.loc[target_idx, '予測誤差']
                    if err < -0.8:
                        st.error(f"🚨 予測との乖離 ({err:+.2f}): 数値データ以外の要因（人間関係など）が強く疑われます。")
                    elif err > 0.8:
                        st.success(f"✨ 予測との乖離 ({err:+.2f}): 条件以上に本人の意欲が高い成功事例です。")

                st.write("**🔴 スコアを下げている要因 (TOP3)**")
                st.dataframe(diag_df.head(3), hide_index=True)
                
                st.write("**🔵 スコアを上げている要因 (TOP3)**")
                st.dataframe(diag_df.tail(3).sort_values('寄与度', ascending=False), hide_index=True)
            
            with r_col:
                person_fig_height = max(400, len(diag_df) * 25)
                fig_p = px.bar(diag_df, x='寄与度', y='要素', orientation='h', color='寄与度', 
                               height=person_fig_height,
                               color_continuous_scale='RdBu_r', title="平均点からのプラスマイナス内訳")
                fig_p.add_vline(x=0, line_dash="dash", line_color="black")
                fig_p.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=200))
                st.plotly_chart(fig_p, use_container_width=True)