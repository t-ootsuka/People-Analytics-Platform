import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import japanize_matplotlib
import matplotlib.font_manager as fm
from collections import Counter
import re

def text_analytics_app():
    st.markdown("# 📝 Insight-Words: Tactical Text Analytics")
    
    # 1. セッションデータの確認
    if 'df_encoded' not in st.session_state:
        st.warning("⚠️ 解析データが見つかりません。先にEDA画面でデータを準備してください。")
        return

    df = st.session_state.df_encoded.copy()
    
    # 2. テキスト列とカテゴリ列の自動抽出
    text_cols = [c for c in df.columns if df[c].dtype == 'object']
    cat_cols = [c for c in df.columns if df[c].dtype == 'object' or df[c].dtype.name == 'category']
    num_cols = df.select_dtypes(include=['number']).columns.tolist()

    if not text_cols:
        st.error("❌ 解析可能なテキスト列が見つかりません。")
        return

    # 3. サイドバー設定
    with st.sidebar:
        st.header("🔍 分析設定")
        selected_text_col = st.selectbox("分析するテキスト項目", text_cols)
        target_score = st.selectbox("比較基準とするスコア (高低判定用)", num_cols)
        filter_cat = st.selectbox("注目する属性（部署・年次など）", cat_cols)

    # --- 4. 属性別コメント閲覧セクション ---
    st.subheader(f"📂 {filter_cat}別の発言一覧")
    
    unique_vals = sorted(df[filter_cat].dropna().unique().tolist())
    selected_val = st.selectbox(f"{filter_cat}を選択してください", unique_vals)
    
    # 属性でフィルタリング
    df_filtered = df[df[filter_cat] == selected_val].copy()
    
    # 【重要】nanと空文字を徹底排除（修正済み）
    df_filtered = df_filtered.dropna(subset=[selected_text_col])
    df_filtered = df_filtered[
        (df_filtered[selected_text_col].astype(str).str.strip() != "") & 
        (df_filtered[selected_text_col].astype(str).str.lower() != "nan")
    ]

    with st.expander(f"📖 {selected_val} の全コメントを表示（{len(df_filtered)}件）", expanded=True):
        if len(df_filtered) == 0:
            st.write("有効なコメントはありません。")
        else:
            for comment in df_filtered[selected_text_col]:
                # 念のため表示直前でもチェック
                if pd.notna(comment) and str(comment).lower() != "nan":
                    st.write(f"・ {comment}")
                    st.divider()

    st.markdown("---")

    # --- 5. スコアによるグループ分割（高低判定） ---
    high_group = df_filtered[df_filtered[target_score] >= 4][selected_text_col]
    low_group = df_filtered[df_filtered[target_score] <= 2][selected_text_col]

    st.subheader(f"📊 {selected_val} のインサイト分析")
    
    if high_group.empty and low_group.empty:
        st.warning("💡 対象となる1,2点または4,5点の有効な回答がこの属性にはありません。")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🔵 満足層 (4,5点)")
            generate_wordcloud(high_group, "Blues")
        with col2:
            st.markdown("### 🔴 不満層 (1,2点)")
            generate_wordcloud(low_group, "Reds")

    # --- 6. キーワード深掘り（スコア付表示を復活） ---
    st.markdown("---")
    st.subheader("🔍 単語から生の声を探る（スコア付）")
    
    all_text = " ".join(df_filtered[selected_text_col].astype(str))
    words_for_select = re.findall(r"[\u4e00-\u9fa5]{2,}|[\u30a1-\u30f6]{2,}", all_text)
    stop_words = ["思う", "ます", "です", "こと", "もの", "ため", "感じ", "そう", "ない", "ある", "ので", "それ"]
    common_words = [w for w, _ in Counter(words_for_select).most_common(50) if w not in stop_words and w.lower() != "nan"]
    
    selected_word = st.selectbox("詳しく見たいキーワードを選択", ["-- 選択してください --"] + common_words)
    
    if selected_word != "-- 選択してください --":
        hit_df = df_filtered[df_filtered[selected_text_col].str.contains(selected_word, na=False)]
        st.write(f"📌 「**{selected_word}**」を含むコメント:")
        
        for _, row in hit_df.iterrows():
            score = row[target_score]
            text_val = row[selected_text_col]
            # 以前の仕様どおり、スコアとセットで色分け表示
            if score >= 4:
                st.info(f"**【スコア:{score}】** {text_val}")
            elif score <= 2:
                st.error(f"**【スコア:{score}】** {text_val}")
            else:
                st.warning(f"**【スコア:{score}】** {text_val}")

    # --- 7. トークスクリプト ---
    st.markdown("---")
    display_talk_script(low_group)

def generate_wordcloud(text_series, colormap):
    text = " ".join(text_series.astype(str))
    words = re.findall(r"[\u4e00-\u9fa5]{2,}|[\u30a1-\u30f6]{2,}", text)
    stop_words = ["思う", "ます", "です", "こと", "もの", "ため", "感じ", "そう", "ない", "ある", "ので"]
    filtered_text = " ".join([w for w in words if w not in stop_words and w.lower() != "nan"])

    if not filtered_text.strip():
        st.write("表示可能な単語がありません")
        return

    try:
        font_path = fm.findfont(fm.FontProperties(family='IPAexGothic'))
    except:
        font_path = None

    wc = WordCloud(font_path=font_path, background_color='white', colormap=colormap, width=800, height=500).generate(filtered_text)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

def display_talk_script(low_group):
    text = " ".join(low_group.astype(str))
    words = re.findall(r"[\u4e00-\u9fa5]{2,}", text)
    top_words = [w for w, _ in Counter(words).most_common(3) if w.lower() != "nan"]
    if top_words:
        st.subheader("🤖 AI 戦略トークスクリプト")
        st.success(f"「最近、現場では **『{'』『'.join(top_words)}』** という声も聞こえるのだけど、何か改善できそうな点はあるかな？」")