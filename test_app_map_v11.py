import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- アプリケーションのタイトル ---
st.set_page_config(layout="wide")
st.title("統計データで発見！あなたの知らない“地域の消費傾向”")

# --- データソースの説明 ---
st.sidebar.header("データについて")
st.sidebar.info(
    "このダッシュボードは、総務省統計局の「家計調査」のデータ (SSDSE-C-2023) を使用して、"
    "地域ごとの特徴的な食の消費傾向を分析・可視化します。"
)

# --- データの読み込み ---
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
        # '地域コード', '都道府県', '市', '世帯人員' 以外のカラムを数値型に変換
        # これらの列名が固定であると仮定
        non_numeric_cols = ['地域コード', '都道府県', '市', '世帯人員']
        cols_to_convert = [col for col in df.columns if col not in non_numeric_cols]
        for col in cols_to_convert:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except FileNotFoundError:
        st.error(f"エラー: ファイルが見つかりません: {file_path}")
        return None
    except Exception as e:
        st.error(f"データの読み込み中にエラーが発生しました: {e}")
        return None

df_kakei_original = load_data("SSDSE-C-2023(utf-8).csv")

if df_kakei_original is not None:
    # 分析対象の品目カラムを動的に取得（最初の4列を除外と仮定）
    # より堅牢にするには、品目カラムのリストを別途定義するか、
    # '世帯人員'カラムの次のカラムから最後まで、といった指定が適切
    potential_item_columns = [col for col in df_kakei_original.columns if col not in ['地域コード', '都道府県', '市', '世帯人員']]
    
    if not potential_item_columns:
        st.error("読み込まれたデータに分析対象となる食料品目が含まれていません。CSVファイルの列名や内容を確認してください。")
        st.stop()

    item_columns = potential_item_columns

    st.sidebar.header("分析設定")
    default_item_name = '生うどん・そば'
    if default_item_name in item_columns:
        default_idx = item_columns.index(default_item_name)
    elif item_columns:
        default_idx = 0
    else: # item_columnsが空の場合 (上記でst.stop()するが念のため)
        st.error("分析可能な品目がありません。")
        st.stop()

    selected_item_column = st.sidebar.selectbox(
        '分析したい食料品目を選択してください:',
        item_columns,
        index=default_idx
    )

    # --- 外れ値検出方法の選択 ---
    st.sidebar.header("注目地域の判定方法")
    outlier_method_options_map = {
        "平均 + 1.5σ": "mean_std",
        "IQR法 (Q3 + 1.5 × IQR)": "iqr",
        "パーセンタイル法": "percentile"
    }
    selected_outlier_method_label = st.sidebar.radio(
        "判定方法を選択:",
        list(outlier_method_options_map.keys()),
        key="outlier_detection_method_radio"
    )
    selected_outlier_method_value = outlier_method_options_map[selected_outlier_method_label]

    percentile_value_for_outlier = None
    if selected_outlier_method_value == "percentile":
        percentile_value_for_outlier = st.sidebar.slider(
            "上位何パーセントを注目地域としますか？ (%)",
            min_value=1, max_value=25, value=5, step=1,
            key="percentile_slider_input",
            help="選択した品目の年間支出額が、全都市の中で上位X%に入る都市を「注目地域」とします。"
        )

    # 都市の緯度経度情報
    city_lat_lon = {
        "札幌市": {"lat": 43.06417, "lon": 141.34694}, "青森市": {"lat": 40.82444, "lon": 140.74},
        "盛岡市": {"lat": 39.70361, "lon": 141.1525}, "仙台市": {"lat": 38.26889, "lon": 140.87194},
        "秋田市": {"lat": 39.71861, "lon": 140.1025}, "山形市": {"lat": 38.255, "lon": 140.34083},
        "福島市": {"lat": 37.75, "lon": 140.46778}, "水戸市": {"lat": 36.36583, "lon": 140.47139},
        "宇都宮市": {"lat": 36.555, "lon": 139.88278}, "前橋市": {"lat": 36.38944, "lon": 139.06333},
        "さいたま市": {"lat": 35.85694, "lon": 139.645}, "千葉市": {"lat": 35.60472, "lon": 140.12306},
        "東京都区部": {"lat": 35.68944, "lon": 139.69167}, "横浜市": {"lat": 35.44778, "lon": 139.6425},
        "新潟市": {"lat": 37.90222, "lon": 139.02361}, "富山市": {"lat": 36.69528, "lon": 137.21361},
        "金沢市": {"lat": 36.59444, "lon": 136.62556}, "福井市": {"lat": 36.06417, "lon": 136.21944},
        "甲府市": {"lat": 35.66222, "lon": 138.56833}, "長野市": {"lat": 36.65139, "lon": 138.18111},
        "岐阜市": {"lat": 35.42306, "lon": 136.76056}, "静岡市": {"lat": 34.97694, "lon": 138.38306},
        "名古屋市": {"lat": 35.18028, "lon": 136.90667}, "津市": {"lat": 34.71722, "lon": 136.505},
        "大津市": {"lat": 35.00444, "lon": 135.86833}, "京都市": {"lat": 35.01111, "lon": 135.76694},
        "大阪市": {"lat": 34.68639, "lon": 135.52}, "神戸市": {"lat": 34.69139, "lon": 135.18306},
        "奈良市": {"lat": 34.68528, "lon": 135.83278}, "和歌山市": {"lat": 34.23028, "lon": 135.17083},
        "鳥取市": {"lat": 35.50361, "lon": 134.23556}, "松江市": {"lat": 35.46806, "lon": 133.04833},
        "岡山市": {"lat": 34.66167, "lon": 133.935}, "広島市": {"lat": 34.39639, "lon": 132.45944},
        "山口市": {"lat": 34.18583, "lon": 131.47139}, "徳島市": {"lat": 34.07028, "lon": 134.55472},
        "高松市": {"lat": 34.34167, "lon": 134.04333}, "松山市": {"lat": 33.83944, "lon": 132.76556},
        "高知市": {"lat": 33.55972, "lon": 133.53111}, "福岡市": {"lat": 33.59028, "lon": 130.40194},
        "佐賀市": {"lat": 33.26389, "lon": 130.30056}, "長崎市": {"lat": 32.74472, "lon": 129.87361},
        "熊本市": {"lat": 32.78972, "lon": 130.74167}, "大分市": {"lat": 33.23806, "lon": 131.6125},
        "宮崎市": {"lat": 31.91111, "lon": 131.42389}, "鹿児島市": {"lat": 31.56028, "lon": 130.55806},
        "那覇市": {"lat": 26.2125, "lon": 127.68111}
    }

    st.header(f"【{selected_item_column}】の消費傾向分析 (年間支出額)")

    # --- データ準備 (緯度経度付与、外れ値判定) ---
    df_processed = df_kakei_original[['市', '都道府県', selected_item_column]].copy()
    df_processed.dropna(subset=[selected_item_column], inplace=True)

    df_processed['lat'] = df_processed['市'].apply(lambda x: city_lat_lon.get(x, {}).get('lat'))
    df_processed['lon'] = df_processed['市'].apply(lambda x: city_lat_lon.get(x, {}).get('lon'))

    # 外れ値の計算
    outlier_cities_set = set()
    high_outliers = pd.DataFrame(columns=df_processed.columns) # 初期化
    calculated_threshold_str = "計算できませんでした"
    method_description_str = "データが不足しているか、判定方法が正しく選択されていません。"

    spend_data = df_processed[selected_item_column].dropna()

    if len(spend_data) >= 2: # 統計計算には最低2つのデータポイントが必要
        outlier_threshold_high = -1 # 初期値

        if selected_outlier_method_value == "mean_std":
            mean_exp = spend_data.mean()
            std_exp = spend_data.std()
            if std_exp > 0 :
                 outlier_threshold_high = mean_exp + 1.5 * std_exp
                 method_description_str = f"平均 ({mean_exp:,.0f} 円) + 1.5σ ({std_exp:,.0f} 円) を超える都市"
            else: # 全てのデータが同じ値の場合など
                 outlier_threshold_high = mean_exp + 1e-9 # 実質、平均より大きいもの（ほぼ存在しない）
                 method_description_str = f"平均 ({mean_exp:,.0f} 円) + 1.5σ (σ=0) を超える都市"
            calculated_threshold_str = f"{outlier_threshold_high:,.0f} 円"

        elif selected_outlier_method_value == "iqr":
            Q1 = spend_data.quantile(0.25)
            Q3 = spend_data.quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                outlier_threshold_high = Q3 + 1.5 * IQR
                method_description_str = f"Q3 ({Q3:,.0f} 円) + 1.5×IQR ({IQR:,.0f} 円) を超える都市"
            else: # データが集中している場合など
                outlier_threshold_high = Q3 + 1e-9 # 実質、Q3より大きいもの
                method_description_str = f"Q3 ({Q3:,.0f} 円) + 1.5×IQR (IQR=0) を超える都市"
            calculated_threshold_str = f"{outlier_threshold_high:,.0f} 円"

        elif selected_outlier_method_value == "percentile":
            if percentile_value_for_outlier is not None:
                outlier_threshold_high = np.percentile(spend_data, 100 - percentile_value_for_outlier)
                calculated_threshold_str = f"{outlier_threshold_high:,.0f} 円"
                method_description_str = f"支出額が上位 {percentile_value_for_outlier}% に入る都市 (閾値: {calculated_threshold_str})"
            else: # スライダーが未設定(通常発生しない)
                 method_description_str = "パーセンタイルが設定されていません。"


        if outlier_threshold_high != -1:
             high_outliers = df_processed[df_processed[selected_item_column] > outlier_threshold_high]
             outlier_cities_set = set(high_outliers['市'])

    else:
        method_description_str = "有効な支出データが2件未満のため、注目地域を計算できませんでした。"
        calculated_threshold_str = "N/A"


    df_processed['is_outlier_high'] = df_processed['市'].isin(outlier_cities_set)
    df_processed['expenditure'] = df_processed[selected_item_column]

    map_plotly_data = df_processed.dropna(subset=['lat', 'lon', 'expenditure']).copy()
    map_plotly_data = map_plotly_data[map_plotly_data['expenditure'] > 0]

    # --- 1. 発見！消費トレンド・外れ値マップ ---
    st.subheader(f"消費トレンド・外れ値マップ： {selected_item_column}")
    st.write(f"地図上で各都市の **{selected_item_column}** への年間支出額を円の大きさで示します。")
    st.write(f"特に支出額が高い「注目地域」は赤色で表示されます。現在の注目地域の判定基準: **{selected_outlier_method_label}**")
    if calculated_threshold_str != "N/A" and calculated_threshold_str != "計算できませんでした":
         if selected_outlier_method_value != "percentile": # パーセンタイルはmethod_description_strに閾値情報含む
            st.write(f"(判定閾値の目安: 約 {calculated_threshold_str})")
    st.markdown("<small>注意: 地図表示は緯度経度が事前に定義された一部の都市のみです。</small>", unsafe_allow_html=True)

    if not map_plotly_data.empty:
        map_plotly_data['消費カテゴリ'] = map_plotly_data['is_outlier_high'].apply(lambda x: '注目地域 (高支出)' if x else '通常')

        fig_map = px.scatter_mapbox(
            map_plotly_data,
            lat="lat",
            lon="lon",
            size="expenditure",
            color="消費カテゴリ",
            hover_name="市",
            hover_data={"都道府県": True, "expenditure": ':.0f', "消費カテゴリ": True, "lat":False, "lon":False},
            color_discrete_map={'注目地域 (高支出)': 'crimson', '通常': 'cornflowerblue'},
            size_max=30,
            zoom=4,
            height=700,
            mapbox_style="carto-positron"
        )
        fig_map.update_layout(
            margin={"r":0,"t":40,"l":0,"b":0},
            legend_title_text='凡例',
            title_text=f'{selected_item_column} の都市別消費マップ (支出額と注目地域)'
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.write(f"選択された品目「{selected_item_column}」について、地図に表示できる有効なデータがありませんでした。")

    # --- 2. 発掘！外れ値ハンター (詳細情報) ---
    st.subheader(f"{selected_item_column} の詳細情報")
    st.write(f"**{selected_item_column}** の消費において、統計的に支出額が高い地域（注目地域）と、全体の支出分布を確認します。")
    st.markdown(f"**現在の注目地域の判定基準:** {selected_outlier_method_label}")
    st.markdown(f"*{method_description_str}*")


    if not spend_data.empty:
        if not high_outliers.empty:
            st.write(f"**「注目地域」（{selected_outlier_method_label}基準）:**")
            st.dataframe(high_outliers[['都道府県', '市', selected_item_column]].sort_values(by=selected_item_column, ascending=False).reset_index(drop=True))

        st.write(f"**{selected_item_column} の支出金額ヒストグラム (全都市):**")
        fig_hist = px.histogram(
            df_processed.dropna(subset=[selected_item_column]),
            x=selected_item_column,
            hover_data=['市', '都道府県'],
            nbins=30,
            title=f"{selected_item_column} の支出金額分布"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.write(f"**{selected_item_column} の支出額ランキング (上位10都市):**")
        st.dataframe(df_processed[['都道府県', '市', selected_item_column]].sort_values(by=selected_item_column, ascending=False).head(10).reset_index(drop=True))
    else:
        st.write(f"{selected_item_column} の有効な支出データが分析のために不足しています。")



else:
    st.error("データの読み込みに失敗したため、ダッシュボードを表示できません。CSVファイルが存在し、内容が正しいか確認してください。")
    st.stop()

# --- フッター ---