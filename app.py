import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ========== 1. データ読み込み・前処理・クラスタリング ==========
df = pd.read_csv("データ.csv")
df = df.fillna(0)
# シーン列をバイナリ列に変換
df["普段使い"] = df["シーン"].apply(lambda x: 1 if "普段使い" in x else 0)
df["ビジネス"] = df["シーン"].apply(lambda x: 1 if "ビジネス" in x else 0)
df["切り替え"] = df["シーン"].apply(lambda x: 1 if "切り替え" in x else 0)
df["特別な時"] = df["シーン"].apply(lambda x: 1 if "特別な時" in x else 0)
df = df.drop(columns=["シーン"])
# 性別をバイナリ列に変換
df["男性"] = df["性別"].apply(lambda x: 1 if x in ["男性", "ユニセックス"] else 0)
df["女性"] = df["性別"].apply(lambda x: 1 if x in ["女性", "ユニセックス"] else 0)
df = df.drop(columns=["性別"])
# 濃度を数値マッピング（例）
df["濃度"] = df["濃度"].map({"EDT": 5, "EDP": 10, "EDC": 1})
# 標準化する列の抽出
columns_to_scale = df.loc[:, '価格':].columns
scaler = StandardScaler()
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
# K-Meansによるクラスタリング
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(df.loc[:, '価格':])
# 特徴量カラム (最後の 'Cluster' 列を除く)
feature_columns = df.select_dtypes(include=[np.number]).columns[:-1]

# ========== 2. レコメンド関数の定義 ==========
def recommend(input_data, df, feature_columns):
    """
    input_data : 標準化済みのユーザー入力データ（1次元）
    df         : クラスタ情報等を含む元データフレーム
    feature_columns : 類似度計算に使用する列
    """
    cluster_label = kmeans.predict([input_data])[0]
    similar_items = df[df['Cluster'] == cluster_label].copy()
    similarities = cosine_similarity([input_data], similar_items[feature_columns])
    similar_items['Similarity'] = similarities[0]
    recommendations = similar_items.sort_values(by='Similarity', ascending=False)
    return recommendations

# ========== 3. Flask アプリの構築 ==========
app = Flask(__name__)

# --- (A) トップページ（ホーム画面）: index.html を表示 ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# --- (B) 診断フォーム画面: home2.html を表示 ---
@app.route('/home', methods=['GET'])
def home_form():
    return render_template('home.html')

# --- (C) フォーム送信後: レコメンド結果を表示 ---
@app.route('/recommend', methods=['POST'])
def get_recommendation():
    price = request.form.get('price', type=float, default=10000)
    fragrance_list = request.form.getlist('fragrance')
    concentration = request.form.get('concentration', type=float, default=5)
    season_list = request.form.getlist('season')
    scene_list = request.form.getlist('scene')
    gender = request.form.get('gender', '男性')
    
    fragrance_mapping = {
        "フルーティ": 1, "ホワイトフローラル": 2, "フローラル": 3, "ウッディ": 4,
        "シトラス": 5, "ムスク": 6, "グリーンフローラル": 7, "アロマティック": 8,
        "アンバー": 9, "アクア": 10, "アニマリック": 11, "パウダリー": 12,
        "グルマン": 13, "グリーン": 14, "スパイシー": 15, "オリエンタル": 16,
        "ハーバル": 17
    }
    fragrance_flags = {}
    for key in fragrance_mapping.keys():
        fragrance_flags[key] = 1 if key in fragrance_list else 0
    season_flags = {
        "春": 1 if "春" in season_list else 0,
        "夏": 1 if "夏" in season_list else 0,
        "秋": 1 if "秋" in season_list else 0,
        "冬": 1 if "冬" in season_list else 0,
    }
    scene_flags = {
        "普段使い": 1 if "普段使い" in scene_list else 0,
        "ビジネス": 1 if "ビジネス" in scene_list else 0,
        "切り替え": 1 if "切り替え" in scene_list else 0,
        "特別な時": 1 if "特別な時" in scene_list else 0,
    }
    male = 1 if gender in ["男性", "ユニセックス"] else 0
    female = 1 if gender in ["女性", "ユニセックス"] else 0

    user_input_list = []
    user_input_list.append(price)
    for key in fragrance_mapping.keys():
        user_input_list.append(fragrance_flags[key])
    user_input_list.append(concentration)
    for s in ["春", "夏", "秋", "冬"]:
        user_input_list.append(season_flags[s])
    for sc in ["普段使い", "ビジネス", "切り替え", "特別な時"]:
        user_input_list.append(scene_flags[sc])
    user_input_list.append(male)
    user_input_list.append(female)
    
    scaled_user_input = scaler.transform([user_input_list])[0]
    recommendations = recommend(input_data=scaled_user_input,
                                df=df,
                                feature_columns=feature_columns)
    top_n = 3
    top_recommendations = recommendations.head(top_n)
    
    # 各香水のリンク先をマッピング
    perfume_url_mapping = {
        "ザ・ワン　フォーメン　オードパルファム": "https://coloria.jp/items/369?referer=search%2Fkeyword_suggest",
        "ソヴァージュ　オードゥ　トワレ": "https://coloria.jp/items/152?referer=search%2Fkeyword_suggest",
        "ジミーチュウ　マン　アイス　オードトワレ": "https://coloria.jp/items/400?referer=perfumes%2Fsearch",
        "ジミーチュウ　ロー": "https://coloria.jp/items/248?referer=search%2Fkeyword_suggest",
        "ピュア　プワゾン": "https://coloria.jp/items/360?referer=search%2Fkeyword_suggest",
        "モン　パリ": "https://coloria.jp/items/86?referer=search%2Fkeyword_suggest"
        # 他の香水についても必要に応じてマッピングを追加してください
    }
    
    result_list = []
    for i, row in top_recommendations.iterrows():
        result_list.append({
            "rank": len(result_list) + 1,
            "brand": row["ブランド名"],
            "perfume": row["香水名"],
            "url": perfume_url_mapping.get(row["香水名"], "#")
        })
    return render_template("result.html",
                           result_list=result_list,
                           price=price,
                           fragrance_list=fragrance_list,
                           concentration=concentration,
                           season_list=season_list,
                           scene_list=scene_list,
                           gender=gender)


if __name__ == "__main__":
    app.run(debug=True)
