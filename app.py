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
    # KMeansからクラスタ予測（入力データがどのクラスタに属するか）
    cluster_label = kmeans.predict([input_data])[0]
    # 同じクラスタのデータを抽出
    similar_items = df[df['Cluster'] == cluster_label].copy()
    # コサイン類似度を計算
    similarities = cosine_similarity([input_data], similar_items[feature_columns])
    similar_items['Similarity'] = similarities[0]
    # 類似度の降順でソートして上位を返す
    recommendations = similar_items.sort_values(by='Similarity', ascending=False)
    return recommendations
# ========== 3. Flask アプリの構築 ==========
app = Flask(__name__)
# --- (A) ホーム画面: 入力フォームを表示 ---
@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')
# --- (B) フォーム送信後: レコメンド結果を表示 ---
@app.route('/recommend', methods=['POST'])
def get_recommendation():
    # 1) フォーム入力を取得
    price = request.form.get('price', type=float, default=10000)
    fragrance_list = request.form.getlist('fragrance')  # チェックボックス or マルチセレクト想定
    concentration = request.form.get('concentration', type=float, default=5)
    season_list = request.form.getlist('season')
    scene_list = request.form.getlist('scene')
    gender = request.form.get('gender', '男性')
    # 香りマッピング辞書 (例：同じものを使う)
    fragrance_mapping = {
        "フルーティ": 1, "ホワイトフローラル": 2, "フローラル": 3, "ウッディ": 4,
        "シトラス": 5, "ムスク": 6, "グリーンフローラル": 7, "アロマティック": 8,
        "アンバー": 9, "アクア": 10, "アニマリック": 11, "パウダリー": 12,
        "グルマン": 13, "グリーン": 14, "スパイシー": 15, "オリエンタル": 16,
        "ハーバル": 17
    }
    # 選択された香りに対応するフラグを作成 (0 or 1)
    fragrance_flags = {}
    for key in fragrance_mapping.keys():
        # 選ばれた香りリスト(fragrance_list)の中に、このkeyが含まれるかどうか
        # ただしフロント側でvalueに "フルーティ" などの文字列を使う想定
        if key in fragrance_list:
            fragrance_flags[key] = 1
        else:
            fragrance_flags[key] = 0
    # 季節フラグ
    season_flags = {
        "春": 1 if "春" in season_list else 0,
        "夏": 1 if "夏" in season_list else 0,
        "秋": 1 if "秋" in season_list else 0,
        "冬": 1 if "冬" in season_list else 0,
    }
    # シーンフラグ
    scene_flags = {
        "普段使い": 1 if "普段使い" in scene_list else 0,
        "ビジネス": 1 if "ビジネス" in scene_list else 0,
        "切り替え": 1 if "切り替え" in scene_list else 0,
        "特別な時": 1 if "特別な時" in scene_list else 0,
    }
    # 性別フラグ
    male = 1 if gender in ["男性", "ユニセックス"] else 0
    female = 1 if gender in ["女性", "ユニセックス"] else 0
    # 2) ユーザー入力を元にベクトルを構築（dfの列順に注意）
    #   例として [価格, (香り17種), 濃度, (季節4種), (シーン4種), 男性, 女性] の順
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
    # 3) 標準化 (scaler は起動時にfit済み)
    scaled_user_input = scaler.transform([user_input_list])[0]
    # 4) レコメンド関数呼び出し
    recommendations = recommend(input_data=scaled_user_input,
                                df=df,
                                feature_columns=feature_columns)
    # 5) 上位5件のみを抽出
    top_n = 5
    top_recommendations = recommendations.head(top_n)
    # 6) 結果をテンプレートへ渡す
    #   必要に応じて DataFrame から列を取り出して list of dicts に変換してもOK
    #   ここでは brand, perfume 名が "ブランド名" と "香水名" 列にある想定
    #   （CSVの列名と合わせてください）
    result_list = []
    for i, row in top_recommendations.iterrows():
        result_list.append({
            "rank": len(result_list) + 1,
            "brand": row["ブランド名"],
            "perfume": row["香水名"]
        })
    # レンダリング
    return render_template("result.html",
                           result_list=result_list,
                           price=price,
                           fragrance_list=fragrance_list,
                           concentration=concentration,
                           season_list=season_list,
                           scene_list=scene_list,
                           gender=gender)
# ========== 4. Flaskアプリ起動 ==========
if __name__ == "__main__":
    app.run(debug=True)