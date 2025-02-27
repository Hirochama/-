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
        "ジェントルマン": "https://coloria.jp/items/133?referer=search%2Fkeyword_suggest",
        "アイネ　オードパルファム": "https://coloria.jp/items/1493?referer=perfumes%2Fsearch",
        "ティファニー　オードパルファム": "https://coloria.jp/items/504?referer=perfumes%2Fsearch",
        "ジミーチュウ　マン　アイス　オードトワレ": "https://coloria.jp/items/400?referer=perfumes%2Fsearch",
        "ファンタジア　マーメイド": "https://coloria.jp/items/277?referer=perfumes%2Fsearch",
        "クリーン　リザーブ　ウォームコットン": "https://coloria.jp/items/179?referer=perfumes%2Fsearch",
        "ピュア　プワゾン": "https://coloria.jp/items/360?referer=search%2Fkeyword_suggest",
        "ヴェルセンス": "https://coloria.jp/items/36?referer=perfumes%2Fsearch",
        "レプリカ　オードトワレ　アンダー　ザ　レモンツリー": "https://coloria.jp/items/708?referer=perfumes%2Fsearch",
        "ホワイトローズ": "https://coloria.jp/items/1237?referer=perfumes%2Fsearch",
        "地中海の庭": "https://coloria.jp/items/106?referer=perfumes%2Fsearch",
        "チェリーブロッサム": "https://coloria.jp/items/1229?referer=perfumes%2Fsearch",
        "コーチ　フローラル　オードパルファム": "https://coloria.jp/items/235?referer=perfumes%2Fsearch",
        "ナンバー01ミネラル": "https://coloria.jp/items/1734?referer=perfumes%2Fsearch",
        "ホワイトモス　オードパルファン": "https://coloria.jp/items/1066?referer=perfumes%2Fsearch",
        "グリーンティー": "https://coloria.jp/items/29?referer=perfumes%2Fsearch",
        "フローラ　バイ　グッチ　ガーデン　ゴージャス　ガーデニア　オードトワレ": "https://coloria.jp/items/98?referer=perfumes%2Fsearch",
        #ニナ
        "レプリカ　オードトワレ　フラワー　マーケット": "https://coloria.jp/items/710?referer=perfumes%2Fsearch",
        "ザ・ワン　フォーメン　オードパルファム": "https://coloria.jp/items/369?referer=search%2Fkeyword_suggest",
        "ミュシャ　オードトワレ　アイリス": "https://coloria.jp/items/1961?referer=perfumes%2Fsearch",
        "レプリカ　オードトワレ　バイ　ザ　ファイヤープレイス": "https://coloria.jp/items/911?referer=perfumes%2Fsearch",
        "ティファニー　シアー　オードトワレ": "https://coloria.jp/items/505?referer=perfumes%2Fsearch",
        "ミュシャ　オードトワレ　カメリア": "https://coloria.jp/items/1962?referer=perfumes%2Fsearch",
        "レプリカ　オードトワレ　スプリングタイム　イン　ア　パーク": "https://coloria.jp/items/1040?referer=perfumes%2Fsearch",
        "マイバーバリー　ブラッシュ　オードパルファム": "https://coloria.jp/items/320?referer=perfumes%2Fsearch",
        "ジミー　チュウ　フローラル": "https://coloria.jp/items/223?referer=perfumes%2Fsearch",
        "テオブロマ　ショコラオランジェ": "https://coloria.jp/items/1676?referer=perfumes%2Fsearch",
        "ミルクムスク　オードパルファン": "https://coloria.jp/items/988?referer=perfumes%2Fsearch",
        "アールグレー　＆　キューカンバー　コロン": "https://coloria.jp/items/70?referer=perfumes%2Fsearch",
        "スウィートオスマンサス": "https://coloria.jp/items/237?referer=perfumes%2Fsearch",
        "ティファニー&ラブ　オードトワレ　フォーヒム": "https://coloria.jp/items/506?referer=perfumes%2Fsearch",
        "ロスト　チェリー　オード　パルファム　スプレィ": "https://coloria.jp/items/542?referer=perfumes%2Fsearch",
        "屋根の上の庭": "https://coloria.jp/items/108?referer=perfumes%2Fsearch",
        "バンブー　オードパルファム": "https://coloria.jp/items/245?referer=perfumes%2Fsearch",
        "ミス　ディオール": "https://coloria.jp/items/148?referer=perfumes%2Fsearch",
        "テオブロマ　マダガスカル": "https://coloria.jp/items/1679?referer=perfumes%2Fsearch",
        "サクラブルーム　オードトワレ": "https://coloria.jp/items/820?referer=perfumes%2Fsearch",
        "シニョリーナ　ミステリオーサ": "https://coloria.jp/items/46?referer=perfumes%2Fsearch",
        "マリー・ミー！": "https://coloria.jp/items/47?referer=perfumes%2Fsearch",
        "リブロアリア　雨の図書館": "https://coloria.jp/items/1849?referer=perfumes%2Fsearch",
        "ロエベ　001　マン　オードゥ　パルファム": "https://coloria.jp/items/566?referer=perfumes%2Fsearch",
        "クラフトティー　オードトワレ": "https://coloria.jp/items/1524?referer=perfumes%2Fsearch",
        "ミュウミュウ　ロー　ロゼ　オードトワレ": "https://coloria.jp/items/167?referer=perfumes%2Fsearch",
        "ウィステリアブロッサム　オードパルファム": "https://coloria.jp/items/780?referer=perfumes%2Fsearch",
        "ジャンヌ・ランバン": "https://coloria.jp/items/23?referer=perfumes%2Fsearch",
        "シーケーワン": "https://coloria.jp/items/178?referer=perfumes%2Fsearch",
        "ロエベ　001　ウーマン　オードゥ　トワレ": "https://coloria.jp/items/563?referer=perfumes%2Fsearch",
        "ウッド　セージ　＆　シー　ソルト　コロン": "https://coloria.jp/items/77?referer=perfumes%2Fsearch",
        "クリーン　クラシック　ウォームコットン": "https://coloria.jp/items/112?referer=perfumes%2Fsearch",
        "ジュド　オードパルファン": "https://coloria.jp/items/1547?referer=perfumes%2Fsearch",
        "レプリカ　オードトワレ　コーヒーブレイク": "https://coloria.jp/items/711?referer=perfumes%2Fsearch",
        "ロエベ　001　マン　オードゥ　トワレ": "https://coloria.jp/items/565?referer=perfumes%2Fsearch",
        "ジミーチュウ　ロー": "https://coloria.jp/items/248?referer=search%2Fkeyword_suggest",
        "李氏の庭": "https://coloria.jp/items/105?referer=perfumes%2Fsearch",
        "レプリカ　オードトワレ　ジャズ　クラブ": "https://coloria.jp/items/165?referer=perfumes%2Fsearch",
        "オムニア　クリスタリン　オードトワレ": "https://coloria.jp/items/101?referer=perfumes%2Fsearch",
        "クロエ　ラブストーリー　オードパルファム": "https://coloria.jp/items/338?referer=perfumes%2Fsearch",
        "アクア　ユニヴェルサリス　オードトワレ": "https://coloria.jp/items/255?referer=perfumes%2Fsearch",
        "デイジー": "https://coloria.jp/items/256?referer=perfumes%2Fsearch",
        "ディオール　アディクト　オー　フレッシュ": "https://coloria.jp/items/145?referer=perfumes%2Fsearch",
        "モン　パリ": "https://coloria.jp/items/86?referer=search%2Fkeyword_suggest",
        "ジミーチュウ": "https://coloria.jp/items/12?referer=perfumes%2Fsearch",
        "ブラックベリー　＆　ベリー　コロン": "https://coloria.jp/items/72?referer=perfumes%2Fsearch",
        "ブルガリ　プールオム": "https://coloria.jp/items/2?referer=perfumes%2Fsearch",
        "ロエベ　001　ウーマン　オードゥ　パルファン": "https://coloria.jp/items/564?referer=perfumes%2Fsearch",
        "モダンプリンセス": "https://coloria.jp/items/174?referer=perfumes%2Fsearch",
        "エクラ・ドゥ・アルページュ": "https://coloria.jp/items/25?referer=perfumes%2Fsearch",
        "オード　ホワイトフローラル": "https://coloria.jp/items/66?referer=perfumes%2Fsearch",
        "イングリッシュ　ペアー　＆　フリージア　コロン": "https://coloria.jp/items/71?referer=perfumes%2Fsearch",
        "スティル": "https://coloria.jp/items/34?referer=perfumes%2Fsearch",
        "リエラ　オードパルファム": "https://coloria.jp/items/1387?referer=perfumes%2Fsearch",
        "レプリカ　オードトワレ　バブルバス": "https://coloria.jp/items/1038?referer=perfumes%2Fsearch",
        "ジプシー　ウォーター": "https://coloria.jp/items/795?referer=perfumes%2Fsearch",
        "ミュシャ　オードトワレ　リリー": "https://coloria.jp/items/1960?referer=perfumes%2Fsearch",
        "ネクタリン　ブロッサム　&　ハニー　コロン": "https://coloria.jp/items/73?referer=perfumes%2Fsearch",
        "グロウ　バイ　ジェイロー": "https://coloria.jp/items/30?referer=perfumes%2Fsearch",
        "シャドール": "https://coloria.jp/items/93?referer=perfumes%2Fsearch",
        "ロマンティックディープバブルス": "https://coloria.jp/items/1651?referer=perfumes%2Fsearch",
        "リブレ　オードパルファム": "https://coloria.jp/items/547?referer=perfumes%2Fsearch",
        "ティファニー&ラブ　オードパルファム　フォーハー": "https://coloria.jp/items/266?referer=perfumes%2Fsearch",
        "ライトブルー": "https://coloria.jp/items/7?referer=perfumes%2Fsearch",
        "ブランシュ": "https://coloria.jp/items/796?referer=perfumes%2Fsearch",
        "ピュアサボン　オードパルファム": "https://coloria.jp/items/1525?referer=perfumes%2Fsearch",
        "オードモワゼル　フローラル": "https://coloria.jp/items/132?referer=perfumes%2Fsearch",
        "ソヴァージュ　オードゥ　トワレ": "https://coloria.jp/items/152?referer=search%2Fkeyword_suggest",
        "ナイルの庭": "https://coloria.jp/items/32?referer=perfumes%2Fsearch",
        "ミス　ディオール　ブルーミングブーケ": "https://coloria.jp/items/94?referer=perfumes%2Fsearch",
        "バーバリー　ウィークエンド": "https://coloria.jp/items/52?referer=perfumes%2Fsearch",
        "レプリカ　オードトワレ　レイジーサンデーモーニング": "https://coloria.jp/items/703?referer=perfumes%2Fsearch",
        "シアーコットン　オードパルファム": "https://coloria.jp/items/732?referer=perfumes%2Fsearch"

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
