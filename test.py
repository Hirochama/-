import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# データ読み込みと前処理
df = pd.read_csv("データ.csv")
df = df.fillna(0)

# シーン列をバイナリ列に変換
df["普段使い"] = df["シーン"].apply(lambda x: 1 if "普段使い" in x else 0)
df["ビジネス"] = df["シーン"].apply(lambda x: 1 if "ビジネス" in x else 0)
df["切り替え"] = df["シーン"].apply(lambda x: 1 if "切り替え" in x else 0)
df["特別な時"] = df["シーン"].apply(lambda x: 1 if "特別な時" in x else 0)
df = df.drop(columns=["シーン"])

# 性別列をバイナリ列に変換
df["男性"] = df["性別"].apply(lambda x: 1 if x in ["男性", "ユニセックス"] else 0)
df["女性"] = df["性別"].apply(lambda x: 1 if x in ["女性", "ユニセックス"] else 0)
df = df.drop(columns=["性別"])

# 性別列を数値に変換
df["濃度"] = df["濃度"].map({"EDT": 5, "EDP": 10, "EDC": 1})

# 標準化
columns_to_scale = df.loc[:, '価格':].columns
scaler = StandardScaler()
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# K-Means クラスタリング
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(df.loc[:, '価格':])

# クラスタリング結果の確認
cluster_summary = df.groupby('Cluster').mean()

#入力データに基づき、リコメンド結果を返す関数

def recommend(input_data, df, feature_columns):
   
    # クラスタ予測
    cluster_label = kmeans.predict([input_data])[0]
    print(f"\n入力データはクラスタ {cluster_label} に属しています。\n")
    
    # 同じクラスタのデータを抽出
    similar_items = df[df['Cluster'] == cluster_label]
    
    # 類似度計算
    similarities = cosine_similarity([input_data], similar_items[feature_columns])
    similar_items = similar_items.copy()
    similar_items['Similarity'] = similarities[0]
    
    # 類似度の降順でソートして上位を返す
    recommendations = similar_items.sort_values(by='Similarity', ascending=False)
    return recommendations

# ユーザー入力関数 
def get_user_input():
    print("\nリコメンドシステムへようこそ!以下の項目を入力してください。\n")

    # 価格を入力
    price = float(input("予算を入力してください（例: 10000）: "))

    # 好みの香りをカンマ区切りで入力
    fragrance = input("好みの香りをカンマ区切りで入力してください（種類: フルーティ, ホワイトフローラル, フローラル, ウッディ, シトラス, ムスク, グリーンフローラル, アロマティック, アンバー, アクア, アニマリック, パウダリー, グルマン, グリーン, スパイシー, オリエンタル, ハーバル）: ").lower().split(',')
    fragrance_mapping = {
        "フルーティ": 1, "ホワイトフローラル": 2, "フローラル": 3, "ウッディ": 4,
        "シトラス": 5, "ムスク": 6, "グリーンフローラル": 7, "アロマティック": 8,
        "アンバー": 9, "アクア": 10, "アニマリック": 11, "パウダリー": 12,
        "グルマン": 13, "グリーン": 14, "スパイシー": 15, "オリエンタル": 16,
        "ハーバル": 17
    }
    fragrance_flags = {key: (1 if key in [f.strip() for f in fragrance] else 0) for key in fragrance_mapping.keys()}

    # 濃度を入力（1～10の間）
    while True:
        concentration = float(input("香りの強さを1～10の間で入力してください（例: 5）: "))
        if 1 <= concentration <= 10:
            break
        else:
            print("濃度は1～10の範囲で入力してください。")

    # 季節を入力
    season = input("好きな季節をカンマ区切りで入力してください（例: 春, 夏, 秋, 冬）: ").lower().split(',')
    season_flags = {
        "春": 1 if "春" in season else 0,
        "夏": 1 if "夏" in season else 0,
        "秋": 1 if "秋" in season else 0,
        "冬": 1 if "冬" in season else 0,
    }

    # 使用シーンを入力
    scene = input("使用シーンをカンマ区切りで入力してください（例: 普段使い, ビジネス, 特別な時，切り替え）: ").lower().split(',')
    scene_flags = {
        "普段使い": 1 if "普段使い" in scene else 0,
        "ビジネス": 1 if "ビジネス" in scene else 0,
        "切り替え": 1 if "切り替え" in scene else 0,
        "特別な時": 1 if "特別な時" in scene else 0,
    }
    
    # 性別を入力
    gender = input("性別を入力してください（男性/女性/ユニセックス）: ").strip()
    male = 1 if gender in ["男性", "ユニセックス"] else 0
    female = 1 if gender in ["女性", "ユニセックス"] else 0

    # データフレームの列に基づいて入力データを作成
    user_input_dict = {
        "価格": price,
        **fragrance_flags,  # 香りフラグ
        "濃度": concentration,
        **season_flags,     # 季節フラグ
        **scene_flags,      # 使用シーンフラグ
        "男性": male,
        "女性": female
    }
    user_input = [
        price,                       #価格
        *fragrance_flags.values(),  # 香りフラグ
        concentration,               #強さ
        *season_flags.values(),     # 季節フラグ
        *scene_flags.values(),      # 使用シーンフラグ
        male,
        female
    ]
    print(user_input)
    return user_input

# 修正済みリコメンド実行部分

feature_columns = df.select_dtypes(include=[np.number]).columns[:-1]

user_input = get_user_input()
scaled_user_input = scaler.transform([user_input])[0]  # 標準化
print(scaled_user_input)
# リコメンド実行
recommendations = recommend(
    input_data=scaled_user_input,
    df=df,
    feature_columns=feature_columns
)

# 表形式で結果を表示する関数
def display_recommendations(recommendations, top_n=5):
    # 必要な列を抽出
    recommendations_clean = recommendations[['ブランド名', '香水名']].head(top_n).reset_index(drop=True)
    # 順位列を追加
    recommendations_clean.insert(0, 'オススメ順', range(1, len(recommendations_clean) + 1))
    # テーブル形式で表示
    from IPython.display import display
    display(recommendations_clean.style.set_caption(f"リコメンド結果（上位 {top_n}）")
                                     .hide_index()
                                     .set_properties(**{'text-align': 'left'})
                                     .set_table_styles([
                                         {'selector': 'th', 'props': [('text-align', 'left')]}
                                     ]))

# 上位5つを表示
#print("\nリコメンド結果（ブランド名と香水名のみ、順位付き、上位5件）:")
display_recommendations(recommendations, top_n=5)