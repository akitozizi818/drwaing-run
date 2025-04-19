import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def opencv_keypoints(image_path, num_points=5, point_size=20):
    # 画像読み込み
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 二値化
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # 輪郭検出
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 最大の輪郭を取得
    contour = max(contours, key=cv2.contourArea)
    
    # 輪郭を単純化（Douglas-Peuckerアルゴリズム）
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # 特徴点を適切な数に間引く
    if len(approx) > num_points:
        indices = np.linspace(0, len(approx) - 1, num_points, dtype=int)
        keypoints = [approx[i][0] for i in indices]
    else:
        keypoints = [point[0] for point in approx]
    
    # 結果を可視化
    result_img = img.copy()
    for point in keypoints:
        cv2.circle(result_img, tuple(point), point_size, (0, 0, 255), -1)
    
    return keypoints, result_img

def process_image(image_path, save_dir, num_points=6, point_size=30):
    # 特徴点抽出
    keypoints, result_img = opencv_keypoints(image_path, num_points, point_size)
    
    # 元のファイル名から拡張子を除いた部分を取得
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 現在の日時を取得してファイル名に使用
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存ファイル名を生成
    result_filename = f"{base_name}_keypoints_{timestamp}.png"
    result_filepath = os.path.join(save_dir, result_filename)
    
    # Matplotlibを使用して画像を表示・保存
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title(f"特徴点: {base_name}")
    plt.axis('off')
    
    # 画像を保存
    plt.savefig(result_filepath, bbox_inches='tight', dpi=300)
    plt.close()  # メモリリークを防ぐために図を閉じる
    
    # 特徴点の座標を保存（CSVファイル）
    keypoints_filename = f"{base_name}_keypoints_{timestamp}.csv"
    keypoints_filepath = os.path.join(save_dir, keypoints_filename)
    np.savetxt(keypoints_filepath, keypoints, delimiter=',', fmt='%d', header='x,y')
    
    print(f"処理完了: {base_name}")
    print(f"  結果画像: {result_filepath}")
    print(f"  特徴点座標: {keypoints_filepath}")
    
    return keypoints

# メイン実行部分
if __name__ == "__main__":
    # 画像ファイルのディレクトリを指定
    image_dir = "imgs"
    
    # 保存先ディレクトリを設定
    save_dir = "keypoints_results"
    
    # 画像ファイル拡張子のリスト
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"ディレクトリを作成しました: {save_dir}")
    
    # 画像ディレクトリが存在するか確認
    if not os.path.exists(image_dir):
        print(f"エラー: 指定された画像ディレクトリが存在しません: {image_dir}")
        exit(1)
    
    # 処理した画像の数をカウント
    processed_count = 0
    
    # ディレクトリ内のすべてのファイルを処理
    for filename in os.listdir(image_dir):
        # 拡張子をチェック
        ext = os.path.splitext(filename)[1].lower()
        if ext in image_extensions:
            image_path = os.path.join(image_dir, filename)
            try:
                # 画像処理
                process_image(image_path, save_dir)
                processed_count += 1
            except Exception as e:
                print(f"エラー: {filename} の処理中にエラーが発生しました: {str(e)}")
    
    print(f"\n処理完了: {processed_count}個のファイルを処理しました")
    
    # すべての処理完了後に結果のサマリーを表示
    if processed_count > 0:
        print(f"すべての結果は {save_dir} ディレクトリに保存されています")
    else:
        print(f"処理された画像はありませんでした。{image_dir} 内に適切な画像ファイルがあることを確認してください")