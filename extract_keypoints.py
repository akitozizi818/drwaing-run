import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def opencv_keypoints(image_path, min_points=6, max_points=9, point_size=20):
    # 画像読み込み
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 二値化
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # 輪郭検出
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 最大の輪郭を取得
    contour = max(contours, key=cv2.contourArea)
    
    # epsilon初期値を計算
    epsilon = 0.01 * cv2.arcLength(contour, True)
    
    # 試行する値の範囲
    epsilon_values = []
    for mult in np.logspace(-2, 1, 30):  # より細かい範囲で多くの値を試す
        epsilon_values.append(mult * cv2.arcLength(contour, True))
    
    # 各epsilonでの点の数を計算
    point_counts = []
    for eps in epsilon_values:
        approx = cv2.approxPolyDP(contour, eps, True)
        point_counts.append(len(approx))
    
    # 10個以上で最も10に近い点数を見つける
    closest_to_10 = float('inf')
    closest_idx = 0
    
    for i, count in enumerate(point_counts):
        if count >= 10 and abs(count - 10) < closest_to_10:
            closest_to_10 = abs(count - 10)
            closest_idx = i
    
    # 10個以上の点が見つからない場合は、最大の点数を選択
    if closest_to_10 == float('inf'):
        closest_idx = np.argmax(point_counts)
    
    # 選択したepsilonで輪郭の近似を行う
    epsilon = epsilon_values[closest_idx]
    approx = cv2.approxPolyDP(contour, epsilon, True)
    keypoints = [point[0] for point in approx]
    
    # 重心を計算
    center_point = np.mean(keypoints, axis=0)
    
    # 各点を角度でソート
    angles = []
    for point in keypoints:
        # 中心からの相対座標
        dx = point[0] - center_point[0]
        dy = point[1] - center_point[1]
        # 角度を計算（-π〜πの範囲）
        angle = np.arctan2(dy, dx)
        angles.append(angle)
    
    # 角度でソート
    sorted_indices = np.argsort(angles)
    keypoints = [keypoints[i] for i in sorted_indices]
    angles = [angles[i] for i in sorted_indices]
    
    # 点の対称性を判定
    # 左右対称: x座標の差が小さい
    # 上下対称: y座標の差が小さい
    symmetric_pairs = []
    
    for i in range(len(keypoints)):
        for j in range(i+1, len(keypoints)):
            p1 = np.array(keypoints[i])
            p2 = np.array(keypoints[j])
            
            # 中心を通る直線に対して対称かをチェック
            # 中心から各点へのベクトル
            v1 = p1 - center_point
            v2 = p2 - center_point
            
            # ベクトルの長さがほぼ同じで、角度が反対（差がπに近い）
            length_ratio = np.linalg.norm(v1) / np.linalg.norm(v2) if np.linalg.norm(v2) > 0 else float('inf')
            angle_diff = abs(abs(angles[i] - angles[j]) - np.pi)
            
            # 対称性の条件：長さの比が0.8〜1.25の範囲、角度差がπ±0.3ラジアン
            if 0.8 < length_ratio < 1.25 and angle_diff < 0.3:
                symmetric_pairs.append((i, j))
    
    # 対称ペアをスコア付け（優先度）
    pair_scores = []
    for i, j in symmetric_pairs:
        p1 = np.array(keypoints[i])
        p2 = np.array(keypoints[j])
        
        # 対称度（角度差がπに近いほど高スコア）
        angle_diff = abs(abs(angles[i] - angles[j]) - np.pi)
        angle_score = 1 - angle_diff / 0.3
        
        # 長さの比（1に近いほど高スコア）
        v1 = p1 - center_point
        v2 = p2 - center_point
        length_ratio = np.linalg.norm(v1) / np.linalg.norm(v2) if np.linalg.norm(v2) > 0 else float('inf')
        length_score = 1 - abs(length_ratio - 1) / 0.25
        
        # 重要度（中心からの距離）
        importance = (np.linalg.norm(v1) + np.linalg.norm(v2)) / 2
        
        # 総合スコア
        score = angle_score * 0.5 + length_score * 0.3 + importance * 0.2
        pair_scores.append((i, j, score))
    
    # スコアの高い順にソート
    pair_scores.sort(key=lambda x: x[2], reverse=True)
    
    # 保持する点のリスト（インデックス）
    keep_indices = set()
    
    # 対称ペアをスコア順に追加
    for i, j, _ in pair_scores:
        keep_indices.add(i)
        keep_indices.add(j)
        # 点の数が上限に達したら終了
        if len(keep_indices) >= max_points:
            break
    
    # もし点の数が少なすぎる場合、残りのなるべく対称的な点を追加
    if len(keep_indices) < min_points:
        remaining = [i for i in range(len(keypoints)) if i not in keep_indices]
        # 中心からの距離でソート
        remaining.sort(key=lambda i: np.linalg.norm(np.array(keypoints[i]) - center_point), reverse=True)
        
        # 必要な数だけ追加
        for i in remaining:
            keep_indices.add(i)
            if len(keep_indices) >= min_points:
                break
    
    # 最終的な特徴点リスト
    filtered_keypoints = [keypoints[i] for i in sorted(keep_indices)]
    
    # もう一度角度でソート
    angles = []
    for point in filtered_keypoints:
        dx = point[0] - center_point[0]
        dy = point[1] - center_point[1]
        angle = np.arctan2(dy, dx)
        angles.append(angle)
    
    sorted_indices = np.argsort(angles)
    filtered_keypoints = [filtered_keypoints[i] for i in sorted_indices]
    
    # 結果を可視化
    result_img = img.copy()
    
    # 中心点を表示
    center_tuple = (int(center_point[0]), int(center_point[1]))
    cv2.circle(result_img, center_tuple, point_size // 2, (0, 255, 0), -1)
    
    # 線で点を接続
    for i in range(len(filtered_keypoints)):
        # 点を描画
        cv2.circle(result_img, tuple(map(int, filtered_keypoints[i])), point_size, (0, 0, 255), -1)
        
        # 次の点と線で接続
        next_idx = (i + 1) % len(filtered_keypoints)
        cv2.line(result_img, tuple(map(int, filtered_keypoints[i])), 
                 tuple(map(int, filtered_keypoints[next_idx])), (255, 255, 255), 2)
    
    return filtered_keypoints, result_img

def process_image(image_path, save_dir, min_points=6, max_points=9, point_size=30):
    # 特徴点抽出
    keypoints, result_img = opencv_keypoints(image_path, min_points, max_points, point_size)
    
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
    plt.title(f"特徴点: {base_name} ({len(keypoints)}点)")
    plt.axis('off')
    
    # 画像を保存
    plt.savefig(result_filepath, bbox_inches='tight', dpi=300)
    plt.close()  # メモリリークを防ぐために図を閉じる
    
    # 特徴点の座標を保存（CSVファイル）
    keypoints_filename = f"{base_name}_keypoints_{timestamp}.csv"
    keypoints_filepath = os.path.join(save_dir, keypoints_filename)
    np.savetxt(keypoints_filepath, keypoints, delimiter=',', fmt='%d', header='x,y')
    
    print(f"処理完了: {base_name}")
    print(f"  特徴点数: {len(keypoints)}")
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
                process_image(image_path, save_dir, min_points=6, max_points=9)
                processed_count += 1
            except Exception as e:
                print(f"エラー: {filename} の処理中にエラーが発生しました: {str(e)}")
    
    print(f"\n処理完了: {processed_count}個のファイルを処理しました")
    
    # すべての処理完了後に結果のサマリーを表示
    if processed_count > 0:
        print(f"すべての結果は {save_dir} ディレクトリに保存されています")
    else:
        print(f"処理された画像はありませんでした。{image_dir} 内に適切な画像ファイルがあることを確認してください")