from ultralytics import YOLO
import cv2
import os

# =========================
# 設定
# =========================
VIDEO_PATH = "videos/input/sample.mov"       # 入力動画
MODEL_PATH = YOLO("yolov8n.pt")    # 顔検出用のYOLO重み
OUTPUT_DIR = "output_frames"   # 保存先フォルダ
FRAME_INTERVAL_SEC = 1         # 何秒ごとに1枚抜くか

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# モデル読み込み
# =========================
model = YOLO(MODEL_PATH)

# =========================
# 動画読み込み
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"動画を開けませんでした: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    raise RuntimeError("FPSを取得できませんでした")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_sec = frame_count / fps

print(f"FPS: {fps}")
print(f"総フレーム数: {frame_count}")
print(f"動画の長さ(秒): {duration_sec:.2f}")

# 何フレームごとに抽出するか
frame_interval = int(fps * FRAME_INTERVAL_SEC)
if frame_interval <= 0:
    frame_interval = 1

print(f"{FRAME_INTERVAL_SEC}秒ごと -> {frame_interval}フレーム間隔で抽出")

# =========================
# フレーム抽出＋顔検出
# =========================
current_frame_idx = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1秒ごとに抽出
    if current_frame_idx % frame_interval == 0:
        timestamp_sec = current_frame_idx / fps

        # 顔検出
        results = model(frame, verbose=False)

        detected = 0
        annotated_frame = frame.copy()

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # 顔枠を描画
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_frame,
                    f"face {conf:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
                detected += 1

        # 保存ファイル名
        if detected > 0:
            if conf >= 0.7:
                output_name = f"frame_{saved_count:04d}_t{timestamp_sec:.1f}s_faces{detected}.jpg"
                output_path = os.path.join(OUTPUT_DIR, output_name)
                cv2.imwrite(output_path, annotated_frame)
                print(f"保存: {output_path}")
                saved_count += 1
    current_frame_idx += 1

cap.release()
print(f"完了: {saved_count}枚保存しました")