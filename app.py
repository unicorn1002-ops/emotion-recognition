from flask import Flask, render_template, Response, jsonify, request
import cv2
import joblib
import numpy as np
from skimage.feature import local_binary_pattern
import dlib
from itertools import combinations

# 初始化 Flask
app = Flask(__name__)

# 參數設置
RADIUS = 4
N_POINTS = 8 * RADIUS
EMOTIONS = ['生氣', '厭惡', '害怕', '開心', '面無表情', '難過', '驚訝']

# 模型與地標檢測器
MODEL_PATH = "ensemble_model.joblib"
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# 載入模型和地標檢測器
try:
    print("嘗試載入模型...")
    model = joblib.load(MODEL_PATH)
    print("模型載入成功")
except Exception as e:
    print(f"模型載入失敗: {e}")
    model = None

try:
    print("初始化地標檢測器...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    print("地標檢測器初始化成功")
except Exception as e:
    print(f"地標檢測器初始化失敗: {e}")
    detector, predictor = None, None

# 特徵計算函數
def compute_features(gray_eq, landmarks):
    """計算完整特徵，包括 LBP、地標點和局部輪廓特徵。"""
    lbp = local_binary_pattern(gray_eq, N_POINTS, RADIUS, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, N_POINTS + 3), density=True)

    landmarks_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)])
    landmarks_flattened = landmarks_points.flatten()

    l2_distances = [np.linalg.norm(p1 - p2) for p1, p2 in combinations(landmarks_points, 2)]

    mouth_points = landmarks_points[48:68]
    mouth_perimeter = cv2.arcLength(mouth_points.reshape((-1, 1, 2)), True)
    mouth_area = cv2.contourArea(mouth_points.reshape((-1, 1, 2)))

    left_eye_points = landmarks_points[36:42]
    left_eye_perimeter = cv2.arcLength(left_eye_points.reshape((-1, 1, 2)), True)
    left_eye_area = cv2.contourArea(left_eye_points.reshape((-1, 1, 2)))

    right_eye_points = landmarks_points[42:48]
    right_eye_perimeter = cv2.arcLength(right_eye_points.reshape((-1, 1, 2)), True)
    right_eye_area = cv2.contourArea(right_eye_points.reshape((-1, 1, 2)))

    local_contour_features = np.array([
        mouth_perimeter, mouth_area,
        left_eye_perimeter, left_eye_area,
        right_eye_perimeter, right_eye_area
    ])

    return np.concatenate([lbp_hist, landmarks_flattened, l2_distances, local_contour_features])

@app.route('/')
def index():
    """返回首頁模板"""
    return render_template("index.html")

@app.route('/capture', methods=['POST'])
def capture():
    """從前端傳來的圖片進行情緒識別"""
    try:
        file = request.files['image']
        file_bytes = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray_image)
        faces = detector(gray_eq)

        if len(faces) == 0:
            return jsonify({"error": "未檢測到人臉"}), 400

        landmarks = predictor(gray_eq, faces[0])
        features = compute_features(gray_eq, landmarks).reshape(1, -1)

        emotion_index = model.predict(features)[0]
        emotion = EMOTIONS[emotion_index]
        return jsonify({"emotion": emotion})
    except Exception as e:
        print(f"捕捉畫面時發生錯誤: {e}")
        return jsonify({"error": "捕捉畫面時發生未知錯誤"}), 500

if __name__ == "__main__":
    try:
        print("啟動 Flask 應用程式...")
        app.run(debug=False, host="0.0.0.0", port=5000)
    except Exception as e:
        print(f"Flask 啟動失敗: {e}")
