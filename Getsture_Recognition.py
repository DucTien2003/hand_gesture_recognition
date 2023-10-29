import cv2  # Thư viện OpenCV cho xử lý ảnh và video
import mediapipe as mp  # Thư viện MediaPipe cho việc nhận dạng điểm mốc trên tay
import os  # Thư viện hệ thống để làm việc với tệp và thư mục
import numpy as np  # Thư viện NumPy cho xử lý dữ liệu số
import matplotlib.pyplot as plt  # Thư viện đồ thị
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split,
)  # Để chia dữ liệu thành tập huấn luyện và tập kiểm tra
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)  # Để đo độ chính xác của mô hình

# Khởi tạo đối tượng Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
)
# static_image_mode: ché độ ảnh tĩnh


# Hàm để đọc dữ liệu từ các tệp và thư mục con
def load_data_from_folders(data_folder):
    gestures = []  # Danh sách chứa dữ liệu tay
    labels = []  # Danh sách chứa nhãn tương ứng

    # Duyệt qua tất cả thư mục trong thư mục dữ liệu
    for folder_name in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, folder_name)
        if os.path.isdir(folder_path):
            # Duyệt qua tất cả tệp tin văn bản trong thư mục con
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".txt"):
                    # Hàm os.path.join sẽ nối hai thành phần này lại với nhau
                    # và tạo ra một đường dẫn đầy đủ đến tệp tin file_name
                    # bên trong thư mục folder_path.
                    # Kết quả sẽ là một chuỗi đại diện cho đường dẫn tới tệp tin đó.
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, "r") as file:
                        data = []
                        # Duyệt qua từng dòng trong tệp và chuyển thành mảng dữ liệu
                        for line in file:
                            x, y, z = map(float, line.strip().split())
                            data.extend([x, y, z])
                        gestures.append(data)  # Thêm dữ liệu vào danh sách tay
                        labels.append(folder_name)  # Thêm nhãn vào danh sách

    return np.array(gestures), np.array(labels)


# Đường dẫn đến thư mục chứa dữ liệu
data_folder = "hand_shape_data"

# Đọc dữ liệu từ thư mục và các thư mục con
gestures, labels = load_data_from_folders(data_folder)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra Perceptron
# test_size=0.2 có nghĩa là 20% của dữ liệu sẽ được sử dụng cho tập kiểm tra và 80% còn lại sẽ được sử dụng cho tập huấn luyện.
# 42 là Lucky number
X_train, X_test, y_train, y_test = train_test_split(
    gestures, labels, test_size=0.3, random_state=42
)


# Huấn luyện mô hình SVM với kernel tuyến tính với C là tham số vùng an toàn
model = SVC(kernel="linear", C=1)
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Khởi tạo webcam
cap = cv2.VideoCapture(0)

is_window_closed = False

while cap.isOpened() and not is_window_closed:
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển đổi khung hình thành ảnh màu RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Nhận diện các điểm mốc trên tay
    results = hands.process(frame_rgb)

    # Vẽ các điểm mốc của các ngón tay và dự đoán cử chỉ tay
    if results.multi_hand_landmarks:
        # i là index, nếu xóa thì han_landmarks sẽ trở thành index
        # hand_landmarks là từng điểm trên bàn tay
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Vẽ các điểm mốc trên tay
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Chuyển dữ liệu điểm mốc thành mảng numpy
            data = []
            for landmark in hand_landmarks.landmark:
                data.extend([landmark.x, landmark.y, landmark.z])
            data = np.array(data).reshape(1, -1)

            # Dự đoán cử chỉ tay
            predicted_gesture = model.predict(data)[0]

            # Hiển thị kết quả dự đoán lên khung hình
            cv2.putText(
                frame,
                f"Gesture: {predicted_gesture}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

    cv2.imshow("Hand Gesture Recognition", frame)

    key = cv2.waitKey(1)
    if (
        key == ord("x") or key == ord("X") or key == 27
    ):  # Kiểm tra khi người dùng nhấn X hoặc phím Escape (ESC)
        break

    if cv2.getWindowProperty("Hand Gesture Recognition", cv2.WND_PROP_VISIBLE) < 1:
        is_window_closed = True

# Tính toán và in ra độ chính xác của từng loại cử chỉ trên tập kiểm tra
y_pred_test = model.predict(X_test)
report = classification_report(y_test, y_pred_test, output_dict=True)

accuracies = []  # Danh sách để lưu trữ độ chính xác của từng loại cử chỉ
gesture_names = list(set(y_test))  # Tên các loại cử chỉ trong tập test

for gesture in gesture_names:
    acc = report[gesture][
        "precision"
    ]  # Lấy độ chính xác (precision) của từng loại cử chỉ
    accuracies.append(acc)

# Hiển thị đồ thị về độ chính xác của từng loại cử chỉ trên tập kiểm tra
plt.figure(figsize=(10, 6))
plt.bar(gesture_names, accuracies, color="skyblue")
plt.xlabel("Gesture Name")
plt.ylabel("Accuracy")
plt.title("Accuracy for each gesture in Test Set")
# plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

cap.release()
cv2.destroyAllWindows()
