# Import các thư viện cần thiết
import cv2  # Thư viện OpenCV cho xử lý ảnh và video
import mediapipe as mp  # Thư viện MediaPipe cho việc nhận dạng điểm mốc trên tay
import os  # Thư viện hệ thống để làm việc với tệp và thư mục
import time  # Thư viện thời gian

# Khởi tạo các đối tượng Mediapipe Hands
mp_drawing = mp.solutions.drawing_utils  # Đối tượng để vẽ các điểm mốc tay
mp_hands = mp.solutions.hands  # Đối tượng Mediapipe Hands

# Khởi tạo đối tượng Mediapipe Hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
)

# Khởi tạo webcam
cap = cv2.VideoCapture(0)

# Biến lưu trạng thái có lưu dữ liệu không
saving = False

# Biến lưu tên của hình
current_name = "hi"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển đổi khung hình thành ảnh màu RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Nhận diện các điểm mốc trên tay
    results = hands.process(frame_rgb)

    # Vẽ các điểm mốc của các ngón tay và lưu dữ liệu
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Vẽ các điểm mốc trên tay
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Nếu đang trong trạng thái lưu
            if saving:
                if current_name:
                    # Tạo thư mục con nếu chưa tồn tại
                    folder_name = os.path.join("hand_shape_data", current_name)
                    os.makedirs(folder_name, exist_ok=True)

                    # Đếm số tệp trong folder con
                    files = os.listdir(f"hand_shape_data/{current_name}")

                    # Lưu lại tọa độ của các điểm mốc tay và ghi vào file trong thư mục con
                    filename = os.path.join(
                        folder_name, f"{current_name}_hand_shape_{len(files)}.txt"
                    )
                    with open(filename, "w") as file:
                        for landmark in hand_landmarks.landmark:
                            file.write(f"{landmark.x} {landmark.y} {landmark.z}\n")
                        print(f"Saved: {filename}")

                saving = False

            # Hiển thị kết quả cử chỉ tay bên cạnh tay
            if current_name:
                cv2.putText(
                    frame,
                    f"Hand Gesture: {current_name}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

    cv2.imshow("Hand Tracking", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        # Nhấn 's' để bắt đầu lưu và yêu cầu nhập tên
        # current_name = input("Enter a name for the next hand shape: ")
        saving = True

    # Thoát nếu nhấn 'q'
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
