import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# Load model YOLOv8
model = YOLO('traffic_sign_yolov8.pt')  # Thay bằng đường dẫn tới file model của bạn

st.title("Phát hiện biển báo giao thông từ ảnh")

# Tải ảnh lên
uploaded_file = st.file_uploader("Chọn một ảnh", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc ảnh
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Resize ảnh về 640x640 để xử lý
    image = cv2.resize(image, (640, 640))

    # Thực hiện dự đoán
    results = model(image)

    # Vẽ kết quả
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            label = model.names[class_id]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{label} {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Hiển thị ảnh
    st.image(image, channels="BGR", caption="Kết quả phát hiện", use_column_width=True)

    # Lưu ảnh
    cv2.imwrite("output_image.jpg", image)
    st.success("Ảnh đã được lưu thành output_image.jpg")