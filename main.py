import cv2
import numpy as np
import os
from ultralytics import YOLO
from tensorflow.keras.models import load_model

class DrowsinessDetector:
    """
    Function Phát hiện buồn ngủ dựa trên các mô hình học máy và phát hiện khuôn mặt.
    """

    def __init__(self, model_dir="C:/Users/ACER/Desktop/I/Models"):
       
        #Khởi tạo với các mô hình.
        
        self.face_model_path = os.path.join(model_dir, "yolov8-face.pt")
        self.face_model = self.load_model(self.face_model_path)

        # Mô hình CNN cho mắt và miệng
        self.eye_model = self.load_keras_model(os.path.join(model_dir, "eye_detector.h5"))
        self.mouth_model = self.load_keras_model(os.path.join(model_dir, "mouth_detector.h5"))

        # Haar cascades
        self.eye_cascade = cv2.CascadeClassifier(os.path.join(model_dir, "haarcascade_eye.xml"))
        self.mouth_cascade = cv2.CascadeClassifier(os.path.join(model_dir, "haarcascade_mcs_mouth.xml"))

        self.colors = {
            'face': (255, 0, 0),
            'eye': (0, 0, 255),
            'mouth': (0, 255, 255)
        }
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.eye_closed_frames = 0
        self.drowsy_threshold = 5

    def load_model(self, model_path):
        """
        Tải mô hình YOLO .
        """
        if os.path.exists(model_path):
            return YOLO(model_path, task='detect')
        else:
            raise FileNotFoundError(f"Model file '{model_path}' not found.")

    def load_keras_model(self, model_path):
        """
        Tải mô hình Keras từ tệp.
        param model_path: Đường dẫn đến tệp mô hình.

        """
        return load_model(model_path)

    def detect_face(self, frame):
        """
        Phát hiện khuôn mặt trong khung hình .
        param frame: Khung hình đầu vào.
        Trả vềề Vùng khuôn mặt và tọa độ (x1, y1, x2, y2) của khuôn mặt.
        """
        face_results = self.face_model(frame, conf=0.5, verbose=False)
        for face in face_results:
            if not hasattr(face, 'boxes') or not face.boxes:
                continue

            for box in face.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                face_roi = frame[y1:y2, x1:x2]
                return face_roi, (x1, y1, x2, y2)
        return None, None

    def detect_eyes(self, face_roi, gray_face):
        """
        TTrạng thái mắt.
        param face_roi: Vùng khuôn mặt đã được cắt ra.
        param gray_face: Hình ảnh khuôn mặt ở dạng xám.
        Trả về Trạng thái của mắt "Opened" hoặc "Closed".
        """
        eye_status = "Closed"
        eye_region = face_roi[:face_roi.shape[0] // 2, :]
        eyes = self.eye_cascade.detectMultiScale(cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5)

        if len(eyes) > 0:
            (ex, ey, ew, eh) = eyes[0]
            eye_region = face_roi[ey:ey + eh, ex:ex + ew]
            eye_roi = cv2.resize(eye_region, (64, 64))
            eye_roi = np.expand_dims(eye_roi, axis=0) / 255.0
            eye_pred = self.eye_model.predict(eye_roi)[0][0]
            eye_status = "Opened" if eye_pred >= 0.7 else "Closed"
        return eye_status

    def detect_mouth(self, face_roi, gray_face):
        """
        Trạng Thái miệng
        param face_roi: Vùng khuôn mặt đã được cắt ra.
        param gray_face: Hình ảnh khuôn mặt ở dạng xám.
        Trả về: Trạng thái của miệng "Yawn" hoặc "Normal".
        """
        mouth_status = "Normal"
        mouth_region = face_roi[face_roi.shape[0] // 3:, :]
        mouths = self.mouth_cascade.detectMultiScale(cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5)

        if len(mouths) > 0:
            (mx, my, mw, mh) = mouths[0]
            mouth_region = face_roi[my:my + mh, mx:mx + mw]
            mouth_roi = cv2.resize(mouth_region, (64, 64))
            mouth_roi = np.expand_dims(mouth_roi, axis=0) / 255.0
            mouth_pred = self.mouth_model.predict(mouth_roi)[0][0]
            mouth_open_ratio = mw / mh
            if mouth_open_ratio > 0.4 and mouth_pred >= 0.4:
                mouth_status = "Yawn"
        return mouth_status

    def detect_drowsiness(self, frame):
        """
        param frame: Khung hình đầu vào.
        Trả về: Khung hình đã được vẽ các thông tin phát hiện.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_roi, bbox = self.detect_face(frame)

        if face_roi is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['face'], 2)

            eye_status = self.detect_eyes(face_roi, gray[y1:y2, x1:x2])
            mouth_status = self.detect_mouth(face_roi, gray[y1:y2, x1:x2])

            if eye_status == "Closed":
                self.eye_closed_frames += 1
            else:
                self.eye_closed_frames = 0

            drowsy = (self.eye_closed_frames >= self.drowsy_threshold) or (mouth_status == "Yawn")
            status = "DROWSY" if drowsy else "AWAKE"
            status_color = (0, 0, 255) if drowsy else (0, 255, 0)

            cv2.putText(frame, f"Eyes: {eye_status}", (x1, y1 - 30),
                        self.font, 0.6, self.colors['eye'], 2)
            cv2.putText(frame, f"Mouth: {mouth_status}", (x1, y1 - 60),
                        self.font, 0.6, self.colors['mouth'], 2)
            cv2.putText(frame, f"Status: {status}", (x1, y1 - 90),
                        self.font, 0.7, status_color, 2)

        return frame

def main():
  
    #khởi tạo hệ thống và thực hiện phát hiện tình trạng buồn ngủ.
    
    print("Initializing drowsiness detection system...")

    model_dir = os.getenv("MODEL_DIR", "C:/Users/ACER/Desktop/I/Models")

    try:
        # detector
        detector = DrowsinessDetector(model_dir=model_dir)

        print("Opening camera...")
        cap = cv2.VideoCapture(0)  
        if not cap.isOpened():
            raise ValueError("Error: Could not open any camera.")
        
        print("Starting detection (press 'q' to quit)...")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Frame read error")
                break

            frame = cv2.flip(frame, 1)
            result_frame = detector.detect_drowsiness(frame)

            cv2.imshow("Drowsiness Detection", result_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("System shutdown.")

if __name__ == "__main__":
    main()
