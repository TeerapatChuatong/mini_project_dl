import os
import pickle
from typing import Optional, Tuple, List, Dict

import face_recognition
import numpy as np
import config


class FaceSystem:
    def __init__(self):
        self.encodings_path = config.FACE_ENCODINGS_PATH
        self.tolerance = config.FACE_RECOGNITION_TOLERANCE

        os.makedirs(self.encodings_path, exist_ok=True)

        self.hat_model = None
        self.hat_model_error = None
        self.model_path = self._resolve_model_path()

        # โหลดโมเดลแบบปลอดภัย
        self._load_hat_model()

    def _resolve_model_path(self) -> str:
        """
        หา path ของโมเดลหมวก:
        1) best.pt ในโฟลเดอร์ backend
        2) best.pt ใน current working directory
        3) yolo11n.pt
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))

        candidates = [
            os.path.join(current_dir, "best.pt"),
            os.path.join(os.getcwd(), "best.pt"),
            "yolo11n.pt",
        ]

        for path in candidates:
            if os.path.exists(path):
                return path

        # fallback
        return "yolo11n.pt"

    def _load_hat_model(self):
        """
        โหลด YOLO แบบปลอดภัย
        ถ้า torch/ultralytics มีปัญหา จะไม่ทำให้แอปล้มตั้งแต่ import
        """
        try:
            from ultralytics import YOLO

            print(f"🔄 กำลังโหลดโมเดลหมวก: {self.model_path}")
            self.hat_model = YOLO(self.model_path)
            self.hat_model_error = None
            print("✅ โหลดโมเดลหมวกสำเร็จ")
        except Exception as e:
            self.hat_model = None
            self.hat_model_error = str(e)
            print(f"❌ โหลดโมเดลหมวกไม่สำเร็จ: {e}")

    def is_hat_model_ready(self) -> bool:
        return self.hat_model is not None

    def get_hat_model_error(self) -> Optional[str]:
        return self.hat_model_error

    def detect_objects_and_hat(self, image_array: np.ndarray) -> Tuple[bool, List[Dict]]:
        """
        ตรวจจับเฉพาะหมวก และคืนค่าพิกัดกรอบ
        ถ้าโมเดลยังไม่พร้อม จะคืน False, []
        """
        if self.hat_model is None:
            return False, []

        try:
            results = self.hat_model(image_array, verbose=False)[0]
            detected_boxes = []
            hat_detected = False

            for box in results.boxes:
                class_id = int(box.cls[0])
                original_label = str(results.names[class_id])

                clean_label = (
                    original_label.lower()
                    .replace("-", " ")
                    .replace("_", " ")
                    .strip()
                )

                confidence = float(box.conf[0])
                coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

                # กรองความมั่นใจขั้นต่ำ
                if confidence < 0.30:
                    continue

                # ตรวจคำที่สื่อว่าเป็นหมวก
                hat_keywords = ["cap", "helmet", "baseball", "hat"]
                is_hat = any(keyword in clean_label for keyword in hat_keywords)

                if not is_hat:
                    continue

                hat_detected = True

                print(
                    f"   -> 🟢 เจอหมวก: '{original_label}' "
                    f"(ความมั่นใจ: {confidence * 100:.2f}%)"
                )

                detected_boxes.append({
                    "coords": coords,
                    "label": f"HAT ({int(confidence * 100)}%)",
                    "confidence": round(confidence, 2),
                    "is_hat": True
                })

            return hat_detected, detected_boxes

        except Exception as e:
            print(f"❌ Error in hat detection: {e}")
            return False, []

    def detect_face_with_boxes(self, image_array: np.ndarray) -> List[Dict]:
        """ตรวจจับใบหน้าและคืนค่าพิกัดกรอบสำหรับวาดบน UI"""
        try:
            face_locations = face_recognition.face_locations(image_array)
            face_boxes = []

            for (top, right, bottom, left) in face_locations:
                face_boxes.append({
                    "coords": [left, top, right, bottom],
                    "label": "FACE",
                    "confidence": 1.0,
                    "is_hat": False
                })

            return face_boxes
        except Exception as e:
            print(f"❌ Error detecting face boxes: {e}")
            return []

    def create_encoding(self, image_array: np.ndarray) -> Optional[np.ndarray]:
        """สร้าง face encoding จากภาพ"""
        try:
            encodings = face_recognition.face_encodings(image_array)
            return encodings[0] if len(encodings) > 0 else None
        except Exception as e:
            print(f"❌ Error creating encoding: {e}")
            return None

    def save_encoding(self, user_id: str, encoding: np.ndarray):
        """บันทึก encoding ลงไฟล์ .pkl"""
        file_path = os.path.join(self.encodings_path, f"{user_id}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(encoding, f)

    def load_encoding(self, user_id: str) -> Optional[np.ndarray]:
        """โหลด encoding ของ user คนเดียว"""
        file_path = os.path.join(self.encodings_path, f"{user_id}.pkl")
        if not os.path.exists(file_path):
            return None

        try:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"❌ Error loading encoding for {user_id}: {e}")
            return None

    def load_all_encodings(self) -> Dict[str, np.ndarray]:
        """โหลด encoding ทั้งหมดในระบบ"""
        all_encodings = {}

        try:
            files = os.listdir(self.encodings_path)
            users = [f.replace(".pkl", "") for f in files if f.endswith(".pkl")]

            for user_id in users:
                encoding = self.load_encoding(user_id)
                if encoding is not None:
                    all_encodings[user_id] = encoding

        except Exception as e:
            print(f"❌ Error loading all encodings: {e}")

        return all_encodings

    def identify(self, image_array: np.ndarray) -> Tuple[Optional[str], float, List[dict]]:
        """ระบุตัวตนจากภาพ"""
        face_encoding = self.create_encoding(image_array)
        if face_encoding is None:
            return None, 0.0, []

        all_encodings = self.load_all_encodings()
        if len(all_encodings) == 0:
            return None, 0.0, []

        results = []

        for user_id, saved_encoding in all_encodings.items():
            distance = face_recognition.face_distance([saved_encoding], face_encoding)[0]
            confidence = (1 - distance) * 100

            results.append({
                "user_id": user_id,
                "confidence": round(float(confidence), 2),
                "match": bool(distance <= self.tolerance)
            })

        results.sort(key=lambda x: x["confidence"], reverse=True)
        best_match = results[0]

        if best_match["match"]:
            return best_match["user_id"], best_match["confidence"], results

        return None, best_match["confidence"], results

    def list_users(self) -> List[str]:
        """ดึงรายชื่อ user จากไฟล์ encoding"""
        try:
            files = os.listdir(self.encodings_path)
            users = sorted([f.replace(".pkl", "") for f in files if f.endswith(".pkl")])
            return users
        except Exception as e:
            print(f"❌ Error listing users: {e}")
            return []

    def delete_encoding(self, user_id: str) -> bool:
        """ลบ encoding ของ user"""
        try:
            file_path = os.path.join(self.encodings_path, f"{user_id}.pkl")
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            print(f"❌ Error deleting encoding for {user_id}: {e}")
            return False


# สร้าง instance
face_system = FaceSystem()