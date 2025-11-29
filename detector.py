from ultralytics import YOLO
import numpy as np


class FaceDetector:
    def __init__(self, model_path="yolov8x-face-lindevs.pt"):
        self.model = YOLO(model_path)

    def detect(self, image):
        """
        输入: BGR 图像
        输出: boxes, keypoints
        """
        try:
            results = self.model(image)[0]
            boxes = results.boxes.xyxy.cpu().numpy()
            if hasattr(results, 'keypoints') and results.keypoints is not None:
                kp = results.keypoints.cpu().numpy()
                # keypoints 格式: [N, num_keypoints, 2] 或 [N, num_keypoints, 3]
                # 提取前两个维度（x, y），忽略置信度
                if kp.shape[-1] >= 2:
                    keypoints = kp[..., :2]  # 只取 x, y 坐标
                else:
                    keypoints = kp
            else:
                keypoints = None
            return boxes, keypoints
        except Exception as e:
            print(f"人脸检测失败: {e}")
            import traceback
            traceback.print_exc()
            return np.array([]), None
if __name__ == '__main__':
    model = YOLO("yolov8x-face-lindevs.pt")
    results = model("1.jpg")
    results[0].show()
