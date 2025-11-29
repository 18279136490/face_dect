from insightface.utils import face_align
import cv2

class FaceAligner:
    def __init__(self, input_size=112):
        self.input_size = input_size

    def align(self, image, keypoints=None, box=None):
        """
        输入: 图像, 人脸关键点或box
        输出: 对齐后的112x112人脸
        """
        if keypoints is not None:
            aligned = face_align.norm_crop(image, keypoints)
        elif box is not None:
            x1, y1, x2, y2 = box.astype(int)
            aligned = cv2.resize(image[y1:y2, x1:x2], (self.input_size, self.input_size))
        else:
            raise ValueError("必须提供 keypoints 或 box")
        return aligned
