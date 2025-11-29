import insightface

class FeatureExtractor:
    def __init__(self, model_name="buffalo_l", ctx_id=-1, det_size=(320,320)):
        """
        ctx_id=-1 表示 CPU
        """
        self.model = insightface.app.FaceAnalysis(name=model_name)
        self.model.prepare(ctx_id=ctx_id, det_size=det_size)

    def extract(self, face_img):
        """
        输入: 对齐后的 BGR 人脸图像
        输出: 128/512维特征向量
        """
        faces = self.model.get(face_img)
        if len(faces) == 0:
            return None
        return faces[0].normed_embedding
