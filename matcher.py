import numpy as np

class FaceMatcher:
    def __init__(self, threshold=0.55):
        self.threshold = threshold

    def match(self, feature, database):
        """
        feature: 待匹配特征
        database: {'name': vector, ...}
        输出: label, similarity
        """
        label = "Unknown"
        max_sim = 0
        for name, db_fea in database.items():
            cos_sim = np.dot(feature, db_fea)
            if cos_sim > self.threshold and cos_sim > max_sim:
                max_sim = cos_sim
                label = name
        return label, max_sim
