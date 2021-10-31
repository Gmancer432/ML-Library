
import numpy as np


class PerceptronStandard:
    def __init__(self, w):
        self.w = w
    def PredictLabel(self, data):
        return 