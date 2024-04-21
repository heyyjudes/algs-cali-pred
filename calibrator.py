from sklearn.linear_model import LogisticRegression
import numpy as np

class Calibrator:
    def __init__(self, name:str):
        self.params = {}
        self.calibrator_fn = None
        self.name = name

class PlattCalibrator(Calibrator):
    def __init__(self):
        self.name = 'PlattScaling'
        super().__init__("PlattCalibrator")

    def calibrate(self, y_prob: np.ndarray, y_true : np.ndarray):
        logistic = LogisticRegression(C=1e10, solver='lbfgs')
        logistic.fit(y_prob.reshape(-1, 1), y_true)
        coeff = logistic.coef_[0]
        intercept = logistic.intercept_
        self.params['coeff'] = coeff
        self.params['intercept'] = intercept
        return

    def transform(self, y_prob):
        out = y_prob * self.params['coeff'] + self.params['intercept']
        return 1/ (1+ np.exp(-out))


class BinningCalibrator(Calibrator):
    def __init__(self, bins=20):
        self.name = 'BinningCalibrator'
        self.B = bins
        super().__init__("BinningCalibrator")

    def calibrate(self, y_prob: np.ndarray, y_true : np.ndarray):
        sorted_y = np.asarray(y_true)[np.argsort(y_prob)]
        scores = np.asarray(y_prob)[np.argsort(y_prob)]
        delta = int(len(y_true) + 1) / self.B
        intervals = []
        new_values = []
        for i in range(self.B):
            if i == 0:
                intervals.append((0, scores[int((i + 1) * delta)]))
            elif i == self.B - 1:
                intervals.append((scores[int(i * delta)], 1))
            else:
                intervals.append((scores[int(i * delta)], scores[int((i + 1) * delta)]))
        for i in range(self.B):
            if int((i + 1) * delta) < len(y_true):
                new_values.append(np.mean(sorted_y[int(i * delta):int((i + 1) * delta)]))
            else:
                new_values.append(np.mean(sorted_y[int(i * delta):]))

        self.params['intervals'] = intervals
        self.params['new_values'] = np.asarray(new_values)
        return

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        bin_indices = np.digitize(y_prob, [interval[1] for interval in self.params['intervals']], right=True)
        return self.params['new_values'][bin_indices - 1]