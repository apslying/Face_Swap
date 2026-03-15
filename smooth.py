import cv2
import numpy as np
# (The LandmarkSmoother class goes here)

class LandmarkSmoother:
    def __init__(self, num_landmarks=68):
        self.num_landmarks = num_landmarks
        self.filters = []
        
        for _ in range(num_landmarks):
            # State: [x, y, dx, dy] | Measurement: [x, y]
            kf = cv2.KalmanFilter(4, 2)
            
            # Transition Matrix (How state evolves: x_new = x + dx)
            kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                            [0, 1, 0, 1],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], np.float32)
            
            # Measurement Matrix (We only observe x and y)
            kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0]], np.float32)
            
            # Process Noise (Higher = more trust in sensor, Lower = smoother/slower)
            kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
            
            # Measurement Noise (How much we trust dlib's points)
            kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
            
            self.filters.append(kf)
        
        self.initialized = False

    def update(self, points):
        smoothed_points = []
        
        for i, (x, y) in enumerate(points):
            meas = np.array([[np.float32(x)], [np.float32(y)]])
            
            if not self.initialized:
                # Initialize state with the first detection
                self.filters[i].statePre = np.array([[np.float32(x)], [np.float32(y)], [0], [0]], np.float32)
                self.filters[i].statePost = np.array([[np.float32(x)], [np.float32(y)], [0], [0]], np.float32)
            
            # Predict & Correct
            self.filters[i].predict()
            estimate = self.filters[i].correct(meas)
            
            smoothed_points.append((estimate[0][0], estimate[1][0]))
            
        self.initialized = True
        return np.array(smoothed_points)