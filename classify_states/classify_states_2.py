import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from data_clip_fix import FixClip
from scipy.signal import find_peaks


class StateClassifier:
    def __init__(self) -> None:
        self.window_size = 10
        pass
                
    def classify(self, euler3, time):
        euler3 = self.clean_data(euler3)
        
        
        pass
    
    def clean_euler3(self, euler3):
        """Cleans euler3 data from IMU brace 1.0 measurement, normalizes between -1 and 1.

        Args:
            euler3 (iterable): Contains euler angles in the plane of forward movement.

        Returns:
            iterable: Cleaned euler angles.
        """
        
        # Fix clipping issue
        new_euler3 = np.array(FixClip(euler3).new_euler)
        
        # Scale euler between -1 and 1
        scaler = MinMaxScaler()
        new_euler3 = scaler.fit_transform(new_euler3.reshape(-1, 1))
        
        return new_euler3
    
    def gradient(self, euler3):
        return np.gradient(np.squeeze(euler3))
    
    def peak_windows(self, y, prominence, window_size):  # *
        peaks = find_peaks(y, prominence=prominence)[0]
        
        lower_bounds = [peak - window_size for peak in peaks]
        upper_bounds = [window_size + peak + 1 for peak in peaks]
        
        out = []
        for lb, ub in zip(lower_bounds, upper_bounds):
            if ub < len(y) + 1:
                out += list(range(lb, ub))
            else:
                out += list(range(lb, len(y)))
                
        return np.array(out)
        

if __name__ == '__main__':
    pass