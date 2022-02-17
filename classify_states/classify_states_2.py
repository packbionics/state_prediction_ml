import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.append('..')
from data_clip_fix import FixClip
from scipy.signal import find_peaks


class StateClassifier:
    def __init__(self) -> None:
        self.window_size = 4
                
    def classify(self, euler3, time):
        euler3 = self.clean_euler3(euler3)
        plt.plot(range(len(euler3)), euler3)
        grad_euler3 = self.gradient(euler3)
        # plt.plot(range(len(euler3)), grad_euler3)
        plt.show()
        
        peak_windows = self.peak_windows(euler3, 0.6)
        valley_windows = self.valley_windows(euler3, 0.6)
        
        mask = np.ones(euler3.size, dtype=bool)
        print(np.concatenate((peak_windows, valley_windows)))
        mask[np.concatenate((peak_windows, valley_windows))] = False
        not_peaks = euler3[mask]
        print(not_peaks)
        
        plt.scatter(time[peak_windows], euler3[peak_windows], color='black', label='Peaks', s=.5)
        plt.scatter(time[valley_windows], euler3[valley_windows], color='black', label='Peaks', s=.5)
        plt.scatter(time[mask], euler3[mask], s=.5)
        plt.show()
    
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
        scaler = MinMaxScaler((-1, 1))
        new_euler3 = new_euler3.reshape(-1, 1)
        new_euler3 = scaler.fit_transform(new_euler3)
        
        return np.squeeze(new_euler3)
    
    def gradient(self, euler3):
        return np.gradient(np.squeeze(euler3))
    
    def peak_windows(self, y, prominence, scalar=1):  # *
        peak_indices = find_peaks(scalar*y, prominence=prominence)[0]
        
        peak_windows = np.array([range(-self.window_size - 1 + peak, self.window_size + peak) 
                                 for peak in peak_indices])
        return peak_windows.flatten()
    
    def valley_windows(self, y, prominence):
        return self.peak_windows(y, prominence, scalar=-1)
        
        # lower_bounds = [peak - self.window_size for peak in peak_indices]
        # upper_bounds = [self.window_size + peak + 1 for peak in peak_indices]
        
        # out = []
        # for lb, ub in zip(lower_bounds, upper_bounds):
        #     if ub < len(y) + 1:
        #         out += list(range(lb, ub))
        #     else:
        #         out += list(range(lb, len(y)))
                
        # return np.array(out)
        

if __name__ == '__main__':
    pass