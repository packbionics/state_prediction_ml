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
        pass
                
    # def classify(self, euler3, time):
    #     euler3 = self.clean_euler3(euler3)
    #     plt.plot(range(len(euler3)), euler3)
    #     grad_euler3 = self.gradient(euler3)
    #     # plt.plot(range(len(euler3)), grad_euler3)
    #     plt.show()
        
    #     peak_windows = self.peak_windows(euler3, 0.6)
    #     valley_windows = self.valley_windows(euler3, 0.6)
        
    #     mask = np.ones(euler3.size, dtype=bool)
    #     print(np.concatenate((peak_windows, valley_windows)))
    #     mask[np.concatenate((peak_windows, valley_windows))] = False
    #     not_peaks = euler3[mask]
    #     print(not_peaks)
        
    #     plt.scatter(time[peak_windows], euler3[peak_windows], color='black', label='Peaks', s=.5)
    #     plt.scatter(time[valley_windows], euler3[valley_windows], color='black', label='Peaks', s=.5)
    #     plt.scatter(time[mask], euler3[mask], s=.5)
    #     plt.show()
    
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
    
    def peak_windows(self, y, window_size, prominence, scalar=1):  # *
        peak_indices = find_peaks(scalar*y, prominence=prominence)[0]
        
        peak_windows = np.array([range(-window_size - 1 + peak, window_size + peak) 
                                 for peak in peak_indices])
        return peak_windows.flatten()
    
    def valley_windows(self, y, window_size, prominence):
        return self.peak_windows(y, window_size, prominence, scalar=-1)
        
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
    classifier = StateClassifier()
    
    window_size = 2
    
    df = pd.read_excel('../processed_ml_data_v2/processed_Keller_Emily_Walking5.xlsx')
    time = df.Time_0.to_numpy()
    euler3 = df.Euler1_2
    
    # Fix clipping and scale between -1 and 1
    clean_euler3 = classifier.clean_euler3(euler3)
    
    # Calculate gradient
    grad_euler3 = classifier.gradient(clean_euler3)
    
    # Find signal peaks and valley pts
    peak_windows = classifier.peak_windows(clean_euler3, window_size, 0.1)
    cleak_euler3 = np.delete(clean_euler3, peak_windows)
    valley_windows = classifier.valley_windows(clean_euler3, window_size, 0.1)
    cleak_euler3 = np.delete(clean_euler3, valley_windows)
    
    # Find gradient peaks and valley pts
    grad_peak_windows = classifier.peak_windows(grad_euler3, window_size, 0.1)
    grad_valley_windows = classifier.valley_windows(grad_euler3, window_size, 0.1)
    
    # Classify remaining pts by sign of gradient
    pos_grad_t = []
    pos_grad_x = []
    neg_grad_t = []
    neg_grad_x = []
    print(grad_peak_windows, grad_valley_windows)
    peak_locs = np.concatenate((peak_windows, valley_windows, grad_peak_windows, grad_valley_windows)) #+ grad_peak_windows + grad_valley_windows
    not_peak_mask = np.ones(len(clean_euler3), dtype=bool)
    not_peak_mask[peak_locs] = False
    for t, x, dx in zip(time[not_peak_mask], clean_euler3[not_peak_mask], grad_euler3[not_peak_mask]):
        if dx < 0:
            neg_grad_t.append(t)
            neg_grad_x.append(x)
        else:
            pos_grad_t.append(t)
            pos_grad_x.append(x)
            
    print(len(time[not_peak_mask]), len(clean_euler3[not_peak_mask]), len(grad_euler3[not_peak_mask]), len(peak_locs))
    
    for i in peak_windows:
        if i in peak_locs:
            print(False)
            
    # plt.plot(time, clean_euler3, zorder=-1)
    plt.scatter(time[peak_windows], clean_euler3[peak_windows], color='b')
    plt.scatter(time[valley_windows], clean_euler3[valley_windows], color='b')
    plt.scatter(time[grad_peak_windows], clean_euler3[grad_peak_windows], color='b')
    plt.scatter(time[grad_valley_windows], clean_euler3[grad_valley_windows], color='b')
    plt.scatter(pos_grad_t, pos_grad_x, color='b')
    plt.scatter(neg_grad_t, neg_grad_x, color='b')
    plt.figure()
    plt.scatter(time, clean_euler3)
    plt.show()
    
    total_len = len(peak_windows) + len(valley_windows) + len(grad_peak_windows) + len(grad_valley_windows) + len(pos_grad_t) + len(neg_grad_t)
    print(total_len)
    print(len(clean_euler3))
            
    
    