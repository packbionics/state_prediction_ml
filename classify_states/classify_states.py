import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import find_peaks

from data_clip_fix import FixClip


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
    
    def clean_euler3(self, euler3):  # *
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
    
    def gradient(self, x):
        """Returns gradient of array.

        Args:
            euler3 (arraylike): Input to take derivative of.

        Returns:
            arraylike: Gradient of x.
        """
        return np.gradient(np.squeeze(x))
    
    def peak_windows(self, y, window_size, prominence, scalar=1):  # *
        peak_indices = find_peaks(scalar*y, prominence=prominence)[0]
        
        peak_windows = np.array([range(-window_size - 1 + peak, window_size + peak) 
                                 for peak in peak_indices])
        return peak_windows.flatten()
    
    def valley_windows(self, y, window_size, prominence):
        """Inverts signal before finding peaks.

        Args:
            y (_type_): _description_
            window_size (_type_): _description_
            prominence (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.peak_windows(y, window_size, prominence, scalar=-1)
    
def bit_vect(length, indices):
    """Generates a bit vector, hot at each index.

    Args:
        length (int): Length of desired bit vector.
        length
        indices (iter(int)): List of indices to equal 1.

    Returns:
        np.array: bit vector.
    """
    out = np.zeros(length)
    out[indices] = 1
    return out

def remove_bit_vect_overlap(bit_vect, overlap_vect):
    overlap_vect = overlap_vect.astype(bool)
    bit_vect[overlap_vect] = 0
    return bit_vect
        

if __name__ == '__main__':
    classifier = StateClassifier()
    
    # Window size for peak and valleys
    peak_window_size = 2  # number of pts to look outward
    
    # Import data and extract time and forward facing euler (z-axis)
    df = pd.read_excel('../processed_ml_data_v2/processed_Keller_Emily_Walking5.xlsx')
    time = df.Time_0.to_numpy()
    euler3 = df.Euler1_2
    
    # Fix clipping and scale between -1 and 1
    clean_euler3 = classifier.clean_euler3(euler3)
    
    # Calculate gradient
    grad_euler3 = classifier.gradient(clean_euler3)
    
    # Find signal peaks and valley windows for euler and gradient
    peak_windows = classifier.peak_windows(clean_euler3, peak_window_size, 0.1)
    valley_windows = classifier.valley_windows(clean_euler3, peak_window_size, 0.1)
    grad_peak_windows = classifier.peak_windows(grad_euler3, peak_window_size, 0.1)
    grad_valley_windows = classifier.valley_windows(grad_euler3, peak_window_size, 0.1)
    
    # Sparsify bit vectors for euler and gradient
    peak_bit_vect = bit_vect(len(clean_euler3), peak_windows)
    valley_bit_vect = bit_vect(len(clean_euler3), valley_windows)
    grad_peak_bit_vect = bit_vect(len(grad_euler3), grad_peak_windows)
    grad_valley_bit_vect = bit_vect(len(grad_euler3), grad_valley_windows)
    
    # Remove overlap progressively
    # Priority order: peak, valley, grad_peak, grad_valley
    valley_bit_vect = remove_bit_vect_overlap(valley_bit_vect, peak_bit_vect)
    grad_peak_bit_vect = remove_bit_vect_overlap(grad_peak_bit_vect, peak_bit_vect + valley_bit_vect)
    grad_valley_bit_vect = remove_bit_vect_overlap(grad_valley_bit_vect, peak_bit_vect + valley_bit_vect + 
                                                   grad_peak_bit_vect)
    
    
    peak_valley_bit_vect = peak_bit_vect + valley_bit_vect + valley_bit_vect + grad_peak_bit_vect + grad_valley_bit_vect
    peak_valley_bit_vect = peak_valley_bit_vect.astype(bool)
    # Classify all remaining pts by sign of gradient
    pos_grad_t = []
    pos_grad_x = []
    neg_grad_t = []
    neg_grad_x = []
    
    
    
    # Loop through not peak values and classify according to gradient sign
    for t, x, dx in zip(time[~peak_valley_bit_vect], clean_euler3[~peak_valley_bit_vect],
                        grad_euler3[~peak_valley_bit_vect]):
        if dx <= 0:
            neg_grad_t.append(t)
            neg_grad_x.append(x)
        else:
            pos_grad_t.append(t)
            pos_grad_x.append(x)
            
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
    
    # Print out accounted for length
    assert sum(peak_valley_bit_vect) + len(pos_grad_t) + len(neg_grad_t) == len(clean_euler3)
            