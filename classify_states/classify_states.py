from turtle import xcor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import find_peaks
from stse import one_hot

from data_clip_fix import FixClip


class StateClassifier:
    def __init__(self) -> None:
        pass
    
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
    
    def assign_classification(self, df, class_dict):
        df['classification'] = np.zeros(len(df))
        
        for key, val in class_dict.items():
            df.loc[val.astype(bool), 'classification'] = key
            
        return df
    
    def peak_valley_windows(self, x, window_size, prominence=0.1):
        peak_windows = self.peak_windows(x, window_size, prominence)
        valley_windows = self.valley_windows(x, window_size, prominence)
        return peak_windows, valley_windows

    def windows(self, x, window_size):
        
        

if __name__ == '__main__':
    classifier = StateClassifier()
    
    # Window size for peak and valleys
    peak_window_size = 2  # number of pts to look outward
    
    # Import data and extract time and forward facing euler (z-axis)
    df = pd.read_excel('data/processed_ml_data_v2/processed_Keller_Emily_Walking5.xlsx')
    # /home/jacob/git_repos/state_prediction_ml/data/processed_ml_data_v2/processed_Keller_Emily_Walking5.xlsx
    time = df.Time_0.to_numpy()
    euler3 = df.Euler1_2
    
    # Fix clipping and scale between -1 and 1
    clean_euler3 = classifier.clean_euler3(euler3)
    
    # Calculate gradient
    grad_euler3 = classifier.gradient(clean_euler3)
    
    # Initialize classification dict
    bit_vect_dict = {}
    
    # Find signal peaks and valley windows for euler and gradient
    peak_windows = classifier.peak_windows(clean_euler3, peak_window_size, 0.1)
    valley_windows = classifier.valley_windows(clean_euler3, peak_window_size, 0.1)
    grad_peak_windows = classifier.peak_windows(grad_euler3, peak_window_size, 0.1)
    grad_valley_windows = classifier.valley_windows(grad_euler3, peak_window_size, 0.1)
    
    # Sparsify bit vectors for euler and gradient
    bit_vect_dict['peak'] = one_hot.bit_vect(len(clean_euler3), peak_windows)
    bit_vect_dict['valley'] = one_hot.bit_vect(len(clean_euler3), valley_windows)
    bit_vect_dict['grad_peak'] = one_hot.bit_vect(len(grad_euler3), grad_peak_windows)
    bit_vect_dict['grad_valley'] = one_hot.bit_vect(len(grad_euler3), grad_valley_windows)
    
    # Remove overlap progressively
    # Priority order: peak, valley, grad_peak, grad_valley
    valley_bit_vect = one_hot.remove_hot_overlap(bit_vect_dict['valley'], bit_vect_dict['peak'])
    grad_peak_bit_vect = one_hot.remove_hot_overlap(bit_vect_dict['grad_peak'], bit_vect_dict['peak'] +
                                                    bit_vect_dict['valley'])
    grad_valley_bit_vect = one_hot.remove_hot_overlap(bit_vect_dict['grad_valley'], bit_vect_dict['peak'] + 
                                                      bit_vect_dict['valley'] + bit_vect_dict['grad_peak'])
    
    peak_valley_bit_vect = sum(bit_vect_dict.values())
    peak_valley_bit_vect = peak_valley_bit_vect.astype(bool)
    
    
    # # Classify all remaining pts by sign of gradient
    # pos_grad_t = []
    # pos_grad_x = []
    # neg_grad_t = []
    # neg_grad_x = []
    
    
    
    # # Loop through not peak values and classify according to gradient sign
    # for t, x, dx in zip(time[~peak_valley_bit_vect], clean_euler3[~peak_valley_bit_vect],
    #                     grad_euler3[~peak_valley_bit_vect]):
    #     if dx <= 0:
    #         neg_grad_t.append(t)
    #         neg_grad_x.append(x)
    #     else:
    #         pos_grad_t.append(t)
    #         pos_grad_x.append(x)
    
    bit_vect_dict['pos_grad'] = np.zeros(len(peak_valley_bit_vect))
    bit_vect_dict['neg_grad'] = np.zeros(len(peak_valley_bit_vect))
    for i, bit in enumerate(peak_valley_bit_vect):
        dx = grad_euler3[i]
        if int(bit) == 0 and dx <= 0:
            bit_vect_dict['neg_grad'][i] = 1
        elif int(bit) == 0 and dx > 0:
            bit_vect_dict['pos_grad'][i] = 1
            
    # # plt.plot(time, clean_euler3, zorder=-1)
    # plt.scatter(time[peak_windows], clean_euler3[peak_windows], color='b')
    # plt.scatter(time[valley_windows], clean_euler3[valley_windows], color='b')
    # plt.scatter(time[grad_peak_windows], clean_euler3[grad_peak_windows], color='b')
    # plt.scatter(time[grad_valley_windows], clean_euler3[grad_valley_windows], color='b')
    # plt.scatter(pos_grad_t, pos_grad_x, color='b')
    # plt.scatter(neg_grad_t, neg_grad_x, color='b')
    # plt.figure()
    # plt.scatter(time, clean_euler3)
    # plt.show()
    
    # Ensure all points are accounted for
    assert sum(peak_valley_bit_vect) + sum(bit_vect_dict['neg_grad']) + sum(bit_vect_dict['pos_grad']) \
        == len(clean_euler3)
    
    
    ape = classifier.assign_classification(df, bit_vect_dict)
    
    print(ape['classification'])
    
    
            