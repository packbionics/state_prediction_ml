import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import find_peaks
from stse import one_hot
import plotly.express as px

from data_clip_fix import FixClip


class StateClassifier:
    def __init__(self, df, euler_name) -> None:
        # Mutable variables
        self.peak_window_radius = 2
        self.prominence = 0.1
        
        self.euler3 = df[euler_name]
        
        # fig = px.scatter(x=range(len(self.euler3)), y=self.euler3)
        # fig.show()
        
        # Calculate gradient
        self.grad_euler3 = self.__gradient(self.euler3)
    
    def __gradient(self, x):
        """Returns gradient of array.

        Args:
            euler3 (arraylike): Input to take derivative of.

        Returns:
            arraylike: Gradient of x.
        """
        return np.gradient(np.squeeze(x))
    
    def __peak_windows(self, y, radius, scalar=1.0):  # *
        """Generates windows for peak points in signal according to window size.

        :param y: Input signal
        :type y: iter[float]
        :param window_size: Radius of window
        :type window_size: int
        :param prominence: scipy.signal.find_peak argument
        :type prominence: float
        :param scalar: Value to multiple vector by, defaults to 1
        :type scalar: float, optional
        :return: Collection of peak indices
        :rtype: arraylike
        """
        peak_indices = find_peaks(scalar*y, prominence=self.prominence)[0]
        
        peak_windows = np.array([range(-radius - 1 + peak, radius + peak) 
                                 for peak in peak_indices])
        return peak_windows.flatten()
    
    def __valley_windows(self, y, radius):
        """Wrapper for self.__peak_windows that first inverts signal by multiplying by -1.

        :param y: Input signal
        :type y: iter[float]
        :param window_size: Radius of window
        :type window_size: int
        :return: Collection of valley indices
        :rtype: arraylike
        """
        return self.__peak_windows(y, radius, scalar=-1)
    
    def __assign_classification(self, df, class_dict):
        df['classification'] = np.zeros(len(df))
        
        for key, val in class_dict.items():
            df.loc[val.astype(bool), 'classification'] = key
            
        return df
    
    def __peak_valley_windows(self, x, window_size):
        peak_windows = self.__peak_windows(x, window_size)
        valley_windows = self.__valley_windows(x, window_size)
        return peak_windows, valley_windows
    
    def __get_windows(self):
         # Find signal peaks and valley windows for euler and gradient
        peak_windows, valley_windows = self.__peak_valley_windows(self.euler3, self.peak_window_radius)
        grad_peak_windows, grad_valley_windows = self.__peak_valley_windows(self.grad_euler3, self.peak_window_radius)
        return (peak_windows, valley_windows), (grad_peak_windows, grad_valley_windows)
    
    def __peak_valley_bit_vects(self, peak_windows, valley_windows, grad_peak_windows, grad_valley_windows):
        length = len(self.euler3)
        bit_vect_dict = {}
        
        bit_vect_dict['peak'] = one_hot.bit_vect(length, peak_windows)
        bit_vect_dict['valley'] = one_hot.bit_vect(length, valley_windows)
        bit_vect_dict['grad_peak'] = one_hot.bit_vect(length, grad_peak_windows)
        bit_vect_dict['grad_valley'] = one_hot.bit_vect(length, grad_valley_windows)
        
        # Remove overlap progressively
        # Priority order: peak, valley, grad_peak, grad_valley
        bit_vect_dict['valley'] = one_hot.remove_hot_overlap(bit_vect_dict['valley'], bit_vect_dict['peak'])
        bit_vect_dict['grad_peak'] = one_hot.remove_hot_overlap(bit_vect_dict['grad_peak'], bit_vect_dict['peak'] +
                                                                bit_vect_dict['valley'])
        bit_vect_dict['grad_valley'] = one_hot.remove_hot_overlap(bit_vect_dict['grad_valley'], bit_vect_dict['peak'] + 
                                                                  bit_vect_dict['valley'] + bit_vect_dict['grad_peak'])
        
        return bit_vect_dict
    
    def __sum_bit_dict(self, bit_vect_dict):
        return np.sum(list(bit_vect_dict.values()), axis=0)
    
    def __classify_remaining(self, bit_vect_dict):
        peak_valley_bit_vect = self.__sum_bit_dict(bit_vect_dict)
        bit_vect_dict['pos_grad'] = np.zeros(len(peak_valley_bit_vect))
        bit_vect_dict['neg_grad'] = np.zeros(len(peak_valley_bit_vect))
        for i, bit in enumerate(peak_valley_bit_vect):
            dx = self.grad_euler3[i]
            if int(bit) == 0 and dx <= 0:
                bit_vect_dict['neg_grad'][i] = 1
            elif int(bit) == 0 and dx > 0:
                bit_vect_dict['pos_grad'][i] = 1
                
        return bit_vect_dict
        
    # Ensure all points are accounted for
    def classify(self):
        # Find signal peaks and valley windows for euler and gradient
        (peak, valley), (grad_peak, grad_valley) = self.__get_windows()
        
        # Sparsify bit vectors for euler and gradient
        bit_vect_dict = self.__peak_valley_bit_vects(peak, valley, grad_peak, grad_valley)
        bit_vect_dict = self.__classify_remaining(bit_vect_dict)
        print(bit_vect_dict)
        return self.__assign_classification(df, bit_vect_dict)
        
def clean_euler3(euler3):  # *
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
        
        

if __name__ == '__main__':
    # Import data
    in_dir = 'data/processed_ml_data_v2/'
    out_dir = 'data/classified/'
    
    import os
    for file in os.listdir(in_dir):
    
        df = pd.read_excel(in_dir + file)
        
        # Fix clipping and scale between -1 and 1
        df['Euler1_2'] = clean_euler3(df['Euler1_2'])
        #  = new_euler
        
        classifier = StateClassifier(df, 'Euler1_2')
        
        # Window size for peak and valleys
        classifier.peak_window_radius = 2  # number of pts to look outward
        
        df = classifier.classify()
        
        df.to_excel(f'{out_dir}classified_{file}')
    
    # Plot
    # fig = px.scatter(df, x='Time_0', y='Euler1_2', color='classification')
    # fig.show()
    
    
        

    
    
    
            