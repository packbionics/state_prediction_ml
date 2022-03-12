import pandas as pd
import numpy as np
from scipy.signal import find_peaks

from test_skeleton import Skeleton
from classify_states.classify_states import StateClassifier


# def peak_windows(y, prominence, window_size):
#     peaks = find_peaks(y, prominence=prominence)[0]
#     return np.array([range(-window_size - 1 + peak, window_size + peak) for peak in peaks]).flatten()

class ClassifierSkeleton(Skeleton):
    def __init__(self):
        super().__init__()
        self.test_df = pd.read_excel('processed_Harm_Jordan_Walking.xlsx')
        self.state_classifier = StateClassifier()


class ClassifierTests(ClassifierSkeleton):
    def test_peak_windows(self):
        test_y = 10*[1] + [5] + 10*[1] + [5, 1]
        window_size = 3
        test_x = np.array([7, 8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 22])
        assert np.array_equal(self.state_classifier.peak_windows(test_y, prominence=0.1, window_size=window_size), 
                              test_x)


class CleaningTests(ClassifierSkeleton):
    def test_clean_euler3(self):
        euler3 = self.test_df.Euler1_2
        new_euler3 = self.state_classifier.clean_euler3(euler3)
        
        # Round to ten decimal places to account for any, extremely miniscule, deviations
        mask = [True if round(x, 10) <= 1 and round(x, 10) >= -1 else False for x in new_euler3]
        
        # Test that everything is scaled between -1 and 1
        assert np.array_equal(new_euler3[mask], new_euler3)
        
        
if __name__ == '__main__':
    # ClassifierTests().run_all_tests()
    CleaningTests().run_all_tests()