import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from data_clip_fix import FixClip


class StateClassifier:
    def __init__(self, euler, time) -> None:
        self.euler = euler.copy()
        self.time = time.copy()
        self.offset = 150
        
        self.standing_straight = ([], []); self.forward_swing = ([], []); self.backward_swing = ([], [])
        
        self.range = max(self.euler) - min(self.euler)
        
        self.third = self.range/3
        
        self.states = {
            0: self.standing_straight,
            -40: self.forward_swing,
            50: self.backward_swing
        }  # degrees
        print('third', self.third)
    def classify(self):
        for ang, t in zip(self.euler, self.time):
            if ang <= self.third:
                self.states[-40][0].append(t)
                self.states[-40][1].append(ang)
            elif ang < 2*self.third:
                self.states[0][0].append(t)
                self.states[0][1].append(ang)
                # print(ang)
            elif ang >= 2*self.third:
                self.states[50][0].append(t)
                self.states[50][1].append(ang)
        


if __name__ == '__main__':
    # Read in data
    df = pd.read_excel('processed_ml_data_v2\processed_Keller_Emily_Walking5.xlsx')
    time = df.Time_0
    euler3 = df.Euler1_2
    
    # Fix clipping
    new_euler3 = FixClip(euler3).new_euler
    
    
    scaler = MinMaxScaler()
    new_euler3 = scaler.fit_transform(np.array(new_euler3).reshape(-1, 1))
    
    classifier = StateClassifier(new_euler3, time)
    
    classifier.classify()
    
    
    plt.plot(time, new_euler3)
    plt.scatter(classifier.standing_straight[0], classifier.standing_straight[1])
    plt.scatter(classifier.forward_swing[0], classifier.forward_swing[1])
    plt.scatter(classifier.backward_swing[0], classifier.backward_swing[1])
    plt.show()