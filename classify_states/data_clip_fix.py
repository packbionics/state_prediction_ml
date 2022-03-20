import pandas as pd
import matplotlib.pyplot as plt


class FixClip:
    def __init__(self, euler_arr) -> None:
        self.new_euler = list(map(self.fix, euler_arr))
    
    def fix(self, euler):
        if euler < 50:
            euler += 360
        return euler - 150
        # return euler
    
    
if __name__ == '__main__':
    df = pd.read_excel('processed_ml_data_v2\processed_Keller_Emily_Walking5.xlsx')
    # df = pd.read_excel('processed_ml_data_v2\processed_Harm_Jordan_Walking.xlsx')

    euler3 = df['Euler1_2']
    time = df['Time_0']


    def clip_fix(euler):
        if euler < 50:
            euler += 360
        # return euler - 150
        return euler


    new_euler = list(map(clip_fix, euler3))


    plt.plot(time, euler3)
    plt.figure()
    plt.plot(time, new_euler)
    plt.show()