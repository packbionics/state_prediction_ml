import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('processed_ml_data_v2\processed_Keller_Emily_Walking5.xlsx')

euler3 = df['Euler1_2']
time = df['Time_0']


def clip_fix(euler):
    if euler < 50:
        euler += 360
    return euler - 150

new_euler = list(map(clip_fix, euler3))

plt.plot(time, new_euler)
plt.show()