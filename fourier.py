import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = 'processed_ml_data_v2\processed_Harm_Jordan_Walking.xlsx'

X = pd.read_excel(file)['Angle_0']
X = [i - X.mean() for i in X]
y = np.fft.fft(X)
freq = np.fft.fftfreq(3345, d=.18000)

fig, ax1 = plt.subplots(1)
ax1.plot(freq, y)

# ax2.plot(range(len(ifft(y))), ifft(y))
plt.show()