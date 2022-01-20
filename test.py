import tsne
import pandas as pd
import matplotlib.pyplot as plt

file = 'processed_ml_data_v2\processed_Harm_Jordan_Walking.xlsx'

df = pd.read_excel(file)

time = df['Time_0']
euler = df['Euler1_2']
grav1 = df['Grav1_1']
grav2 = df['Grav1_2']

fig, (ax1, ax2) = plt.subplots(2)

ax1.plot(time, grav1)
ax1.plot(time, euler)
ax2.plot(time, grav2)
plt.show()

# features = list(df.columns)
# features.remove('Angle_0')
# for feat in features:
#     tsne.plot_tsne(df, feat, 'Angle_0')