import not_used.tsne as tsne
import pandas as pd
import matplotlib.pyplot as plt

# file = 'processed_standing_still.xlsx'
file = 'processed_ml_data_v2\processed_Harm_Jordan_Walking.xlsx'

df = pd.read_excel(file)
print(df.head)

time = df['Time_0']
euler = df['Euler1_2']
grav1 = df['Grav1_1']
grav2 = df['Grav1_2']

# fig, (ax1, ax2) = plt.subplots(2)

# print(list(euler))

# ax1.plot(time, grav1)
# ax1.plot(time, euler)
# ax2.plot(time, grav2)
# plt.show()

plt.plot(time, euler)
plt.show()

print(euler)

# features = list(df.columns)
# features.remove('Angle_0')
# for feat in features:
#     tsne.plot_tsne(df, feat, 'Angle_0')



# standing straight: 120-140 degrees
# swing forward: 60 degrees
# swing backwards: above 180 (sometimes clips)

# fix euler3 first
# 1. classify states using euler3
# 2. use all features to predict state


# 1. identify peaks
# 2. classify based on peaks (forward, backward) derivatives
# 3. identify 0 - 100 of gait cycle