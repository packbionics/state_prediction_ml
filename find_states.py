import numpy as np
import pandas as pd
import ruptures as rpt
import matplotlib.pyplot as plt

# Change point detection

# test_set = [1,2,3,
#             11,12,13,
#             21,22,23,
#             31,32,33]

# n_breakpoints = 2

# signal, breakpts = rpt.pw_constant(len(test_set), 1, n_breakpoints)

# algo = rpt.Pelt(model='rbf').fit(signal)
# result = algo.predict()
# rpt.display(signal, breakpts, result)
# plt.show()
# print(breakpts)
# print(test_set)
df = pd.read_excel('merged_data.xlsx')

accel = df['Acc1_0']

plt.plot(list(range(0, len(accel))), accel)
plt.show()