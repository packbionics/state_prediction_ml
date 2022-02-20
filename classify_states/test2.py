import pandas as pd
import matplotlib.pyplot as plt
from classify_states_2 import StateClassifier


classifier = StateClassifier()

data_path = '../processed_ml_data_v2\processed_Keller_Emily_Walking5.xlsx'
df = pd.read_excel(data_path)

time = df.Time_0
euler3 = df.Euler1_2

classifier.classify(euler3, time)

grad = classifier.gradient(euler3)

plt.plot(list(range(len(grad))), grad)
plt.show()