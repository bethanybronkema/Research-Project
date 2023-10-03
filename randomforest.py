import numpy as np
import csv

# fetch dataset 
with open('/Users/bethanybronkema/Documents/mushroom/agaricus-lepiota.data', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

mushroom = np.array(data)

targets = mushroom[:, 0]
features = mushroom[:, 1:]