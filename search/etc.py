import os
import json
import matplotlib.pyplot as plt
import numpy as np



def openJson(jsonDir) -> json:
    with open(jsonDir, "r") as f:
        data = json.load(f)

    return data

json_path = "./data/illustData.json"
data = openJson(json_path)

for i in data:
    bookmark = data[i]["bookmark"]
    view = data[i]["view"]
    if view <= 10000:
        plt.plot(view, bookmark, marker=".")

x = np.arange(0, 8000, 0.1)
y = x
plt.plot(x, y)

plt.show()