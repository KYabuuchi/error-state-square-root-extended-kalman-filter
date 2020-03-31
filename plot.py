#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("orbit", size = 20)
ax.set_xlabel("x", size = 14)
ax.set_ylabel("y", size = 14)
ax.set_zlabel("z", size = 14)
ax.set_xticks([-2.0, -1.0, 0.0, 1.0, 2.0])
ax.set_yticks([-2.0, -1.0, 0.0, 1.0, 2.0])
ax.set_zticks([-2.0, -1.0, 0.0, 1.0, 2.0])


df = pd.read_csv("result.csv",header=None, delim_whitespace=True)
print(df.head())
print(df.tail())
gps=df.iloc[:,1:4].values
ekf=df.iloc[:,4:7].values
ans=df.iloc[:,7:10].values
ax.scatter(gps[:,0],gps[:,1], gps[:,2],s=0.5, color = "red")
ax.plot(ekf[:,0],ekf[:,1], ekf[:,2], color = "green")
ax.plot(ans[:,0],ans[:,1], ans[:,2], color = "blue")

plt.show()