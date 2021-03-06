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
gt=df.iloc[:,7:10].values
imu=df.iloc[:,10:13].values

ax.scatter(gps[:,0],gps[:,1], gps[:,2],s=0.5, color = "red",label="noisy gps")
ax.plot(ekf[:,0], ekf[:,1], ekf[:,2], color = "green", label="ekf")
ax.plot( gt[:,0], gt[:,1],   gt[:,2], color = "blue",  label="ground truth")
ax.plot(imu[:,0], imu[:,1], imu[:,2], color = "orange",label="only imu")
ax.legend()

plt.show()