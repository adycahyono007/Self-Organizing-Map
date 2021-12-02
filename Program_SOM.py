# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 18:28:26 2020

@author: QXRZDRAGON
"""


from numpy import *
import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(x1,x2,y1,y2):
    dist = sqrt((x1-y1)**2+(x2-y2)**2)
    return dist

dataset = array([
            [95.0, 89.0, 91.0, 87.0, 80.0, 70.0, 67.0, 70.0, 81.0, 72.0],
            [86.0, 83.0, 80.0, 79.0, 70.0, 85.0, 78.0, 90.0, 87.0, 89.0],
        ])

cluster = array([
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0]
     ])

clustertemp = array([
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0]
     ])

dist = array([
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    ])
    
neuron = array([
            [80.0,72.0],
            [70.0,89.0]
        ])

learning_rate=0.1
k=2
iterasi=0
sama = np.array_equal(cluster, clustertemp)

print('Data yang dipakai: \n')
print('dataset: \n', dataset)
print('Centroid awal: \n', neuron[:][0],'\n',neuron[:][1])

while(sama):
    print("\n\niterasi: ", iterasi+1)
    print('variable cluster sebelum clustering: \n',cluster)  
    print('Centroid: \n', neuron)      
    
        
    for i in range(dataset.shape[1]):
        dist[0][i] = round(euclidean_distance(dataset[0][i], dataset[1][i], neuron[0][0], neuron[1][0]),2)
        dist[1][i] = round(euclidean_distance(dataset[0][i], dataset[1][i], neuron[0][1], neuron[1][1]),2)
        if dist[0][i] < dist[1][i]:
            clustertemp[0][i] = 1
            clustertemp[1][i] = 0
        else:
            clustertemp[0][i] = 0
            clustertemp[1][i] = 1                           
    print("dist: \n", dist)
    print("Cluster setelah proses clustering: \n", clustertemp)
    if np.array_equal(cluster, clustertemp):
        sama=False
    else:
        z1 = np.array([[],[]])
        z2 = np.array([[],[]])
        cluster = clustertemp.copy()
        iterasi += 1
        index = 0
        for i in range(dataset.shape[0]):
            for j in range(dataset.shape[1]):
                if (cluster[0][j] == 1):
                    z1 = np.concatenate((z1,dataset[:, index:index+1]),axis=1)
                else:
                    z2 = np.concatenate((z2,dataset[:, index:index+1]),axis=1)
                index += 1
        
        c = array([[0.0,0.0],[0.0,0.0]]) 
        temp  = neuron[0][0]
        temp1 = neuron[1][0] 
        #pembaharuan winning neuron 1
        for i in range(z1.shape[1]):
            c[0][0]= temp + learning_rate*(z1[0][i]-temp)
            temp = c[0][0]
            c[1][0]= temp1 + learning_rate*(z1[1][i]-temp1)
            temp1 = c[1][0]                
        
        temp  = neuron[0][1]
        temp1 = neuron[1][1] 
        #pembaharuan winning neuron 2         
        for i in range(z2.shape[1]):
            c[0][1] = temp + learning_rate*(z2[0][i]-temp)
            temp = c[0][1]
            c[1][1]= temp1 + learning_rate*(z2[1][i]-temp1)
            temp1 = c[1][1]
        
        neuron = c
    
    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[1]):
            if cluster[0][j] == 1:        
                plt.scatter(dataset[0,j], dataset[1, j], c='green', s=50, alpha=0.5)
            else:
                plt.scatter(dataset[0,j], dataset[1, j], c='purple', s=50, alpha=0.5)
    plt.scatter(neuron[0][0], neuron[1][0], marker='+', c='green', s=50, alpha=0.5)
    plt.scatter(neuron[0][1], neuron[1][1], marker='+', c='purple', s=50, alpha=0.5)