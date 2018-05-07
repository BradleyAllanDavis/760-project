# -*- coding: utf-8 -*-
"""
Created on Mon May  7 17:09:05 2018

@author: pragathi
"""
import numpy as np

with open("analyze.txt") as f:
    data = f.readlines()

ap_list = []
for line in data:
    class_ = int(line.split(",")[0].split(" ")[2][:-2])
    retrieval = line.split(",")[2].split(":")[1][2:-2].strip(" ").split(" ")
    retrieval = [int(r.strip(".")) for r in retrieval if r is not ""]
    counter = 0
    for i, val in enumerate(retrieval):
        if val==class_:
            counter +=1
            retrieval[i] = counter/(i+1)
        else:
            retrieval[i] = 0
    retrieval = np.array(retrieval)
    ap = np.sum(retrieval)/np.count_nonzero(retrieval)
    if np.isnan(ap):
        ap = 0
    ap_list.append(ap)
    
print(np.average(np.array(ap_list)))
    
    
