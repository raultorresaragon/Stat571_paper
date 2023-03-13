#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 10:50:13 2019

@author: marco
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("toy.csv", sep = ",")

ids_, counts = np.unique(df['RID'], return_counts=True) 
pos = np.where(counts == 7)
ids = ids_[pos[0]]

pos = np.isin(df['RID'], ids)
sub_df = df.iloc[pos,:]

D = 7
y = sub_df['ADAS11'].values
y = y.reshape(-1,D)


#################
## removing NA ##
#################

# looking first at the distribution of the observed data
_na = np.isnan(sub_df["ADAS11"])
# pos_na = np.where( _na == True )
obs_val = sub_df["ADAS11"][~_na]
val = np.median(obs_val)
n_bins = int(max(obs_val))
#plt.hist(obs_val, bins = n_bins)

# # NA replacing (median)
# for idx in range(y.shape[0]):
#     _na = np.isnan(y[idx,:])
#     if np.sum(_na == D):
#         print(" No observed data for the individual {}".format(idx))
#         break
#     obs_val = y[idx, ~_na]
#     med = np.median(obs_val)
#     y[idx,_na] = med
    
# all_val = y.reshape(-1,1)    
# plt.hist(all_val, bins = n_bins)  # good!

# NA replacing (Knn)
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=2)
all_val = imputer.fit_transform(y)    



#all_val = (all_val - all_val.mean(axis = 1, keepdims = True))/np.std(all_val)

all_val = (all_val - np.mean(all_val))/np.std(all_val)
#all_val.std(axis = 1, keepdims= True)   
y = all_val.reshape((-1, D))

plt.figure()
sub_s = np.random.choice(np.arange(1,len(y),1), size = 15, replace = False)
for idx in sub_s:
     plt.plot(y[idx], color = "blue" )
plt.xlabel('Date time number')
plt.ylabel('ADAS11')
plt.show()
    
# Scaling time     
X = sub_df['Month'].copy()
X = (X-np.mean(X))/np.std(X)        # why not a standard normalization?
sub_df = sub_df.assign(Time = X) 
X = sub_df['Time'].values[:D]
X = X.reshape((D,1))

#################
## running Fit ##
#################

exec(open("functions_H.py").read())

from sklearn.cluster import KMeans
import pickle
times_ = X.ravel()  # from matrix to array - the same of c() in R

poly_ord = 3
n_init = 1
stock_icl = []
for n_groups in range(1,9):
    print(" \n Number of clusters: {}".format(n_groups))
#    for ctr in range(10):
#        print("\n Run: {}".format(ctr+1)) 
    for run in range(n_init):
        print("\n Run number: {} \n".format(run))  
        kmeans = KMeans(n_clusters=n_groups, random_state = None).fit(y)
        iZ = kmeans.labels_ + 1
        cZ, eta, a, b, alpha, final_icl, loss = Fit(y, times_, n_groups, ia = 1., ib = 1., ieta = .01, ialpha = 10., poly_ord = poly_ord, Z = iZ.copy(), lr = 1e-4, verbose = 1)
        stock_icl.append(final_icl.cpu())
        out_list = [cZ, eta, a, b, alpha, final_icl.cpu()]
        pickle_out = open(("out/out_" + str(n_groups) + "_" + str(run) + ".pickle"), "wb")
        pickle.dump(out_list,  pickle_out)
        pickle_out.close()
#Plot(y, times,poly_ord, cZ, eta, a, b)


##################################
## Looking at the output in out ##
##################################

import pickle
#
store_icl_out = []
outlist_out = []
for n_groups in range(1,9):
    store_icl_in = []
    outlist_in = []
    for run in range(n_init):
        outlist_in.append(pickle.load(open(("out/out_"+str(n_groups) + "_" + str(run) + ".pickle"), "rb" )))
        store_icl_in.append(outlist_in[run][5])
    #best = np.argmax(store_icl_in)
    best = torch.argmax(torch.stack(store_icl_in))
    outlist_out.append(outlist_in[best])
    store_icl_out.append(outlist_out[n_groups-1][5])
    

est_Q = torch.argmax(torch.stack(store_icl_out)).item() + 1
store_icl_in = []
outlist_in = []
for run in range(n_init):
    outlist_in.append(pickle.load(open(("out/out_"+str(est_Q) + "_" + str(run) + ".pickle"), "rb" )))
    store_icl_in.append(outlist_in[run][5])   
#best = np.argmax(store_icl_in)
best = torch.argmax(torch.stack(store_icl_in)).item()
outlist = pickle.load(open(("out/out_"+ str(est_Q) + "_" + str(best) + ".pickle"), "rb" ))
cZ, eta, a, b, alpha, final_icl = outlist
Plot(y, times_,poly_ord, cZ, eta, a, b, n_traj = 20000, SplitFigures = False)
plt.show()

###############
 ## Analisi ##
###############

from collections import Counter

ids_ori = np.unique(sub_df['RID'])

def get_cluster(sub_df, cZ, k):
    pos  = np.where(cZ == k)
    ids_k = ids[pos[0]]
    pos_k = np.isin(sub_df['RID'], ids_k)
    return(pos_k)

def PrintMostCommon(diagnosi, Q = 7):
    m_diagnosi = np.array(diagnosi, dtype = str).reshape(Q,-1, order = 'F')
    m_diagnosi = m_diagnosi[[0,Q-1],:]
    out,counter = np.unique(m_diagnosi,axis=1,return_counts=True) 
    print(out, counter)
    print(' ')
    print('Most common: ',out[:,np.argmax(counter)] )

   
for q in range(est_Q):
    pos_q = get_cluster(sub_df, (cZ-1), q)
    sub_df.iloc[pos_q, :]
    diagnosi = sub_df.iloc[pos_q, :]['DX']
    
    # c = Counter(diagnosi)
    print(' ')
    print('Cluster ' + str(q) + ' :')
    #print(c)
    PrintMostCommon(diagnosi)
    print(' ')

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(y)
y_small =  pca.transform(y)
plt.figure()
plt.scatter(y_small[:,0], y_small[:,1]) #, c = cZ.astype('float'))
plt.xlabel('PCA Axis 1')
plt.ylabel('PCA Axis 2')
plt.show()