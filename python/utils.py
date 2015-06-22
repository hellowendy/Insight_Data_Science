#!/usr/bin/python

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 12:54:57 2015

@author: wenjun

Here stores util functions for data analysis
"""

import math
from sklearn.metrics.pairwise import cosine_similarity
#from scipy.spatial.distance import euclidean
#import numpy as np
import pandas as pd


# Return list of shared item Index of np.Series X and Y
def get_share_items(x, y):
    share = []
    for i in x.index:
        if not math.isnan(x.loc[i]) and i in y.index and not math.isnan(y.loc[i]):
            share.append(i)
    return share
    

#Returns the Pearson correlation coefficient np.Series X and Y.
def sim_pearson(x, y):
    share = get_share_items(x, y)
    
    #return 0 iff shared item number < 2
    if len(share) < 2:
        return 0
    
    #length of shared items
    n = len(share)
    
    sum_x = x.loc[share].sum()
    sum_y = y.loc[share].sum()
    sum_square_x = sum([pow(x.loc[i], 2) for i in share])
    sum_square_y = sum([pow(y.loc[i], 2) for i in share])
    sum_x_y = sum([x.loc[i] * y.loc[i] for i in share])
    
    #Calculate Pearson score (r)
    num = sum_x_y - (sum_x * sum_y / n)
    den = math.sqrt((sum_square_x - pow(sum_x,2)/n) * (sum_square_y - pow(sum_y,2)/n))
    
    if den == 0:
        return 0
    else:
        return num/den
        

# Compute Euclidean distance between np.Series X and Y.
def sim_euclidean(x, y):
    share = get_share_items(x, y)
    
    #return 0 iff no shared item1
    if len(share) < 1:
        return 0
        
    return math.sqrt(sum([pow(x.loc[i]-y.loc[i],2) for i in share]))
    

# Compute the distance score from np.Series X and Y.
def sim_distance(x, y):
    share = get_share_items(x, y)
    
    #return 0 iff no shared item
    if len(share) < 1:
        return 0
    
    #Add up the squares of all differences
    sum_of_squares = sum([pow(x.loc[i]-y.loc[i],2) for i in share])
    
    return 1 / (1 + sum_of_squares)


# Compute cosine distance between np.Series X and Y.
def sim_cosine(x, y):
    share = get_share_items(x, y)
    
    #return 0 iff shared item number < 2
    if len(share) < 2:
        return 0
    
    return cosine_similarity(x.loc[share], y.loc[share])[0][0]


#Returns number of shared items between np.Series X and Y.
def share_len(x, y):
    share = get_share_items(x, y)

    return len(share)


#Return pair-wise distances
def cal_distances(df):
    result = []
    
    for app1 in df.columns:
        for app2 in df.columns:
            # skip when compare to itself
            if app1 == app2:
                continue
            
            # calculate each kind of distance
            share = share_len(df[app1], df[app2])
            eu = sim_euclidean(df[app1], df[app2])
            dis = sim_distance(df[app1], df[app2])
            cos = sim_cosine(df[app1], df[app2])
            r = sim_pearson(df[app1], df[app2])
            
            # store the result as dict
            result.append({'app1': app1,
                           'app2': app2,
                           'common': share,
                           'euclidean': eu,
                           'distance': dis,
                           'cosine': cos,
                           'pearson': r})
    
    # Convert list of dict to pandas dataframe
    out_df = pd.DataFrame([datum for datum in result],
                          columns = ['app1','app2','common','euclidean',
                          'distance','cosine','pearson'])
    return out_df
    

#Calculate scores using given scores DF and similarity matrix SM
def cal_scores(df, sm, similarity='cosine'):
    result = pd.DataFrame(index=df.index, columns=df.columns)
    
    for app in df.columns:
        for ad in df.index:
            score = 0
            weight = 0
            for other in df.columns:
                # skip when compare to itself
                if other == app:
                    continue
                
                pre = df.loc[ad, other]
                
                # continue if no previous score exists
                if not math.isnan(pre):
                    
                    # calculated new distance weighted scores
                    dist = sm.loc[app, other][similarity]
                    weight += dist
                    score += dist * df.loc[ad, other]
                    
            # normalize by the sum of weights
            val = 0
            if not weight == 0:
                val = score / weight
                
            result.loc[ad, app] = val
    
    return result
    

# Recommend ads for an app based on the weighted average
def ads_recommendation(df_old, df_new, app):
    
    #separate indices without record in the old file
    inds = []
    for i in df_old.index:
        if math.isnan(df_old[app][i]):
            inds.append(i)
    
    #extract the predicted scores
    new = df_new[app][inds]