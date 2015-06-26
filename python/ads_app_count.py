#!/usr/bin/python

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17

@author: wenjun
"""

from elasticsearch import Elasticsearch
import eslogin

import pandas as pd

def main():
    es = Elasticsearch([{'host': eslogin.host, 'port': eslogin.port}],
                        http_auth=(eslogin.user,eslogin.password))
    
    columns = range(150)
    index = range(360)
    df = pd.DataFrame(index=index, columns=columns)
    df = df.fillna(0)
    
    for col in columns:
        for ind in index:
            df.loc[ind,col] = es.count(index="events-2015.05.*", body={'query': {'bool': {'must':[{'match': { 'ai' : col }}, {'match': { 'cr' : ind }}, {'match': { 'et' : 'AD_SHOW' }}],'must_not':[{'match': { 'fr' : 'true' }}]}}})['count']

    df.to_csv("../data/ad_show_5_2015.tab",sep='\t')

if __name__ == '__main__':
    main()