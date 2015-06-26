#!/usr/bin/python

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 13:35:35 2015

@author: wenjun
"""

#import sys
#import os
import re

def main():
    header = 'cr\tname\tad_type'
    print header
    f = open('../data/creatives.txt', 'r')
    for line in f:
        m = re.match('^id: (\d+), name: ([\w_ \#]+), format: .+, ad_type: ([\w_]+), categories:',line)
        if m:
            print "%s\t%s\t%s" % (m.group(1), m.group(2), m.group(3))
    f.close()

if __name__ == '__main__':
    main()