#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
from numpy import *
import pickle

numbers = 102400
#numbers = 10
maxN = 200000

nums = random.random_integers(maxN,size=(numbers,))
print nums
print type(nums)
ff = open('n.txt','w')
pickle.dump(nums,ff)
ff.close()

f = open('n.txt','r')
aux = pickle.load(f)
print(aux)