#!/usr/bin/env python
# coding: utf-8

# In[4]:


import re
import os
from functools import wraps
from collections.abc import Iterable
from logging import getLogger, WARNING
from pkg_resources import get_distribution
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

try:
    from pip import main as pipmain
except ImportError:
    from pip._internal import main as pipmain                

getLogger("pip").setLevel(WARNING)
    
def rerun(n=None):
    if n is None: n = 3
    #assert n >= 1, "Please insert integer value which is greater or equals to 1"            
    def inner_function(function):
        @wraps(function)                        
        def wrapper(*args, **kwargs):                                           
            #global n            
            for _ in range(n):            
                try:                                          
                    return function(*args, **kwargs)                
                except Exception as err_msg:
                    print(err_msg)                             
                else:                    
                    break                                    
        return wrapper
    return inner_function

def install_package(package, verbose=True):
    """ Install packages if module not found. Input: String"""    
    try:
        __import__(package)            
    except:                        
        status = pipmain(['install', package])                
        
        if verbose:
            if status == 0:
                print(f'Installed {get_distribution(package)}')                                          
            else:
                print(f'Fail to install "{package}". Please check if this package is available.')
    else:
        if verbose: print(f'Package {package} already available. Skip installation')  
            
def multithreads(n_threads=None):  
    def inner_function(function):        
        @wraps(function)
        def wrapper(*args, n_threads=n_threads):            
            if n_threads is None: n_threads = os.cpu_count()    
            #n_threads = min(n_threads, len(args))
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                results = executor.map(function, *args)
            
            return list(results)
        return wrapper
    return inner_function
    
def multiprocess(n_cores=None):  
    def inner_function(function):        
        @wraps(function)
        def wrapper(*args, n_cores=n_cores):            
            if n_cores is None: n_cores = os.cpu_count()    
            #n_cores = min(n_cores, len(*args))
            with ProcessPoolExecutor(max_workers=n_cores) as executor:
                results = executor.map(function, *args)
            
            return list(results)
        return wrapper
    return inner_function    

def methods(obj):
    return [func for func in dir(obj) if callable(getattr(obj, func)) and not func.startswith("__")]

methods(list)