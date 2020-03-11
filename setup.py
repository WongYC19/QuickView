#!/usr/bin/env python
# coding: utf-8

# In[1]:


import setuptools
from distutils.core import setup

with open("README.md", "r") as fh:
    long_description = fh.read()
    
    setup(
    name="equitorium", # Replace with your own username
    version="0.0.3",
    author="YC WONG",
    author_email="ycfkjc@hotmail.com",
    description="To find the investment opportunity in Bursa Market (KLSE)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/WongYC19/Equitorium",
    keywords=['bursa', 'KLSE', 'Technical Analysis', 'finance', 'investment', 'screener'],
    packages=setuptools.find_packages(),
    #packages=['numpy', 'pandas', 'bs4', 'tqdm', 'scipy', 'scikit-learn'],
    #install_requires=['pandas', 'tqdm', 'scipy', 'plotly', 'scikit-learn', 'dash', 'dash-bootstrap-components'], #'pyarrow', 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)