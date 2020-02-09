#!/usr/bin/env python
# coding: utf-8

# In[1]:


import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="equitorium", # Replace with your own username
    version="0.0.2",
    author="YC WONG",
    author_email="ycfkjc@hotmail.com",
    description="To identify the investment opportunit in Bursa Market (KLSE)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/WongYC19/EquityAnalyzer",
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'pandas', 'tqdm', 'scipy', 'pyarrow', 'plotly', 'scikit-learn', 'dashtable'],
    #packages=['numpy', 'pandas', 'bs4', 'tqdm', 'scipy', 'scikit-learn'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)