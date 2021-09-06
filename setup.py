# -*- coding: utf-8 -*-

from setuptools import setup


setup(
    # name of the package
    name='rankorder', 
    
    # version of package
    version='0.1.0', 
    
    # package description
    description='rankorder: universal rank-order method to analyze noisy data', 
    
    # author information
    author='Samuel Häusler', 
    url='https://github.com/samuehae/rankorder', 
    
    # package license
    license='MIT', 
    
    # packages to process (build, distribute, install)
    packages=['rankorder'], 
    
    # required packages
    install_requires=['numpy'], 
    extras_require={'examples': ['matplotlib', 'scipy'], }
)
