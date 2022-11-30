# NeuBTF

In this repo you can find our implementation of

> **Neural Bidirectional Texture Function Compression and Rendering**
> 
> Luca Quartesan, Carlos Santos

## Requirements
 
We provide an `environment.yml` to install all requirements using anaconda. If you use another package manager this file shoudl still provide the core packages required to run our experiments.

We recommend the use of a GPU with cuda capabilities to obtain expected perfomance.



## Structure

Our pytorch implementation can be found in the folder `/src`

We provide a series of notebooks in the folder `/nbs`:
+ [ubo2014 dataset exploration](/nbs/ubo2014_dataset.ipynb) which also provides a script to download the dataset
+ [ubo2014 training](/nbs/ubo2014_train.ipynb)

The provided code expects the dataset to be stored in `/dataset`, if downloaded with the code provided in the [dataset notebook](/nbs/ubo2014_dataset.ipynb) will be automatically collected there

## Scenes
> To succesfully render in Mitsuba 2 using the provided scenes read [this](scenes/README.md)