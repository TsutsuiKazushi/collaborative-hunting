# Collaborative hunting in artificial agents with deep reinforcement learning
![](https://img.shields.io/badge/python-3.6_|_3.7-blue)

> **[Collaborative hunting in artificial agents with deep reinforcement learning](https://biorxiv.org/cgi/content/short/2022.10.10.511517v1)** <br>
> accepted by *XXXX* 2022 as an article <br>
> Kazushi Tsutsui, Ryoya Tanaka, Kazuya Takeda, Keisuke Fujii

[[biorxiv]](https://biorxiv.org/cgi/content/short/2022.10.10.511517v1)


## Model Architecture
<img src="https://user-images.githubusercontent.com/57206162/207561838-a177918c-19fd-439a-8cf4-9198d6334ff0.jpg" width="800">

## Examlpes
### One-predator
<img src="https://user-images.githubusercontent.com/57206162/207836150-91983d55-519c-4874-aa75-12f7d829043d.gif" width="800"> <br>
### Two-predator
<img src="https://user-images.githubusercontent.com/57206162/207836175-ca2e69e5-02fa-4a55-bf0d-ff77cf58ba29.gif" width="800"> <br>
### Three-predator
<img src="https://user-images.githubusercontent.com/57206162/207836201-bd2da6d5-add5-4a1e-b5c9-f57b15095bca.gif" width="800"> <br>

 From left to right: Fast (indivdual, share), Equal (indivdual, share), Slow (indivdual, share)

## Setup
- This repository is tested with python 3.6 and 3.7
- To set up the environment, please run the following command: <br>
```pip install -r requirements.txt```

## Training
- To run the train code, please move on the deirectory corresponding to the number of predators (```c1ae```=one-predator, ```c2ae```=two-predator, or ```c3ae```=three-predator). <br>
- Then, please run the python file specifying the predators' movement speed (```3.6```=fast, ```3.0```=equal, or ```2.4```=slow) and whether the reward is shared (```indiv``` or ```share```), as follows: <br>
```python c2ae.py 2.4 share```

- The output files (network weights) are in the ```model``` deirectory.
