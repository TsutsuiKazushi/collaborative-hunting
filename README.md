# Collaborative hunting in artificial agents with deep reinforcement learning

> **[Collaborative hunting in artificial agents with deep reinforcement learning](https://biorxiv.org/cgi/content/short/2022.10.10.511517v1)** <br>
> accepted by *XXXX* 2022 as an article <br>
> Kazushi Tsutsui, Ryoya Tanaka, Kazuya Takeda, Keisuke Fujii

[[biorxiv]](https://biorxiv.org/cgi/content/short/2022.10.10.511517v1)


## Model Architecture
<img src="https://user-images.githubusercontent.com/57206162/207561838-a177918c-19fd-439a-8cf4-9198d6334ff0.jpg" width="800">
An agent’s policy is represented by a deep neural network. An observation of the environment is given as input to the network. An action is sampled from the network’s output, and the agent receives a reward and the subsequent observation. The agent learns to select actions that maximizes cumulative future rewards. In this study, each agent learned its policy network independently, that is, each agent treats the other agents as part of the environment. This illustration shows a case with three predators.

## Examlpes
<img src="https://user-images.githubusercontent.com/57206162/208328484-648f119a-3720-4eea-9dcb-288010818d50.gif" width="800"> <br>
<img src="https://user-images.githubusercontent.com/57206162/208328488-f5dc2576-25d6-4b50-bb72-78a523138f3d.gif" width="800"> <br>
<img src="https://user-images.githubusercontent.com/57206162/208328490-98d76850-8620-4dfe-bb1e-0291c75a8474.gif" width="800"> <br>

The videos are examples of predator(s) (dark blue, blue, and light blue) and prey (red) interactions in each experimental condition. The experimental condition was set as the number of predators (one, two, and three), relative mobility (fast, equal, and slow), and reward sharing (individual and shared), based on ecological findings.


## Setup
- This repository was tested with python 3.6 and 3.7
- To set up the environment, please run the following command: <br>
```pip install -r requirements.txt```

## Training
- To run the train code, please move on the deirectory corresponding to the number of predators (```c1ae```=one-predator, ```c2ae```=two-predator, or ```c3ae```=three-predator). <br>
- Then, please run the python file specifying the predators' movement speed (```3.6```=fast, ```3.0```=equal, or ```2.4```=slow) and whether the reward is shared (```indiv```=individual or ```share```=shared), as follows: <br>
```python c2ae.py 2.4 share```

- The output files (network weights) are in the ```model``` directory.

## Data availability
- The data and models are available in the following figshare repository. These data and models can be used to replicate the figures in the article in the ```notebooks``` directory.
```bash
https://doi.org/10.6084/m9.figshare.21184069.v3
```

## Author
Kazushi Tsutsui ([@TsutsuiKazushi](https://github.com/TsutsuiKazushi)) <br>
E-mail: ```k.tsutsui6<at>gmail.com```
