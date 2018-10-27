[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

## Introduction

This project illustrates how to train an agent to navigate (and collect bananas!) in a large, square world.

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of our agent is to collect as many yellow bananas (healthy) as possible while avoiding blue bananas (rotten).

The state space has 37 dimensions and contains the agent's velocity, along with a ray-based perception of objects around the agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:

- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and to solve the environment, our agent must get an average score of +13 over 100 consecutive episodes.

## Getting started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    _Note_ unfortunately, I could only test the project in OSX 10.11.6.

2. Place the file in the root folder of this repository, and unzip (or decompress) the file.

3. Install all the required dependencies:

    The main requirements of this project are Python==3.6, numpy, matplotlib, jupyter, pytorch and unity-agents. To ease its installation, I recommend the following procedure:

    - [Install miniconda](https://conda.io/docs/user-guide/install/index.html).

      > Feel free to skip this step, if you already have anaconda or miniconda installed in your machine.

      > For OSX users, I would recommend trying the step outlined [here](#Installation-for-conda-and-OSX-users)

    - Creating the environment.

      `conda create -n drlnd-navigation python=3.6`

    - Activate the environment

      `conda activate drlnd-navigation`

    - Installing dependencies.

      `pip install -r requirements.txt`

### Installation for conda and OSX users

You can use the environment [YAML file](environment_osx.yml) provided with repo as follows:

`conda env create -f environment_osx.yml`

## Instructions

Launch a jupyter notebook and follow the tutorial in [Navigation.ipynb](Navigation.ipynb) to train your own agent!

> In case you close the shell running the jupyter server, don't forget to activate the environment. `conda activate drlnd-navigation`

## Do you like the project?

Please gimme a ⭐️ in the GitHub banner 😉. I am also open for discussions especially accompany with ☕ or 🍺.