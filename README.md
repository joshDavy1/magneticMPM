# magneticMPM
## A Framework for Simulation of Magnetic Soft Robots using the Material Point Method 
Paper is avaliable on IEEE Xplore: [Link](https://ieeexplore.ieee.org/document/10103621)

Please Cite:
J. Davy, P. Lloyd, J. H. Chandler and P. Valdastri, "A Framework for Simulation of Magnetic Soft Robots using the Material Point Method," in IEEE Robotics and Automation Letters, doi: 10.1109/LRA.2023.3268016.

[Bibtex](https://github.com/joshDavy1/magneticMPM/blob/main/davy.bib)


### Joshua Davy, Peter Lloyd, James H. Chandler and Pietro Valdastri


Magnetic
soft robots are formed of silicone polymers embedded with
magnetic elements.
Due to the use of external magnetic fields
The relationship between magnetic
soft materials and external sources of magnetic fields present
significant complexities in modelling due to the relationship
between material elasticity and magnetic wrench (forces and
torques) 

We aim to contribute an easy to use simulator for magnetic soft robots capable of modelling dynamic behaviour using the Material Point Method. 

magneticMPM is built with the [Taichi](https://www.taichi-lang.org/) programming language. Install instructions at bottom of README.


## Magnetic Beam Bending
![](https://github.com/joshDavy1/magneticMPM/blob/main/images/beam_bending.gif)


<img src="https://github.com/joshDavy1/magneticMPM/blob/main/images/figure.PNG" width="800" height="350">

## Magnetic Continuum Robot  (Pittiglio et al.)

![](https://github.com/joshDavy1/magneticMPM/blob/main/images/tentacle.gif)
![](https://github.com/joshDavy1/magneticMPM/blob/main/images/tentacle2.gif)

## Small Scale Soft Robot (Hu et al.)

![](https://github.com/joshDavy1/magneticMPM/blob/main/images/sittiWorm.gif)

## Six armed gripper (Xu et al.)

![](https://github.com/joshDavy1/magneticMPM/blob/main/images/gripper.gif)

# Install
magneticMPM has been tested with Python 3.8 and Taichi 1.5.0. [Anaconda](https://www.anaconda.com/) is highly recommended.

Create Environment
`conda create --name py38 python=3.8`
`conda activate py38`

Install Dependencies
`pip install taichi==1.5.0 numpy trimesh pyyaml matplotlib rtree`

Optional
`pip install pyembtree`

Run
`python beam_bending\main.py`
`python runGripper.py`
`python runTentacle.py`
`python runSmallScaleBot.py`



