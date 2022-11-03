# magneticMPM

### Joshua Davy, Peter Lloyd, James H. Chandler and Pietro Valdastri
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
magneticMPM has been tested with Python 3.8 and Taichi 1.1.2. [Anaconda](https://www.anaconda.com/) is highly recommended.

Create Environment
`conda create --name py38 python=3.8`
`conda activate py38`

Install Dependencies
`pip install taichi==1.1.2 numpy trimesh pyyaml matplotlib rtree`

Run
`python beam_bending\main.py`
`python runGripper.py`
`python runTentacle.py`
`python runSmallScaleBot.py`




