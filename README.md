# Tactile-based Door Opening with Franka Panda Robot in Simulation

## Description:

![Screenshot from 2021-03-01 21-40-09](/home/quantumiracle/Documents/Screenshot from 2021-03-01 21-40-09.png)

Robot: Franka Emika Panda

* DOF: 7 joints + 1 gripper (symmetric for left/right fingers)
* control mode: velocity control (given velocity target at each timestep) + forward kinematics

Environment: Door Opening Task

Additional sensor: self-developed capacitive tactile sensor (simulated with force sensor in MuJoCo)

Simulator: MuJoCo

Dependencies: robosuite, openai gym, etc

## Installation:

* Check the requirements of *robosuite* package, we use a local version of it.

  ```bash
  pip install -r requirements.txt
  cd environment/robolite
  pip install -e .
  ```

   Install with ```-e .``` so that later change of package *robosuite* will  no longer require re-installation.

## Important Files:

* In ```./environment/robolite/robosuite/models/assets/grippers```  file ```panda_gripper_tactile.xml``` defines the model of Franka gripper with tactile sensors mounted on left and right fingers, also with a 3D-printed adaptors for mounting. ```panda_gripper.xml``` is the original gripper.
* In ```./environment/robolite/robosuite/models/grippers``` file ```panda_gripper_tactile.py``` defines the script of the gripper with tactile sensors.
* In ```./environment/robolite/robosuite/models/assets/arenas``` file ```table_cabinet_arena.xml``` defines the model of environment with the door (mounted on a cabinet) on the table, with necessary STL files in ```cabinet/```.
* In ```./environment/robolite/robosuite/models/arenas``` file ```table_cabinet_arena.py``` defines the script of the environment.
* In ```./environment/robolite/robosuite/environments``` file ```panda_open_door.py``` defines the robotic door opening task (**important!**).
* In ```./environment``` file ```pandaopendoorfktactile.py``` defines a wrapper of door opening task for training.
* ```./train.py```  is the training script.
* ```tactile_finger.py``` is a testing script for testing the simulated tactile sensing in an independent environment.

## Start

* Test tactile sensor in an independent environment:

  ```bash
  python tactile_finger.py
  ```

* Train the *PandaOpenDoorFKTactile* environment with RL algorithm TD3:

  ```bash
  python train.py --train --env pandapushfktactile --process 5 
  ```

* Test a trained model with saved in `data/weights/**MODEL_TIME**`  and MODEL_INDEX (an int number indicating at which episode the mode is saved):

  ```bash
  python train.py --test --env pandapushfktactile --model MODEL_TIME --model_id MODEL_INDEX --render
  ```

  
