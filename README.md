# Tactile-based Door Opening with Franka Panda Robot in Simulation

Official code for our paper [**Sim-to-Real Transfer for Robotic Manipulation with Tactile Sensory**
Zihan Ding, Ya-Yen Tsai, Wang Wei Lee, Bidan Huang *International Conference on Intelligent Robots and Systems (IROS) 2021*](https://arxiv.org/abs/2103.00410) (for simulation part only).

## Description:

![image](https://github.com/quantumiracle/Robotic_Door_Opening_with_Tactile_Simulation/blob/master/img/tactile_robot_door_open.png)

Robot: Franka Emika Panda

* DOF: 7 joints + 1 gripper (symmetric for left/right fingers)
* control mode: velocity control (given velocity target at each timestep) + forward kinematics

Environment: Door Opening Task

Additional sensor: self-developed capacitive tactile sensor (simulated with force sensor in MuJoCo)

Simulator: MuJoCo

Dependencies: robosuite, openai gym, MuJoCo, mujoco-py, torch, etc

## Installation:
* Needs MuJoCo and mujoco-py installed first.

* Check the requirements of *robosuite* package, we use a local version of it.

  ```bash
  pip install -r requirements.txt
  cd environment/robolite
  pip install -e .
  ```

   Install with ```-e .``` so that later change of package *robosuite* will  no longer require re-installation.

## Important Files:

* In ```./environment/robolite/robosuite/models/assets/grippers```  file ```panda_gripper_tactile.xml``` defines the model of Franka gripper with tactile sensors mounted on left and right fingers, also with a 3D-printed adaptors for mounting. ```panda_gripper.xml``` is the original gripper. The tactile sensors are modeled as force sensors in Mujoco, named as "touch_xx_body" in the xml.
* In ```./environment/robolite/robosuite/models/assets/grippers/meshes/panda_gripper```  file ```finger_vis_tactile.stl``` is the 3D cad model of the tactile base on the fingertip used in this project. 
* In ```./environment/robolite/robosuite/models/assets/grippers/meshes/panda_gripper```  file ```finger_vis.stl``` is the original 3D cad model of the Franka gripper finger tip.
* In ```./environment/robolite/robosuite/models/grippers``` file ```panda_gripper_tactile.py``` defines the script of the gripper with tactile sensors. It reads the ```panda_gripper_tactile.xml```. It is called by ```gripper_factor.py```. 
* In ```./environment/robolite/robosuite/models/grippers``` file ```gripper_factory.py``` is called by ```./environments/panda.py```. The robot and the gripper are loaded independently.
* In ```./environment/robolite/robosuite/models/assets/arenas``` file ```table_cabinet_arena.xml``` defines the model of environment with the door (mounted on a cabinet) on the table, with necessary STL files in ```cabinet/```.
* In ```./environment/robolite/robosuite/models/arenas``` file ```table_cabinet_arena.py``` defines the script of the environment.
* In ```./environment/robolite/robosuite/environments``` file ```panda_open_door.py``` defines the robotic door opening task (**important!**). Reward is defined here.
* In ```./environment``` file ```pandaopendoorfktactile.py``` defines a wrapper of door opening task for training RL in gym API.
* ```train.py```  is the main training script.
* ```tactile_finger.py``` is a testing script for testing the simulated tactile sensing in an independent environment. It places a block on a tactile sensor and reads the force distribution. The force readings are saved in ```./data/```.
* ```default_params.py``` defines all the hyperparameters for training, including the ```randomized_params```, which specifies the randomized parameters as defined in ```panda_open_door.py```.
* ```./rl/td3/train_td3.py``` is for training/fine-tuning/testing with TD3 algorithm.

## Start

* Test tactile sensor in an independent environment:

  ```bash
  python tactile_finger.py
  ```

* Train the *PandaOpenDoorFKTactile* environment with RL algorithm TD3:

  ```bash
  python train.py --train --env pandaopendoorfktactile --process 2 
  ```

* Test a trained model with saved in `data/weights/MODEL_TIME/MODEL_INDEX_td3_*`, where ```MODEL_TIME``` indicates the time for training the model and ```MODEL_INDEX``` is an int number indicating at which episode the model is saved:

  ```bash
  python train.py --test --env pandaopendoorfktactile --model MODEL_TIME --model_id MODEL_INDEX --render
  ```
  
  

  
