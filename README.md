# RobotArm - Isaac Lab Project

This project provides a reinforcement learning environment based on **Isaac Lab** for controlling a **UR10e robot arm** interacting with a workpiece.
The training and evaluation are implemented using the `skrl` framework.
<br />
<br /> 
## Train & Play
### Train
Run the following command to start reinforcement learning training:

```bash
~/IsaacLab/isaaclab.sh -p ~/RobotArm/scripts/skrl/train.py --task Template-Robotarm-v0
```

### Play
Run a trained policy using:

```bash
~/IsaacLab/isaaclab.sh -p ~/RobotArm/scripts/skrl/play.py --task Template-Robotarm-v0
```
<br />
<br />
## Robot Model Configuration
The UR10e robot model is loaded from a USD file.
You can change the robot model path in the following file:

### File

```bash
RobotArm/source/RobotArm/RobotArm/robots/ur10e_w_spindle.py
```

### Code

```python
UR10E_USD_PATH = "/home/eunseop/isaac/isaac_save/ur10e_tuning2.usd"
```

Update the UR10E_USD_PATH variable to use a different UR10e USD model.
<br />
<br />
## Workpiece Configuration
The workpiece is defined as a USD asset in the environment configuration.

### File

```bash
RobotArm/source/RobotArm/RobotArm/robots/ur10e_w_spindle.py
```

### Code

```python
workpiece = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Workpiece",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/eunseop/isaac/isaac_save/flat_surface_2.usd"
    ),
)
```

Change the usd_path to load a different workpiece model.
<br />
<br />
## Mesh Path Change for Reward Computation
The **mesh prim path** must match the structure of the loaded USD file.

### File

```bash
source/RobotArm/RobotArm/tasks/manager_based/robotarm/mdp/rewards.py
```

### Code

```python
mesh_prim = (
    workpiece_prim
    .GetChild("World")
    .GetChild("flat_surface_5")
    .GetChild("mesh_")
    .GetChild("Mesh")
)
```

This mesh path depends on the **internal hierarchy of the USD file** and may need to be modified when using a different workpiece model.

### How to Find the Correct Mesh Path (Isaac Sim)

1. Open the workpiece USD file in Isaac Sim
2. Open the Stage tab on the right
3. Navigate the prim hierarchy to locate the mesh
4. Replicate the hierarchy using GetChild() calls in rewards.py

Example Stage hierarchy:

```markdown
World
 └── flat_surface_5
     └── mesh_
         └── Mesh
```
<br />
<br />
## Notes
When changing the robot USD or workpiece USD, make sure to:
- Update the USD path in the corresponding configuration file
- Update the mesh prim path in rewards.py
- Incorrect mesh paths may lead to reward computation errors or simulation failures.
