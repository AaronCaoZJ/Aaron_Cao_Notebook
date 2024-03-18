# Robosuite v1.4

> `robosuite` is a simulation framework powered by the MuJoCo physics engine for robot learning.

> This release of `robosuite` contains `seven robot  models`, `eight gripper models`, `six controller modes`, and `nine  standardized tasks`. It also offers a modular design of APIs for building new environments with procedural generation. We highlight these primary features below:
>
> - [standardized tasks](https://robosuite.ai/docs/modules/environments.html): a set of standardized manipulation tasks of large diversity and varying complexity and RL benchmarking results for reproducible research;
> - [procedural generation](https://robosuite.ai/docs/modules/overview.html): modular APIs for programmatically creating new environments and new  tasks as combinations of robot models, arenas, and parameterized 3D  objects;
> - [controller supports](https://robosuite.ai/docs/modules/controllers.html): a selection of controller types to command the robots, such as  joint-space velocity control, inverse kinematics control, operational  space control, and 3D motion devices for teleoperation;
> - [multi-modal sensors](https://robosuite.ai/docs/modules/sensors.html): heterogeneous types of sensory signals, including low-level physical states, RGB cameras, depth maps, and proprioception;
> - [human demonstrations](https://robosuite.ai/docs/algorithms/demonstrations.html): utilities for collecting human demonstrations, replaying demonstration  datasets, and leveraging demonstration data for learning.
> - [photorealistic rendering](https://robosuite.ai/docs/modules/renderers.html): integration with advanced graphics tools that provide real-time photorealistic renderings of simulated scenes.

> *ref:*
>
> *Robosuite 官方文档，`https://robosuite.ai/docs/overview.html`*



# A. Environments

1. Running Standardized Environments

```python
import numpy as np
import robosuite as suite

# create environment instance
env = suite.make(
    env_name="Lift", # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

# reset the environment
env.reset()

for i in range(1000):
    action = np.random.randn(env.robots[0].dof) # sample random action
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display
```

2. Building Your Own Environments

```python
# Step 1: Creating the world
from robosuite.models import MujocoWorldBase
world = MujocoWorldBase()

# Step 2: Creating the robot
from robosuite.models.robots import Panda
mujoco_robot = Panda()

# Add gripper to the robot
from robosuite.models.grippers import gripper_factory
gripper = gripper_factory('PandaGripper')
mujoco_robot.add_gripper(gripper)

#  Place the robot on to a desired position and merge it into the world
mujoco_robot.set_base_xpos([0, 0, 0])
world.merge(mujoco_robot)

# Step 3: Creating the table
from robosuite.models.arenas import TableArena
mujoco_arena = TableArena()
mujoco_arena.set_origin([0.8, 0, 0])
world.merge(mujoco_arena)

# Step 4: Adding the object
from robosuite.models.objects import BallObject
from robosuite.utils.mjcf_utils import new_joint
sphere = BallObject(
    name="sphere",
    size=[0.04],
    rgba=[0, 0.5, 0.5, 1]).get_obj()
sphere.set('pos', '1.0 0 1.0')
world.worldbody.append(sphere)

# Step 5: Running Simulation
model = world.get_model(mode="mujoco")

# Question in how to simulate the Env built with MujocoWorldBase
import mujoco
data = mujoco.MjData(model)
while data.time < 1:
    mujoco.mj_step(model, data)
```

3. 重新自定义现有环境程序

> *ref:*
>
> *机器人技能学习-robosuite-0-入门介绍，`https://blog.csdn.net/weixin_42823098/article/details/135201337?spm=1001.2014.3001.5502`*

- 复制`robosuite/robosuite/environments/manipulation`下的操作任务到`My_env`，此处即可尽情修改
- 修改 `robosuite/robosuite/__init__.py`，添加代码:

```python
import sys
sys.path.append("robosuite/robosuite/My_env") 

# Manipulation environments
from robosuite.environments.manipulation.lift import Lift
# from robosuite.environments.manipulation.myEnv import MyEnv
from robosuite.My_env import My_Lift
```



# B. Collect Human demonstrations

> *ref:*
>
> *机器人技能学习-构建自己的数据集并进行训练，`https://blog.csdn.net/weixin_42823098/article/details/135547162?spm=1001.2014.3001.5502`*

- 数据集保存路径：`robosuite/robosuite/models/assets/demonstrations`
