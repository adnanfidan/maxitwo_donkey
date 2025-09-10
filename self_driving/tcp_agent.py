from TCP.config import GlobalConfig
from collections import deque
from TCP.train import TCP_planner 
from TCP.model import TCP
from collections import OrderedDict
import torch
from PIL import Image
import numpy as np
from self_driving.agent import Agent
from utils.dataset_utils import preprocess



import math
from transforms3d.euler import quat2euler


class TcpAgent(Agent):
    def __init__(self,
                 env_name: str,
                model_path: str,
                ):

        self.config = GlobalConfig()
        self.net = TCP(self.config)
        self.env_name = env_name

        self.status = 0
        self.alpha = 0.3		
        self.steer_step = 0


        ckpt = torch.load(model_path, map_location=torch.device('cpu'))
        ckpt = ckpt["state_dict"]
        new_state_dict = OrderedDict()
        for key, value in ckpt.items():
            new_key = key.replace("model.","")
            new_state_dict[new_key] = value
        self.net.load_state_dict(new_state_dict, strict = False)
        self.net.eval()

        self.last_steers = deque()
        self.pure_pursuit = PurePursuitContoller()

    def normalize_angle(self, angle):
        """Normalize angle to [-π, π]"""
        while angle >= math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def predict(self, obs, state) -> np.ndarray:

        # Need to be filled
        steering = 0
        throttle = 0
        speed = state["speed"]
        compass = state["compass"]
        next_wp = state["next_wp"]
        pos = state["pos"]
        rgb = obs
        target_point = None
        command = 4

        from torchvision import transforms

        transform = transforms.Compose([
            transforms.ToPILImage(),                  # If image is NumPy array
            transforms.ToTensor(),                   # Converts to [C, H, W] and scales to [0,1]
        ])

        # Image Preprocessing
        rgb = preprocess(image=obs, env_name=self.env_name, fake_images=False)

        rgb = transform(rgb).unsqueeze(0)

        # Target Point Calculation
        theta = compass + np.pi/2
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
            ])

        local_command_point = np.array([next_wp[0]-pos[0], next_wp[1]-pos[1]])
        local_command_point = R.T.dot(local_command_point)
        target_point = tuple(local_command_point)
        target_point = [torch.FloatTensor([target_point[0]]), torch.FloatTensor([target_point[1]])]
        target_point = torch.stack(target_point, dim=1).to(dtype=torch.float32)

        # Next Command
        next_command = [0] * 6
        next_command[command] = 1
        next_command = torch.tensor(next_command).view(1, 6)
        gt_velocity = torch.FloatTensor([speed])

        speed_t = torch.tensor([[speed]], dtype=torch.float32)
        speed_t = speed_t / 12

        model_state = torch.cat([speed_t, target_point, next_command], 1)
        pred = self.net(rgb, model_state, target_point)
        state["target_diff"] = pred["pred_wp"]

        steer_ctrl, throttle_ctrl, brake_ctrl, metadata = self.net.process_action(pred, next_command, gt_velocity, target_point)
        steer_pid, throttle_pid = self.pure_pursuit.predict(state=state)
        #steer_traj, throttle_traj, brake_traj, metadata_traj = self.net.control_pid(pred['pred_wp'], gt_velocity, target_point)
        print("steering_ctrl = ", steer_ctrl)
        print("steering_pid = ", steer_pid)

        status = 1
        if status == 0:
            alpha = 0.3
            steering = np.clip(alpha*steer_ctrl + (1-alpha)*steer_pid, -1, 1)
            throttle = np.clip(alpha*throttle_ctrl + (1-alpha)*throttle_pid, 0, 0.75)
            #breaking = np.clip(alpha*brake_ctrl + (1-alpha)*brake_traj, 0, 1)
        else:
            alpha = 0.3
            steering = np.clip(alpha*steer_pid + (1-alpha)*steer_ctrl, -1, 1)
            throttle = np.clip(alpha*throttle_pid + (1-alpha)*throttle_ctrl, 0, 0.75)
            #breaking = np.clip(alpha*brake_traj + (1-alpha)*brake_ctrl, 0, 1)

        if speed > 38:
            speed_limit = 15  # slow down
        else:
            speed_limit = 38


        throttle = np.clip(
            a=1.0 - steer_ctrl**2 - (speed / speed_limit) ** 2, a_min=0.0, a_max=1.0
        )

        if len(self.last_steers) >= 20:
            self.last_steers.popleft()
        self.last_steers.append(abs(float(steering)))

        num = 0
        for s in self.last_steers:
            if s > 0.10:
                num += 1
        if num > 10:
            self.status = 1
            self.steer_step += 1

        else:
            self.status = 0

        return np.asarray([steer_ctrl, throttle], dtype = np.float32)
    


class PurePursuitContoller:
    def predict(self, state):
        curr_x, curr_y = state["pos"]
        rotation = state["rot"]


        target = state["target_diff"].squeeze(0).detach().cpu().numpy()

        dx = target[0][0]
        dy = target[0][1]
        distance = math.sqrt((dx)**2 + (dy)**2)

        _, curr_angle, _ = self._calculate_angle_from_rotation(rotation=rotation)
        target_angle = math.atan2(dx, dy)
        print("target_angle = ", target_angle)
        print("current_angle = ", curr_angle)

        rad_diff = target_angle - curr_angle
        angle_diff = self.normalize_angle(rad_diff)
        
        steering = max(-1.0 , min(1.0, angle_diff))

        return steering, distance/10

    
    def _calculate_angle_from_rotation(self, rotation):
        return quat2euler(rotation, 'sxyz')
    
    def normalize_angle(self, angle):
        """Normalize angle to [-π, π]"""
        while angle >= math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle