import argparse
import datetime
import time
import math
from transforms3d.euler import quat2euler
from shapely.geometry import Point

import os
import json
import datetime
import pathlib
import numpy as np
from PIL import Image
import cv2


import gym
import numpy as np

import torch
from TCP.config import GlobalConfig
from TCP.train import TCP_planner 
from TCP.model import TCP
from collections import OrderedDict
from self_driving.tcp_agent import TcpAgent



SAVE_PATH = "/Users/adnanfidan/Documents/Thesis/data/donkey_data/"


from config import (
    SIMULATOR_NAMES,
    AGENT_TYPES,
    DONKEY_SIM_NAME,
    BEAMNG_SIM_NAME,
    TEST_GENERATORS,
    NUM_CONTROL_NODES,
    MAX_ANGLE,
    NUM_SAMPLED_POINTS,
)
from envs.beamng.config import MAP_SIZE
from factories import make_env, make_agent, make_test_generator
from global_log import GlobalLog
from utils.dataset_utils import save_archive
from utils.randomness import set_random_seed

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", help="Log folder", type=str, default="logs")
parser.add_argument(
    "--env-name", help="Env name", type=str, choices=SIMULATOR_NAMES, required=True
)
parser.add_argument(
    "--donkey-exe-path",
    help="Path to the donkey simulator executor",
    type=str,
    default=None,
)
parser.add_argument(
    "--udacity-exe-path",
    help="Path to the udacity simulator executor",
    type=str,
    default=None,
)
parser.add_argument(
    "--beamng-user-path", help="Beamng user path", type=str, default=None
)
parser.add_argument(
    "--beamng-home-path", help="Beamng home path", type=str, default=None
)
parser.add_argument("--seed", help="Random seed", type=int, default=-1)
parser.add_argument(
    "--add-to-port", help="Modify default simulator port", type=int, default=-1
)
parser.add_argument(
    "--num-episodes", help="Number of tracks to generate", type=int, default=3
)
parser.add_argument(
    "--headless", help="Headless simulation", action="store_true", default=False
)
parser.add_argument(
    "--no-save-archive",
    help="Disable archive storing",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--agent-type", help="Agent type", type=str, choices=AGENT_TYPES, default="random"
)
parser.add_argument(
    "--test-generator",
    help="Which test generator to use",
    type=str,
    choices=TEST_GENERATORS,
    default="random",
)
parser.add_argument(
    "--num-control-nodes",
    help="Number of control nodes of the generated road (only valid with random generator)",
    type=int,
    default=NUM_CONTROL_NODES,
)
parser.add_argument(
    "--max-angle",
    help="Max angle of a curve of the generated road (only valid with random generator)",
    type=int,
    default=MAX_ANGLE,
)
parser.add_argument(
    "--num-spline-nodes",
    help="Number of points to sample among control nodes of the generated road (only valid with random generator)",
    type=int,
    default=NUM_SAMPLED_POINTS,
)
parser.add_argument(
    "--model-path",
    help="Path to agent model with extension (only if agent_type == 'supervised')",
    type=str,
    default=None,
)
parser.add_argument(
    "--predict-throttle",
    help="Predict steering and throttle. Model to load must have been trained using an output dimension of 2",
    action="store_true",
    default=False,
)
# cyclegan options
parser.add_argument(
    "--cyclegan-experiment-name",
    type=str,
    default=None,
    help="name of the experiment. It decides where to store samples and models",
)
parser.add_argument(
    "--gpu-ids",
    type=str,
    default="-1",
    help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU",
)
parser.add_argument(
    "--cyclegan-checkpoints-dir", type=str, default=None, help="models are saved here"
)
parser.add_argument(
    "--cyclegan-epoch",
    type=str,
    default=-1,
    help="which epoch to load? set to latest to use latest cached model",
)

args = parser.parse_args()

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.prev_error = 0
        self.output_limit = 1.0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = max(-self.output_limit, min(self.output_limit, output))
        return output 


class PurePursuitContoller:
    def __init__(self):
        self.waypoints = []  # List of (x, y)
        self.current_waypoint_idx_far = 0
        self.current_waypoint_idx_near = 0
        self.advance_step = 5
        self.lookahead_far = 3
        self.lookahead_near = 1
        self.pid = PIDController(kp=1.0, ki=0.0, kd=0.1)

    def set_waypoints(self, waypoints):
        if len(self.waypoints) == 0:
            self.waypoints = self.smooth_waypoints(waypoints)
            #self.draw_waypoints(self.waypoints)


    def smooth_waypoints(self, waypoints, smoothness=2.0, density=500):
        """Cubic spline ile waypoint yumuşatma"""
        x = [p.x for p in waypoints]
        y = [p.y for p in waypoints]
        from scipy.interpolate import splprep, splev


        # Spline oluştur
        tck, u = splprep([x, y], s=smoothness)
        unew = np.linspace(0, 1.0, num=density)
        smooth_x, smooth_y = splev(unew, tck)

        return [Point(xi, yi) for xi, yi in zip(smooth_x, smooth_y)]
            
    def predict(self, state):
        if "pos" not in state or state["pos"] is None:
            return np.array([0.0, 0.0], dtype=np.float32), None

        curr_x, curr_y = state["pos"]
        rotation = state["rot"]

        w_far = 0.7
        w_near = 0.3

        _, curr_angle, _ = self._calculate_angle_from_rotation(rotation=rotation)

        target_idx_far = self.find_target_waypoint(curr_x, curr_y, curr_angle,self.current_waypoint_idx_far, lookahead=self.lookahead_far)
        self.current_waypoint_idx_far = target_idx_far
        target_far = self.waypoints[self.current_waypoint_idx_far]

        distance_far, angle_far = self.calculate_angle_between_two_points(curr_x=curr_x, curr_y=curr_y, curr_angle=curr_angle, target_x=target_far.x, target_y=target_far.y)

        target_idx_near = self.find_target_waypoint(curr_x, curr_y, curr_angle,self.current_waypoint_idx_near, lookahead=self.lookahead_near)
        self.current_waypoint_idx_near = target_idx_near
        target_near = self.waypoints[self.current_waypoint_idx_near]

        distance_near, angle_near = self.calculate_angle_between_two_points(curr_x=curr_x, curr_y=curr_y, curr_angle=curr_angle, target_x=target_near.x, target_y=target_near.y)

        if abs(angle_far) > math.radians(90):
            geometric_steering = angle_near
        else:
            geometric_steering = angle_far * w_far + angle_near * w_near

        cte_far = distance_far * math.sin(angle_far)
        cte_near = distance_near * math.sin(angle_near)

        cte_ort = cte_far

        dt = 1 / 30.0  
        pid_steering = self.pid.compute(cte_ort, dt) if hasattr(self, 'pid') else 0.0

        steering = geometric_steering + pid_steering
        steering = max(-1.0, min(1.0, steering))

        throttle = max(0.3, min(1.0, distance_near / 5.0))  # min hız = 0.3

        target_point = self.waypoints[len(self.waypoints) - 1]
        results = {
            "far_node" : np.array([target_far.x, target_far.y], dtype=np.float32),
            "near_node" : np.array([target_near.x, target_near.y], dtype=np.float32),
            "curr_angle" : curr_angle,
            "target_node" : np.array([target_point.x, target_point.y], dtype=np.float32),
        }

        return np.array([steering, throttle], dtype=np.float32), results
    
    def is_within_road(self, point_x, point_y, road_width):
        # En yakın merkez hattı noktasını bul
        nearest_wp = min(self.waypoints, key=lambda wp: math.hypot(wp.x - point_x, wp.y - point_y))
        dist_to_center = math.hypot(point_x - nearest_wp.x, point_y - nearest_wp.y)
        return dist_to_center <= (road_width / 2)
    
    def calculate_angle_between_two_points(self, curr_x, curr_y, curr_angle, target_x, target_y):
        dx = target_x - curr_x
        dy = target_y - curr_y
        distance = math.hypot(dx, dy)

        target_angle = math.atan2(dx, dy)
        angle_diff = self.normalize_angle(target_angle - curr_angle)

        max_angle_rad = math.radians(45)
        angle_diff = max(-max_angle_rad, min(max_angle_rad, angle_diff))
        return distance , angle_diff



    def predict_2(self, state):
        if "pos" not in state or state["pos"] is None:
            return np.array([0.0, 0.0], dtype=np.float32)
         
        curr_x, curr_y = state["pos"]
        speed = state["speed"]
        rotation = state["rot"]


        target = self.waypoints[self.current_waypoint_idx]
        dx = target.x - curr_x
        dy = target.y - curr_y
        distance = math.sqrt((dx)**2 + (dy)**2)

        if distance < self.threshold_distance:
            self.current_waypoint_idx = min(self.current_waypoint_idx + self.advance_step, len(self.waypoints) -1)
            target = self.waypoints[self.current_waypoint_idx]
            dx = target.x - curr_x
            dy = target.y - curr_y
            distance = math.sqrt((dx)**2 + (dy)**2)

        _, curr_angle, _ = self._calculate_angle_from_rotation(rotation=rotation)
        target_angle = math.atan2(dx, dy)
        rad_diff = target_angle - curr_angle
        angle_diff = self.normalize_angle(rad_diff)

        cte = distance * math.sin(angle_diff)

        dt = 1 / 30

        pid_steering = self.pid.compute(cte, dt)


        geometric_steering = (angle_diff / math.pi) * 1.5

        steering = max(-1.0 , min(1.0, geometric_steering + pid_steering))

        print(f"[DEBUG] pos=({curr_x:.2f},{curr_y:.2f}) target=({target.x:.2f},{target.y:.2f}) angle_diff={math.degrees(angle_diff):.2f} steering={steering:.2f}, new_steering={delta_safe:.2f}")

        return np.array([steering, distance/15], dtype=np.float32)

    
    def _calculate_angle_from_rotation(self, rotation):
        return quat2euler(rotation, 'sxyz')
    
    def reset(self):
        self.current_waypoint_idx_far = 0
        self.current_waypoint_idx_near = 0
        self.waypoints = []
    
    def normalize_angle(self, angle):
        """Normalize angle to [-π, π]"""
        return (angle + math.pi) % (2 * math.pi) - math.pi
    
    def find_target_waypoint(self, curr_x, curr_y, heading, current_idx, lookahead=1.0):
        for i in range(current_idx, len(self.waypoints)):
            wp = self.waypoints[i]
            dx = wp.x - curr_x
            dy = wp.y - curr_y
            dist = math.hypot(dx, dy)

            # Araç yönü ile hedef arasındaki açı
            target_angle = math.atan2(dx, dy)
            angle_diff = self.normalize_angle(target_angle - heading)

            # Hedef açı ±90° içinde olmalı → yani araç "öne" bakıyor olmalı
            if dist >= lookahead and abs(angle_diff) < math.radians(90):
                return i

        return len(self.waypoints) - 1

    def draw_waypoints(self, waypoints, title="Waypoints"):
        import matplotlib.pyplot as plt

        """
        waypoints: List of (x, y) tuples or numpy array with shape (N, 2)
        """
        #waypoints = list(waypoints)
        x = [pt.x for pt in waypoints]
        y = [pt.y for pt in waypoints]

        plt.figure(figsize=(8, 8))
        plt.plot(x, y, marker='o', linestyle='-', color='blue')
        plt.scatter(x[0], y[0], color='green', s=100, label='Start')
        plt.scatter(x[-1], y[-1], color='red', s=100, label='End')
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.show()



from torchvision import transforms as T

def crop_image_donkey(image: np.ndarray):
    return image[60:, :, :]

def resize_image_donkey(image: np.ndarray):
    return cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)

def bgr2yuv( image: np.ndarray):
    return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

def preprocessing(image: np.ndarray):
    image = crop_image_donkey(image)
    image = resize_image_donkey(image)
    image = bgr2yuv(image)

    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToPILImage(),                  # Convert to PIL Image
        transforms.ToTensor(),                   # Converts to [C, H, W] and scales to [0,1]
    ])

    return transform(image)


import matplotlib.pyplot as plt

def preprocessing_with_show_images(image: np.ndarray):
		plt.imshow(image)
		plt.title("Original image")
		plt.axis('off')
		plt.show()

		image = crop_image_donkey(image)
		plt.imshow(image)
		plt.title("After crop_image_donkey")
		plt.axis('off')
		plt.show()

		image = resize_image_donkey(image)
		plt.imshow(image)
		plt.title("After resize_image_donkey")
		plt.axis('off')
		plt.show()

		image = bgr2yuv(image)
		plt.imshow(image)
		plt.title("After bgr2yuv")
		plt.axis('off')
		plt.show()

		from torchvision import transforms

		transform = transforms.Compose([
			transforms.ToPILImage(),  # Convert to PIL Image
			transforms.ToTensor(),    # [H, W, C] -> [C, H, W] ve [0, 255] -> [0, 1]
		])

		tensor_image = transform(image)

		# Tensoru numpy formatında görselleştirmek için:
		img_np = tensor_image.permute(1, 2, 0).numpy()
		plt.imshow(img_np)
		plt.title("After torchvision transforms (Tensor to numpy)")
		plt.axis('off')
		plt.show()

		return tensor_image



if __name__ == "__main__":

    folder = args.folder
    logger = GlobalLog("collect_images")

    if args.seed == -1:
        args.seed = np.random.randint(2**32 - 1)

    set_random_seed(seed=args.seed)

    test_generator = make_test_generator(
        generator_name=args.test_generator,
        map_size=MAP_SIZE,
        simulator_name=args.env_name,
        agent_type=args.agent_type,
        num_control_nodes=args.num_control_nodes,
        max_angle=args.max_angle,
        num_spline_nodes=args.num_spline_nodes,
    )

    env = make_env(
        simulator_name=args.env_name,
        seed=args.seed,
        port=args.add_to_port,
        test_generator=test_generator,
        donkey_exe_path=args.donkey_exe_path,
        udacity_exe_path=args.udacity_exe_path,
        beamng_home=args.beamng_home_path,
        beamng_user=args.beamng_user_path,
        headless=args.headless,
        beamng_autopilot=args.agent_type == "autopilot",
        cyclegan_experiment_name=args.cyclegan_experiment_name,
        gpu_ids=args.gpu_ids,
        cyclegan_checkpoints_dir=args.cyclegan_checkpoints_dir,
        cyclegan_epoch=args.cyclegan_epoch,
    )

    #model_agent = make_agent(
    #    env_name=args.env_name,
    #    env=env,
    #    model_path=args.model_path,
    #    agent_type=args.agent_type,
    #    predict_throttle=args.predict_throttle,
    #    fake_images=args.cyclegan_experiment_name is not None
    #    and args.cyclegan_checkpoints_dir is not None
    #    and args.cyclegan_epoch != -1,
    #)

    actions = []
    observations = []
    tracks = []
    times_elapsed = []
    is_success_flags = []
    car_position_x_episodes = []
    car_position_y_episodes = []
    episode_lengths = []

    success_sum = 0

    episode_count = 0
    state_dict = dict()
    pure_agent = PurePursuitContoller()
    tcp_model = TcpAgent(env_name=args.env_name, model_path=args.model_path)

    while episode_count < args.num_episodes:
        done, state = False, None
        episode_length = 0
        car_positions_x = []
        car_positions_y = []

        obs = env.reset()
        start_time = time.perf_counter()
        pure_agent.reset()
        target_t = None
        model_action = np.array([0.0, 0.0], dtype=np.float32)
        result_dict = None
        state_dict = {}
        alpha = 0.3
        

        while not done:
            if len(state_dict) != 0 :
                pursuit_action, result_dict = pure_agent.predict(state=state_dict)

                state_dict["next_wp"] = result_dict["near_node"]
                state_dict["compass"] = result_dict["curr_angle"]

                model_action = tcp_model.predict(obs=obs, state=state_dict)
                
                #print("Steering and Throttle = ", model_action)

            if isinstance(model_action, torch.Tensor):
                model_action = model_action.detach().cpu().numpy()


            # Clip Action to avoid out of bound errors
            if isinstance(env.action_space, gym.spaces.Box):
                model_action = np.clip(model_action, env.action_space.low, env.action_space.high)
            obs, done, info = env.step(model_action)

            car_positions_x.append(info["pos"][0])
            car_positions_y.append(info["pos"][1])

            state_dict["cte"] = info.get("cte", None)
            state_dict["cte_pid"] = info.get("cte_pid", None)
            state_dict["speed"] = info.get("speed", None)
            state_dict["pos"] = info.get("pos", None)
            state_dict["rot"] = info.get("rot", None)
            state_dict["rgb"] = info.get("rgb", None)
            
            lateral_position = info.get("lateral_position", None)

            state_dict["last_steering"] = info.get("last_steering", None)
            state_dict["steering"], state_dict["throttle"] = model_action
            state_dict["results_dict"] = result_dict
            waypoints = info.get("track", None).road_points

            pure_agent.set_waypoints(waypoints)


            # FIXME: harmonize the environments such that all have the same action space
            if args.env_name == BEAMNG_SIM_NAME and args.agent_type != "autopilot":
                assert (
                    info.get("throttle", None) is not None
                ), "Throttle is not defined for BeamNG"
                model_action = np.asarray([model_action[0], info.get("throttle")])

            # FIXME: first action is random for autopilots
            if episode_length > 0 and args.agent_type == "autopilot":
                actions.append(model_action)
                observations.append(obs)
            elif args.agent_type != "autopilot" and args.agent_type != "supervised":
                actions.append(model_action)
                observations.append(obs)
            elif args.agent_type == "supervised":
                actions.append(model_action)

            episode_length += 1

            if done:

                times_elapsed.append(time.perf_counter() - start_time)
                car_position_x_episodes.append(car_positions_x)
                car_position_y_episodes.append(car_positions_y)

                if info.get("track", None) is not None:
                    tracks.append(info["track"])

                if info.get("is_success", None) is not None:
                    success_sum += info["is_success"]
                    is_success_flags.append(info["is_success"])

                logger.debug("Episode #{}".format(episode_count + 1))
                logger.debug("Episode Length: {}".format(episode_length))
                logger.debug("Is success: {}".format(info["is_success"]))

                if episode_length <= 5:
                    # FIXME: for very short episodes (see Udacity where there is a bug that causes the CTE to be
                    #  very high at the beginning of the episodes) remove the actions and the observations from
                    #  the data and repeat the episode.
                    logger.warn("Removing short episode")
                    if args.agent_type == "autopilot":
                        original_length_actions = len(actions)
                        original_length_observations = len(observations)
                        items_to_remove = (
                            episode_length - 1
                            if args.agent_type == "autopilot"
                            else episode_length
                        )
                        # first random action of each episode is not included
                        condition = (
                            episode_length > 1
                            if args.agent_type == "autopilot"
                            else episode_length > 0
                        )
                        while condition:
                            actions.pop()
                            observations.pop()
                            episode_length -= 1
                            condition = (
                                episode_length > 1
                                if args.agent_type == "autopilot"
                                else episode_length > 0
                            )

                        assert (
                            len(actions) + items_to_remove == original_length_actions
                        ), "Error when removing actions. To remove: {}, Original: {}, New: {}".format(
                            items_to_remove, original_length_actions, len(actions)
                        )
                        assert (
                            len(observations) + items_to_remove
                            == original_length_observations
                        ), "Error when removing observations. To remove: {}, Original: {}, New: {}".format(
                            items_to_remove,
                            original_length_observations,
                            len(observations),
                        )
                    elif args.agent_type == "supervised":
                        original_length_actions = len(actions)
                        items_to_remove = episode_length
                        while episode_length > 0:
                            actions.pop()
                            observations.pop()
                            episode_length -= 1
                            condition = (
                                episode_length > 1
                                if args.agent_type == "autopilot"
                                else episode_length > 0
                            )

                        assert (
                            len(actions) + items_to_remove == original_length_actions
                        ), "Error when removing actions. To remove: {}, Original: {}, New: {}".format(
                            items_to_remove, original_length_actions, len(actions)
                        )

                    track_to_repeat = tracks.pop()
                    test_generator.set_road_to_generate(road=track_to_repeat)

                else:
                    episode_lengths.append(episode_length)
                    episode_count += 1

                state_dict = {}
                if args.no_save_archive:
                    actions.clear()
                    observations.clear()

    logger.debug("Success rate: {:.2f}".format(success_sum / episode_count))
    logger.debug("Mean time elapsed: {:.2f}s".format(np.mean(times_elapsed)))

    print("is success flags : ", is_success_flags)

    if not args.no_save_archive:
        save_archive(
            actions=actions,
            observations=observations,
            is_success_flags=is_success_flags,
            tracks=tracks,
            car_positions_x_episodes=car_position_x_episodes,
            car_positions_y_episodes=car_position_y_episodes,
            episode_lengths=episode_lengths,
            archive_path=folder,
            archive_name="{}-{}-archive-agent-{}-seed-{}-episodes-{}-max-angle-{}-length-{}".format(
                args.env_name,
                datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
                args.agent_type,
                args.seed,
                args.num_episodes,
                args.max_angle,
                args.num_control_nodes,
            ),
        )

    if args.env_name == BEAMNG_SIM_NAME:
        env.reset()
    else:
        env.reset(skip_generation=True)

    if args.env_name == DONKEY_SIM_NAME:
        time.sleep(2)
        env.exit_scene()
        env.close_connection()

    time.sleep(5)
    env.close()