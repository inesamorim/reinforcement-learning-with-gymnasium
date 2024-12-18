import gymnasium as gym
from gymnasium.envs.box2d.car_racing import CarRacing
from gymnasium.envs.box2d.car_dynamics import Car
import numpy as np

'''
MODIFICATIONS: add obstacles
			   vary characteristics    (curvature, width, lenght)
			   weather (rain, snow)    (slippery)

REWARDS: - off-tracking    (check if car on the track)
		 + distance traveled    (compute distance in each step)
		 - taking too long to complete    (- small penalty each timestep)


The reward is -0.1 every frame and +1000/N for every track tile visited, 
where N is the total number of tiles visited in the track. 
Ex: finished in 732 frames, reward is 1000 - 0.1*732 = 926.8 points
'''

class EnhancedCarRacing(CarRacing):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.obstacles = []
        self.weather_condition = 'normal'
        self.previous_position = None
        self.total_distance = 0
        self.time_penalty = 0
        self.car = None  # Initialize car as None

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        if self.car is None:
            self.car = Car(self.world, *self.track[0][1:4])
        self.create_obstacles()
        self.modify_track_features()
        self.set_weather()
        self.previous_position = None
        self.total_distance = 0
        self.time_penalty = 0
        return observation, info
    
    
    def get_car_position(self, observation):
        # Implement this method to extract the car's position from the observation
        # This is a placeholder implementation
        return np.array([observation.shape[1] / 2, observation.shape[0]])

    def create_obstacles(self):
        self.obstacles = []
        for _ in range(10):
            # Use slightly narrower limits to ensure obstacles are near the track
            x = np.random.uniform(-0.9, 0.9)
            y = np.random.uniform(-0.9, 0.9)
            
            # Check if the obstacle is on the track
            if self.is_on_track_position(x, y):
                self.obstacles.append((x, y))

    def is_on_track_position(self, x, y):
        # Convert normalized coordinates to pixel coordinates
        pixel_x = int((x + 1) * self.state.shape[1] / 2)
        pixel_y = int((-y + 1) * self.state.shape[0] / 2)  # Note the negative y here
    
        # Check if this pixel is on the track (not green)
        return not np.all(self.state[pixel_y, pixel_x, 1] > 200)

    def modify_track_features(self):
        self.track_width = np.random.uniform(0.8, 1.2)  # Modify track width
        # Note: Modifying curvature and length would require more complex changes to the track generation

    def set_weather(self):
        self.weather_condition = np.random.choice(['normal', 'rain', 'snow'])

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)

        # Get car position
        car_position = self.get_car_position(observation)
        
        # Calculate distances to obstacles
        distances = [np.linalg.norm(car_position - obstacle) for obstacle in self.obstacles]
        
        # Apply penalty for being close to obstacles
        obstacle_penalty = sum(1 / (d + 1e-6) for d in distances)  # Add small epsilon to avoid division by zero
        reward -= obstacle_penalty

        # Apply weather effects
        if self.weather_condition in ['rain', 'snow']:
            action = self.apply_weather_effect(action)

        # Check if car is on track
        if not self.is_on_track(observation):
            reward -= 10

        # Reward for distance traveled
        distance = self.calculate_distance(observation)
        self.total_distance += distance
        reward += distance

        # Time penalty
        self.time_penalty += 0.1
        reward -= self.time_penalty

        # Add obstacles to the observation
        observation = self.add_obstacles_to_observation(observation)

        return observation, reward, terminated, truncated, info

    ''' 
    def apply_weather_effect(self, action):
        # For discrete actions:
        # 0: do nothing, 1: left, 2: right, 3: gas, 4: brake
        if self.weather_condition == 'rain':
            # In rain, there's a chance to slip when turning
            if action in [1, 2]:  # If turning left or right
                if np.random.random() < 0.2:  # 20% chance to slip
                    action = 0  # Do nothing instead of turning
        elif self.weather_condition == 'snow':
            # In snow, gas and brake are less effective
            if action == 3:  # If gas
                if np.random.random() < 0.3:  # 30% chance to reduce effectiveness
                    action = 0  # Do nothing instead of accelerating
            elif action == 4:  # If brake
                if np.random.random() < 0.3:  # 30% chance to reduce effectiveness
                    action = 0  # Do nothing instead of braking
        return action

    def apply_weather_effect(self, action):
        if self.weather_condition == 'rain':
            # In rain, turning actions are less effective
            if isinstance(action, np.ndarray):
                action = np.where(action == 2, 0, action)  # Replace left turn with no action
                action = np.where(action == 1, 0, action)  # Replace right turn with no action
            elif action in [1, 2]:
                action = 0  # Do nothing instead of turning
        elif self.weather_condition == 'snow':
            # In snow, gas and brake are less effective
            if isinstance(action, np.ndarray):
                action = np.where(action == 3, 0, action)  # Reduce gas effectiveness
            elif action == 3:
                if np.random.random() < 0.3:
                    action = 0  # Do nothing instead of accelerating
        return action
    '''
    def apply_weather_effect(self, action):
        if self.weather_condition == 'rain':
            # In rain, turning actions are less effective
            if isinstance(action, np.ndarray):
                action = np.where(action == 2, 0, action)  # Replace left turn with no action
                action = np.where(action == 1, 0, action)  # Replace right turn with no action
            elif action in [1, 2]:
                action = 0  # Do nothing instead of turning
        elif self.weather_condition == 'snow':
            # In snow, gas and brake are less effective
            if isinstance(action, np.ndarray):
                action = np.where(action == 3, 0, action)  # Reduce gas effectiveness
            elif action == 3:
                if np.random.random() < 0.3:
                    action = 0  # Do nothing instead of accelerating
        return action


    def is_on_track(self, observation):
        # Simplified check: assuming green represents off-track
        return not np.all(observation[0:84, :, 1] > 200)

    def calculate_distance(self, observation):
        car_position = self.get_car_position(observation)
        if self.previous_position is None:
            self.previous_position = car_position
            return 0
        distance = np.linalg.norm(car_position - self.previous_position)
        self.previous_position = car_position
        return distance

    def get_car_position(self, observation):
        # Simplified: assume the car is at the center bottom of the image
        return np.array([observation.shape[1] / 2, observation.shape[0]])

    def add_obstacles_to_observation(self, observation):
        for x, y in self.obstacles:
            pixel_x = int((x + 1) * observation.shape[1] / 2)
            pixel_y = int((y + 1) * observation.shape[0] / 2)
            if 0 <= pixel_x < observation.shape[1] and 0 <= pixel_y < observation.shape[0]:
                observation[pixel_y-2:pixel_y+3, pixel_x-2:pixel_x+3] = [255, 0, 0]
        return observation

    def render(self):
        if self.render_mode == "rgb_array":
            return self.add_obstacles_to_observation(super().render())
        return super().render()