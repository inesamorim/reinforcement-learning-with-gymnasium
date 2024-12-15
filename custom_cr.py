import gym
from gym.envs.box2d.car_racing import CarRacing
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

class CustomCarRacing(CarRacing):
    def __init__(self):
        super().__init__()
        self.obstacles = self.create_obstacles()
        self.track_length = 0  #distancia

    def create_obstacles(self):
        obstacles = []  #poições random
        for _ in range(5):
            x = np.random.uniform(-1.0, 1.0) #ajustar limites 
            y = np.random.uniform(-1.0, 1.0)
            if self.is_on_track((x,y)): #obstacles apenas na track
            	obstacles.append((x, y))
        return obstacles

    def modify_track_features(self):
		self.track_curvature = random.uniform(0.5, 1.5)
		self.track_width = random.uniform(1.0, 2.5)
    	self.track_length = random.uniform(100.0, 400.0)

    def step(self, action):
        state, reward, done, info = super().step(action)

        if not self.is_on_track(state):  #going off-track
            reward -= 10

        distance_traveled = self.calculate_distance(state)
        reward += distance_traveled #acrescenta a distance

        reward -= 0.01  #small penalty each timestep

        return state, reward, done, info

    def is_on_track(self, state):
        car_x = state[0]
        car_y = state[1]

        #environment resolution 96x96
        pixel_x = int((car_x + 1.2) * (96 / 2))
        pixel_y = int((car_y + 0.6) * (96 / 2))

        img = self.render(mode='rgb_array')

        if 0 <= pixel_x < img.shape[1] and 0 <= pixel_y < img.shape[0]:
        	pixel_color = img[pixel_y, pixel_x]

        	green_min = np.array([0, 150, 0])
        	green_max = np.array([100, 255, 100])

       		if np.all(pixel_color >= green_min) and np.all(pixel_color <= green_max):
            	return False  #on the grass = outside the track
        	return True

        return False

    def calculate_distance(self, state):
        current_position = np.array([state[0], state[1]])

        if self.previous_position is None:
        	self.previous_position = current_position
            return 0.0

        distance_traveled = np.linalg.norm(current_position - self.previous_position)
        self.previous_position = current_position

        return float(distance_traveled)


    #def render(): para obstacles

'''
if __name__ == "__main__":
    from stable_baselines3 import PPO
    env = CustomCarRacing()
    model = PPO("CnnPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    model.save("ppo_custom_carracing")
    obs = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
'''
