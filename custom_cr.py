import gymnasium as gym
from gymnasium.envs.box2d.car_racing import CarRacing, SCALE, ZOOM, WINDOW_H, WINDOW_W, VIDEO_W, VIDEO_H, STATE_W, STATE_H, TRACK_RAD, BORDER_MIN_COUNT, TRACK_TURN_RATE, BORDER, TRACK_DETAIL_STEP, TRACK_WIDTH
from gymnasium.envs.box2d.car_dynamics import Car
import numpy as np
import pygame
import math

from Box2D.b2 import (
    polygonShape,
)

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
    def __init__(self, render_mode=None, obstacle_speed_factor=0.2, moving_obstacles=True):
        super().__init__(render_mode=render_mode, continuous=False)
        self.obstacles = []
        self.obstacle_speed_factor = obstacle_speed_factor
        self.moving_obstacles = moving_obstacles
        self.weather_condition = 'normal'
        self.previous_position = None
        self.total_distance = 0
        self.time_penalty = 0
        self.car = None  # Initialize car as None
        self.max_car_speed = 100
        self.obstacle_speed = self.max_car_speed * self.obstacle_speed_factor
        self.zoom = 0.1  
        self.width = 96
        self.height = 96

        self.track_width = TRACK_WIDTH + self.np_random.uniform(-0.5, 0.5)

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        if self.car is None:
            self.car = Car(self.world, *self.track[0][1:4])
        self.create_obstacles()
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
        track_length = len(self.track)

        for _ in range(10):
            track_pos = np.random.randint(0, track_length)
            x, y = self.track[track_pos][2:4] + self.np_random.uniform(-self.track_width, self.track_width, size=2)

            poly = self.create_obstacle(x, y)
            
            self.obstacles.append(poly)  # Store angle for movement

    def set_weather(self):
        self.weather_condition = np.random.choice(['normal', 'rain', 'snow'])

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)

        # Get car position
        car_position = self.get_car_position(observation)
        
        # Calculate distances to obstacles (using only x and y coordinates)
        distances = [np.linalg.norm(car_position - np.array(obstacle[:2])) for obstacle in self.obstacles]
        
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

        return observation, reward, terminated, truncated, info

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
    
    def create_obstacle(self, x, y):
        num_segments = 8
        radius = self.np_random.uniform(1, 3)
        polygon = []
        for i in range(num_segments):
            angle = 2 * math.pi * i / num_segments
            px = x + radius * math.cos(angle)
            py = y + radius * math.sin(angle)
            polygon.append((px, py))

        # Create a static body for the obstacle
        self.world.CreateStaticBody(
            position=(x, y),
            shapes=polygonShape(vertices=[(p[0]-x, p[1]-y) for p in polygon]),
            userData='obstacle'
        )

        return polygon

    def _render(self, mode: str):
        assert mode in self.metadata["render_modes"]

        pygame.font.init()
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))

        assert self.car is not None
        # computing transformations
        angle = -self.car.hull.angle
        # Animating first second zoom.
        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        scroll_x = -(self.car.hull.position[0]) * zoom
        scroll_y = -(self.car.hull.position[1]) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])

        self._render_road(zoom, trans, angle)

        # Render obstacles - NEW!!!
        obstacle_color = (250, 0, 0)
        for poly in self.obstacles:
            self._draw_colored_polygon(
                self.surf, poly, obstacle_color, zoom, trans, angle
            )

        self.car.draw(
            self.surf,
            zoom,
            trans,
            angle,
            mode not in ["state_pixels_list", "state_pixels"],
        )

        self.surf = pygame.transform.flip(self.surf, False, True)

        # showing stats
        self._render_indicators(WINDOW_W, WINDOW_H)

        font = pygame.font.Font(pygame.font.get_default_font(), 42)
        text = font.render("%04i" % self.reward, True, (255, 255, 255), (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (60, WINDOW_H - WINDOW_H * 2.5 / 40.0)
        self.surf.blit(text, text_rect)

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
        elif mode == "rgb_array":
            return self._create_image_array(self.surf, (VIDEO_W, VIDEO_H))
        elif mode == "state_pixels":
            return self._create_image_array(self.surf, (STATE_W, STATE_H))
        else:
            return self.isopen

    def render_obstacles(self):
        # computing transformations
        angle = -self.car.hull.angle
        # Animating first second zoom.
        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        scroll_x = -(self.car.hull.position[0]) * zoom
        scroll_y = -(self.car.hull.position[1]) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])

        for poly in self.obstacles:
            self._draw_colored_polygon(
                self.surf, poly, self.grass_color, zoom, trans, angle
            )    

    def _create_track(self):
        CHECKPOINTS = 12

        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            noise = self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
            alpha = 2 * math.pi * c / CHECKPOINTS + noise
            rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)

            if c == 0:
                alpha = 0
                rad = 1.5 * TRACK_RAD
            if c == CHECKPOINTS - 1:
                alpha = 2 * math.pi * c / CHECKPOINTS
                self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                rad = 1.5 * TRACK_RAD

            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5 * TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi

            while True:  # Find destination from checkpoints
                failed = True

                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break

                if not failed:
                    break

                alpha -= 2 * math.pi
                continue

            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            # destination vector projected on rad:
            proj = r1x * dest_dx + r1y * dest_dy
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3:
                beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3:
                beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False  # Failed
            pass_through_start = (
                track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
            )
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        if self.verbose:
            print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2 - i1))
        assert i1 != -1
        assert i2 != -1

        track = track[i1 : i2 - 1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2]))
            + np.square(first_perp_y * (track[0][3] - track[-1][3]))
        )
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # Red-white border on hard turns
        border = [False] * len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i - neg - 0][1]
                beta2 = track[i - neg - 1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i - neg] |= border[i]

        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            road1_l = (
                x1 - self.track_width * math.cos(beta1),
                y1 - self.track_width * math.sin(beta1),
            )
            road1_r = (
                x1 + self.track_width * math.cos(beta1),
                y1 + self.track_width * math.sin(beta1),
            )
            road2_l = (
                x2 - self.track_width * math.cos(beta2),
                y2 - self.track_width * math.sin(beta2),
            )
            road2_r = (
                x2 + self.track_width * math.cos(beta2),
                y2 + self.track_width * math.sin(beta2),
            )
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01 * (i % 3) * 255
            t.color = self.road_color + c
            t.road_visited = False
            t.road_friction = 1.0
            t.idx = i
            t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (
                    x1 + side * self.track_width * math.cos(beta1),
                    y1 + side * self.track_width * math.sin(beta1),
                )
                b1_r = (
                    x1 + side * (self.track_width + BORDER) * math.cos(beta1),
                    y1 + side * (self.track_width + BORDER) * math.sin(beta1),
                )
                b2_l = (
                    x2 + side * self.track_width * math.cos(beta2),
                    y2 + side * self.track_width * math.sin(beta2),
                )
                b2_r = (
                    x2 + side * (self.track_width + BORDER) * math.cos(beta2),
                    y2 + side * (self.track_width + BORDER) * math.sin(beta2),
                )
                self.road_poly.append(
                    (
                        [b1_l, b1_r, b2_r, b2_l],
                        (255, 255, 255) if i % 2 == 0 else (255, 0, 0),
                    )
                )
        self.track = track
        return True