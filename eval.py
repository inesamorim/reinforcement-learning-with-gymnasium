import numpy as np
import matplotlib.pyplot as plt

#=============================== CALCULATIONS ===============================
def calculate_track_length(env):
    """
    Calculates the total length of the track.
    """
    return sum(np.linalg.norm(np.diff(np.array(env.unwrapped.track), axis=0), axis=1))

def calculate_lap_time(env):
    """
    Calculates the time taken to complete one lap.
    """
    return env.unwrapped.episode_steps / env.unwrapped.track_length

def track_coverage(env, total_track_length):
    """
    Calculates the percentage of the track covered.
    """
    current_distance = env.unwrapped.car.hull.position[0]  # Adjust to your env's state
    return min(100 * current_distance / total_track_length, 100)

def record_agent_dynamics(env, steering_angles, speeds):
    """
    Record the agent's steering angles and speed.
    """
    steering = env.unwrapped.car.wheels[0].joint.angle  # Example for CarRacing
    speed = np.linalg.norm(env.unwrapped.car.hull.linearVelocity)
    steering_angles.append(steering)
    speeds.append(speed)

    return steering_angles, speeds


#=============================== MONITORING ===============================

def detect_collision(env, colisions_count):
    """
    Detect collisions and count them.
    """
    if env.unwrapped.car.hull.contact:  # Example condition, adjust to your env
        collision_count += 1
    return colisions_count

def record_reward_components(env, reward, reward_components):
    """
    Break down the reward into its components.
    """
    # Example: Adjust this logic based on your environment's reward function
    reward_components["survival"].append(reward.get("survival", 0))
    reward_components["speed"].append(reward.get("speed", 0))
    reward_components["safety"].append(reward.get("safety", 0))

    return reward_components

#=============================== PLOTTING ===============================

def plot_rewards(episode_rewards, reward_components):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.bar(reward_components.keys(), reward_components.values())
    plt.title("Reward Components")
    plt.xlabel("Component")
    plt.ylabel("Value")
    plt.grid()
    plt.show()

def plot_metrics(episode_lengths, track_coverages, lap_times):
    """
    Plot the metrics over episodes.
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(episode_lengths)
    plt.title("Episode Lengths")
    plt.xlabel("Episode")
    plt.ylabel("Length")
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(track_coverages)
    plt.title("Track Coverages")
    plt.xlabel("Episode")
    plt.ylabel("Coverage (%)")
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot(lap_times)
    plt.title("Lap Times")
    plt.xlabel("Episode")
    plt.ylabel("Time (s)")
    plt.grid()

    plt.subplot(2, 2, 4)
    
def plot_metric(metric_values, metric_name):
    """
    Plot a given metric over episodes.
    """
    plt.plot(metric_values)
    plt.title(f"{metric_name} Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel(metric_name)
    plt.grid()
    plt.show()