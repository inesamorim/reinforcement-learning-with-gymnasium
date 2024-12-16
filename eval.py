import numpy as np
import matplotlib.pyplot as plt

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