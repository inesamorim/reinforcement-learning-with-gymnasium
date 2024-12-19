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


def plot_mean_rewards(timesteps, results):
    # Plot mean rewards over timesteps
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, results.mean(axis=1), label="Mean Rewards", color='b')
    plt.fill_between(timesteps, results.mean(axis=1) - results.std(axis=1),
                    results.mean(axis=1) + results.std(axis=1), color='b', alpha=0.3, label="Reward Std Dev")
    plt.xlabel("Timesteps")
    plt.ylabel("Evaluation Rewards")
    plt.title("Evaluation Rewards Over Training Timesteps")
    plt.legend()
    plt.grid()
    plt.show()



def plot_episode_lengths(timesteps, ep_lengths):# Plot episode lengths over timesteps
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, ep_lengths.mean(axis=1), label="Mean Episode Lengths", color='g')
    plt.fill_between(timesteps, ep_lengths.mean(axis=1) - ep_lengths.std(axis=1),
                    ep_lengths.mean(axis=1) + ep_lengths.std(axis=1), color='g', alpha=0.3, label="Episode Length Std Dev")
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Length")
    plt.title("Evaluation Episode Lengths Over Training Timesteps")
    plt.legend()
    plt.grid()
    plt.show()


def compare_rewards(timesteps_baseline, timesteps_custom, results_baseline, results_custom, scale=False):
    # Compare mean rewards
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps_baseline, results_baseline.mean(axis=1), label="Baseline Model", color='purple')
    plt.plot(timesteps_custom, results_custom.mean(axis=1), label="Custom Model", color='pink')
    plt.xlabel("Timesteps")
    plt.ylabel("Evaluation Rewards")
    if scale:
        plt.yscale('symlog')
    plt.title("Comparison of Mean Evaluation Rewards")
    plt.legend()
    plt.grid()
    plt.show()

def compare_rewards_by_algorithm(timesteps_DQN, timesteps_PPO, results_DQN, results_PPO, scale=False):
    # Compare mean rewards
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps_DQN, results_DQN.mean(axis=1), label="DQN Model", color='purple')
    plt.plot(timesteps_PPO, results_PPO.mean(axis=1), label="PPO Model", color='pink')
    plt.xlabel("Timesteps")
    plt.ylabel("Evaluation Rewards")
    if scale:
        plt.yscale('symlog')
    plt.title("Comparison of Mean Evaluation Rewards")
    plt.legend()
    plt.grid()
    plt.show()


def compare_episode_lengths(timesteps_baseline, timesteps_custom, ep_lengths_baseline, ep_lengths_custom):
    # Compare mean episode lengths
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps_baseline, ep_lengths_baseline.mean(axis=1), label="Baseline Model", color='purple')
    plt.plot(timesteps_custom, ep_lengths_custom.mean(axis=1), label="Custom Model", color='pink')
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Length")
    plt.title("Comparison of Mean Evaluation Episode Lengths")
    plt.legend()
    plt.grid()
    plt.show()


def compare_episode_lengths_by_algorithm(timesteps_DQN, timesteps_PPO, ep_lengths_DQN, ep_lengths_PPO):
    # Compare mean episode lengths
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps_DQN, ep_lengths_DQN.mean(axis=1), label="DQN Model", color='purple')
    plt.plot(timesteps_PPO, ep_lengths_PPO.mean(axis=1), label="PPO Model", color='pink')
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Length")
    plt.title("Comparison of Mean Evaluation Episode Lengths")
    plt.legend()
    plt.grid()
    plt.show()



def plot_noise_rewards_comparison(model1_rewards, model2_rewards, model1_label='Model 1', model2_label='Model 2'):
    """
    Plots a bar graph comparing the mean noise rewards of two models with standard deviation as error bars.

    Args:
        model1_rewards (list or np.array): Noise rewards for model 1.
        model2_rewards (list or np.array): Noise rewards for model 2.
        model1_label (str): Label for model 1.
        model2_label (str): Label for model 2.

    Returns:
        None. Displays a bar chart comparing the mean and standard deviation of noise rewards.
    """
    # Compute means and standard deviations
    model1_mean = np.mean(model1_rewards)
    model1_std = np.std(model1_rewards)

    model2_mean = np.mean(model2_rewards)
    model2_std = np.std(model2_rewards)

    # Plot bar graph
    labels = [model1_label, model2_label]
    means = [model1_mean, model2_mean]
    stds = [model1_std, model2_std]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(x, means, yerr=stds, capsize=10, color=['#1f77b4', '#ff7f0e'], alpha=0.7)

    # Add labels
    ax.set_ylabel('Mean Noise Rewards')
    ax.set_title('Comparison of Noise Rewards Between Models')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    # Add standard deviation line above bars
    for i, bar in enumerate(bars):
        plt.plot([bar.get_x(), bar.get_x() + bar.get_width()], [means[i] + stds[i]] * 2, color='black', linestyle='--')

    plt.tight_layout()
    plt.show()


