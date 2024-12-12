import gymnasium as gym
import time

def run_human_rendering():
    # Create the CarRacing environment with human rendering
    env = gym.make("CarRacing-v3", render_mode="human")

    try:
        obs, info = env.reset()  # Initialize the environment
        done = False

        while not done:
            action = env.action_space.sample()  # Take a random action
            obs, reward, done, truncated, info = env.step(action)  # Step through the environment
            time.sleep(0.01)  # Add a small delay to slow down the rendering for better visibility

            if done or truncated:
                print("Episode finished!")
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        env.close()  # Make sure to close the environment properly

if __name__ == "__main__":
    run_human_rendering()