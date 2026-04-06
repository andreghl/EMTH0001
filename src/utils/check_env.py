import gymnasium as gym

from gymnasium.utils.env_checker import check_env

def check(env : gym.Env):

    try:
        check_env(env.unwrapped)
        print("Environment passes all checks!")
    except Exception as e:
        print(f"Environment has issues: {e}")
