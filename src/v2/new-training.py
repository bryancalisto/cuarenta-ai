import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from .env import CuarentaEnv
from gymnasium.utils.env_checker import check_env

import numpy as np

from stable_baselines3.common.evaluation import evaluate_policy

env = CuarentaEnv()

try:
    check_env(env)
    print("Environment passes all checks!")
except Exception as e:
    print(f"Environment has issues: {e}")
    
model = RecurrentPPO("MlpLstmPolicy", env , verbose=1)
model.learn(5000)

vec_env = model.get_env()
mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=20, warn=False)
print(mean_reward)

model.save("ppo_recurrent")
del model # remove to demonstrate saving and loading

model = RecurrentPPO.load("ppo_recurrent")

obs = vec_env.reset()
# cell and hidden state of the LSTM
lstm_states = None
num_envs = 1
# Episode start signals are used to reset the lstm states
episode_starts = np.ones((num_envs,), dtype=bool)
while True:
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    episode_starts = dones
    vec_env.render("human")

# 1. Create Environments
# It is best practice to wrap the eval_env in a Monitor to track stats
train_env = gym.make("CartPole-v1")
eval_env = gym.make("CartPole-v1")
eval_env = Monitor(eval_env)

# 2. Setup the EvalCallback
# This will save the best model to the './logs/' directory
eval_callback = EvalCallback(
    eval_env, 
    best_model_save_path="./logs/",
    log_path="./logs/", 
    eval_freq=10,
    deterministic=True, 
    render=False
)

# 3. Train with the callback
model = PPO("MlpPolicy", train_env, verbose=1)
model.learn(total_timesteps=10000, callback=eval_callback)
