import gymnasium as gym
from env import ApartmentEnv

def run_sanity_check():
    print("--- Running ApartmentEnv Sanity Check ---")
    env = ApartmentEnv(T=4, K=4, seed=42)
    obs, info = env.reset()
    
    done = False
    while not done:
        t, u_observed = obs[0], obs[1]
        action = env.action_space.sample()  
        
        true_u = env.current_U 
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        print(f"Week t: {int(t)} | True Quality U_t: {true_u} | Observed: {u_observed:.2f} | "
              f"Action Taken: {'ACCEPT' if action == 1 else 'REJECT'} | "
              f"Reward Earned: {reward} | Done: {done}")
        obs = next_obs

if __name__ == "__main__":
    run_sanity_check()