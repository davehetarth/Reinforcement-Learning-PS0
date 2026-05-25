import numpy as np
import matplotlib.pyplot as plt
from env import ApartmentEnv
from policies import RandomPolicy, ThresholdPolicy, OptimalPolicy

def evaluate_policy(env, policy, num_episodes=10000):
    returns = []
    rejected_all_count = 0
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_return = 0.0
        accepted = False
        
        while not done:
            action = policy.act(obs)
            if action == 1:
                accepted = True
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if terminated:
                ep_return = reward
                
        returns = np.append(returns, ep_return)
        if not accepted:
            rejected_all_count += 1
            
    mean_utility = np.mean(returns)
    std_error = np.std(returns) / np.sqrt(num_episodes)
    fraction_rejected_all = rejected_all_count / num_episodes
    
    return returns, mean_utility, std_error, fraction_rejected_all

def run_benchmarks():
    T, K = 4, 4
    num_episodes = 10000
    base_env = ApartmentEnv(T=T, K=K, noise_std=0.0, seed=42)
    
    print(f"=== PART (c): Baseline Evaluation (T={T}, K={K}, No Noise) ===")
    
    rand_policy = RandomPolicy(T=T)
    opt_policy = OptimalPolicy()
    
    _, rand_mean, rand_se, rand_rej = evaluate_policy(base_env, rand_policy, num_episodes)
    opt_returns, opt_mean, opt_se, opt_rej = evaluate_policy(base_env, opt_policy, num_episodes)
    
    print(f"Random Policy  -> Mean: {rand_mean:.4f} ± {rand_se:.4f} | Rejection Rate: {rand_rej*100:.2f}%")
    print(f"Optimal Policy -> Mean: {opt_mean:.4f} ± {opt_se:.4f} | Rejection Rate: {opt_rej*100:.2f}%")
    
    best_u_min = None
    best_thresh_mean = -1
    thresh_data = {}
    
    for u_min in [1, 2, 3, 4]:
        t_policy = ThresholdPolicy(u_min=u_min)
        t_returns, t_mean, t_se, t_rej = evaluate_policy(base_env, t_policy, num_episodes)
        thresh_data[u_min] = t_returns
        print(f"Threshold ({u_min}) -> Mean: {t_mean:.4f} ± {t_se:.4f} | Rejection Rate: {t_rej*100:.2f}%")
        
        if t_mean > best_thresh_mean:
            best_thresh_mean = t_mean
            best_u_min = u_min
            
    plt.figure(figsize=(10, 5))
    plt.hist(opt_returns, bins=np.arange(6)-0.5, alpha=0.5, label='Optimal Policy', edgecolor='black', rwidth=0.8)
    plt.hist(thresh_data[best_u_min], bins=np.arange(6)-0.5, alpha=0.4, label=f'Best Threshold ({best_u_min})', edgecolor='black', rwidth=0.6)
    plt.title("Distribution of Episode Returns (Noiseless Environment)")
    plt.xlabel("Utility Value")
    plt.ylabel("Frequency Count")
    plt.xticks(range(5))
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("baseline_comparison.png")
    plt.close()
    
    print(f"\n=== PART (d): Noise Robustness Sweep ===")
    sigmas = [0.0, 0.5, 1.0, 2.0]
    
    for sigma in sigmas:
        noisy_env = ApartmentEnv(T=T, K=K, noise_std=sigma, seed=42)
        _, r_m, _, _ = evaluate_policy(noisy_env, rand_policy, num_episodes)
        _, t_m, _, _ = evaluate_policy(noisy_env, ThresholdPolicy(best_u_min), num_episodes)
        _, o_m, _, _ = evaluate_policy(noisy_env, opt_policy, num_episodes)
        
        print(f"Sigma = {sigma:<3} | Random Mean: {r_m:.3f} | Best Fixed Thresh ({best_u_min}) Mean: {t_m:.3f} | Optimal Policy Mean: {o_m:.3f}")

if __name__ == "__main__":
    run_benchmarks()