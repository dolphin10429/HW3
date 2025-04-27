import numpy as np
import matplotlib.pyplot as plt

# 基本設定
n_actions = 10
n_steps = 1000
epsilon = 0.1
c = 2  # UCB參數
tau = 0.5  # Softmax溫度參數

true_rewards = np.random.normal(0, 1, n_actions)

# 各演算法的初始化
Q_eps = np.zeros(n_actions)
N_eps = np.zeros(n_actions)
eps_rewards = []

Q_ucb = np.zeros(n_actions)
N_ucb = np.zeros(n_actions)
ucb_rewards = []

Q_soft = np.zeros(n_actions)
N_soft = np.zeros(n_actions)
soft_rewards = []

Q_thompson_mean = np.zeros(n_actions)
Q_thompson_var = np.ones(n_actions)
thompson_rewards = []

# 模擬過程
for step in range(1, n_steps + 1):
    # Epsilon-Greedy
    if np.random.rand() < epsilon:
        a_eps = np.random.randint(n_actions)
    else:
        a_eps = np.argmax(Q_eps)
    reward = np.random.normal(true_rewards[a_eps], 1)
    N_eps[a_eps] += 1
    Q_eps[a_eps] += (reward - Q_eps[a_eps]) / N_eps[a_eps]
    eps_rewards.append(reward)

    # UCB
    ucb_values = Q_ucb + c * np.sqrt(np.log(step) / (N_ucb + 1e-5))
    a_ucb = np.argmax(ucb_values)
    reward = np.random.normal(true_rewards[a_ucb], 1)
    N_ucb[a_ucb] += 1
    Q_ucb[a_ucb] += (reward - Q_ucb[a_ucb]) / N_ucb[a_ucb]
    ucb_rewards.append(reward)

    # Softmax
    exp_q = np.exp(Q_soft / tau)
    probs = exp_q / np.sum(exp_q)
    a_soft = np.random.choice(np.arange(n_actions), p=probs)
    reward = np.random.normal(true_rewards[a_soft], 1)
    N_soft[a_soft] += 1
    Q_soft[a_soft] += (reward - Q_soft[a_soft]) / N_soft[a_soft]
    soft_rewards.append(reward)

    # Thompson Sampling
    samples = np.random.normal(Q_thompson_mean, np.sqrt(Q_thompson_var))
    a_thompson = np.argmax(samples)
    reward = np.random.normal(true_rewards[a_thompson], 1)
    N = N_eps[a_thompson] + 1  # 用Epsilon-Greedy的拉動次數估算
    Q_thompson_var[a_thompson] = 1 / N
    Q_thompson_mean[a_thompson] += (reward - Q_thompson_mean[a_thompson]) / N
    thompson_rewards.append(reward)

# 畫圖
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(eps_rewards) / (np.arange(n_steps) + 1), label="Epsilon-Greedy")
plt.plot(np.cumsum(ucb_rewards) / (np.arange(n_steps) + 1), label="UCB")
plt.plot(np.cumsum(soft_rewards) / (np.arange(n_steps) + 1), label="Softmax")
plt.plot(np.cumsum(thompson_rewards) / (np.arange(n_steps) + 1), label="Thompson Sampling")

plt.xlabel('Steps')
plt.ylabel('Cumulative Average Reward')
plt.title('Performance of Multi-Armed Bandit Algorithms')
plt.legend()
plt.grid(True)
plt.show()
