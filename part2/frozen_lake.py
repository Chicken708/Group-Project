import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

# ====== Hyperparameters you are allowed to tune ======
NUM_EPISODES = 15000            # 訓練 episodes（照作業要求不要改）
MIN_EXPLORATION_RATE = 0.01     # epsilon 最小值，可微調 0.01 ~ 0.1
EPSILON_DECAY_RATE   = 0.001    # epsilon 衰減速度，可微調
INITIAL_EPSILON      = 1.0      # 一開始完全探索
# =====================================================


def print_success_rate(rewards_per_episode: np.ndarray) -> float:
    """Calculate and print the success rate of the agent."""
    total_episodes = len(rewards_per_episode)
    success_count = int(np.sum(rewards_per_episode))
    success_rate = (success_count / total_episodes) * 100.0
    print(f"✅ Success Rate: {success_rate:.2f}% ({success_count} / {total_episodes} episodes)")
    return success_rate


def run(episodes: int, is_training: bool = True, render: bool = False) -> None:
    """Train or evaluate the agent on FrozenLake-v1 (default 4x4)."""
    render_mode = "human" if render else None
    env = gym.make(
        "FrozenLake-v1",           # 預設 4x4，和投影片一致
        is_slippery=True,
        render_mode=render_mode,
    )

    # 建立或載入 Q-table
    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        with open("frozen_lake4x4.pkl", "rb") as f:
            q = pickle.load(f)

    # 這兩個你就先不要動，重點放在 epsilon 設定
    learning_rate_a = 0.1         # alpha：較小的學習率比較穩定
    discount_factor_g = 0.99      # gamma：重視長期回報

    # exploration setting
    if is_training:
        epsilon = INITIAL_EPSILON
    else:
        # evaluation：完全不探索，只用學到的 q-table
        epsilon = 0.0

    rng = np.random.default_rng()
    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False

        while not terminated and not truncated:
            # epsilon-greedy policy（只在訓練時用）
            if is_training and rng.random() < epsilon:
                # explore
                action = env.action_space.sample()
            else:
                # exploit
                action = int(np.argmax(q[state, :]))

            new_state, reward, terminated, truncated, _ = env.step(action)

            # Q-learning 更新（只在訓練時更新）
            if is_training:
                best_next_q = np.max(q[new_state, :])
                td_target = reward + discount_factor_g * best_next_q
                td_error = td_target - q[state, action]
                q[state, action] = q[state, action] + learning_rate_a * td_error

            state = new_state

        # 只在訓練時更新 epsilon（乘法衰減，比較常見）
        if is_training:
            epsilon = max(MIN_EXPLORATION_RATE, epsilon * (1.0 - EPSILON_DECAY_RATE))

        # 成功定義：reward == 1
        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()

    # 訓練時畫 moving-sum 曲線
    if is_training:
        window = 100
        moving_sum = np.zeros(episodes)
        for t in range(episodes):
            moving_sum[t] = np.sum(rewards_per_episode[max(0, t - window + 1) : t + 1])

        plt.figure()
        plt.plot(moving_sum)
        plt.xlabel("Episode")
        plt.ylabel(f"Successes in last {window} episodes")
        plt.title("FrozenLake 4x4 Training Performance")
        plt.tight_layout()
        plt.savefig("frozen_lake4x4.png")
        plt.close()

        # 存 Q-table
        with open("frozen_lake4x4.pkl", "wb") as f:
            pickle.dump(q, f)

    else:
        # 評估階段印出 success rate
        print_success_rate(rewards_per_episode)


if __name__ == "__main__":
    # 1. 訓練（不要改 NUM_EPISODES）
    run(NUM_EPISODES, is_training=True, render=False)

    # 2. 評估 success rate（例如 1000 次）
    EVAL_EPISODES = 1000
    run(EVAL_EPISODES, is_training=False, render=False)
