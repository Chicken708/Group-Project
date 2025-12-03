import abc
from typing import List, Tuple

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


class Agent(abc.ABC):
    """抽象 Agent 介面：定義一 episode 中的行為流程"""

    def __init__(self):
        # training 代表目前 Agent 是否在訓練模式
        # True：會更新參數（Q-table, 網路權重）
        # False：只做決策，不更新
        self.training: bool = True

    def begin_episode(self, training: bool = True) -> None:
        """
        在每一集開始前呼叫
        用來切換訓練 / 測試模式，並做一些 per-episode 初始化
        子類別可以 override
        """
        self.training = training

    @abc.abstractmethod
    def select_action(self, state: np.ndarray) -> int:
        """
        給定環境回傳的狀態 state，決定要採取的動作
        - state: 通常是 np.ndarray，取決於環境觀測空間
        - 回傳 int: 此 int 對應到離散動作空間中的一個 action index
        """
        raise NotImplementedError

    @abc.abstractmethod
    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        在訓練時更新內部參數
        對應到 RL 轉移 (s, a, r, s', done)：
        - state: 目前狀態 s
        - action: 在 s 執行的動作 a
        - reward: 環境回饋的即時回饋 r
        - next_state: 執行完動作後的下一個狀態 s'
        - done: 是否為終止狀態（episode 結束）
        """
        raise NotImplementedError

    def end_episode(self) -> None:
        """
        每一集結束後呼叫
        預設不做事，子類自行 override
        """
        pass


class RandomAgent(Agent):
    """完全隨機的 Agent，拿來當 baseline（不做學習）"""

    def __init__(self, action_space: gym.Space):
        super().__init__()
        # action_space 是 Gym 的動作空間物件
        # 在 CartPole 為 Discrete(2) => 兩個動作：向左施力 / 向右施力
        self.action_space = action_space

    def select_action(self, state: np.ndarray) -> int:
        # 直接從動作空間中 sample 一個動作
        # 這個 Agent 不關心 state，純隨機
        return int(self.action_space.sample())

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        # 隨機 agent 不學習，所以這裡直接 pass
        # 但為了符合抽象方法介面，仍然保留函式定義
        return


class StateDiscretizer:
    """
    專門負責把 CartPole 連續的觀測值離散化成有限格子 (bins)

    CartPole observation: [x, x_dot, theta, theta_dot]
    其中：
      - x: 小車位置
      - x_dot: 小車速度
      - theta: 杆子的角度（相對垂直）
      - theta_dot: 杆子角速度

    如果直接對連續值建 Q-table，維度是無限的，因此這裡先把每一維切成 n_bins 格
    讓整體的狀態空間變成有限的離散集合，方便用 Q-table
    """

    def __init__(
        self,
        n_bins: Tuple[int, int, int, int] = (6, 6, 12, 12),
    ):
        """
        n_bins: 每一維切割的 bin 數量
          - 第 1 維: x
          - 第 2 維: x_dot
          - 第 3 維: theta
          - 第 4 維: theta_dot
        """
        # 手動設定觀測值的合理範圍，用來做 clip
        # 避免狀態出現非常大的值或 inf，導致離散化失真
        self.obs_low = np.array([-4.8, -3.0, -0.418, -3.5], dtype=np.float32)
        self.obs_high = np.array([4.8, 3.0, 0.418, 3.5], dtype=np.float32)

        # n_bins: 每一維的離散格子數
        self.n_bins = np.array(n_bins, dtype=np.int32)

    def discretize(self, obs: np.ndarray) -> Tuple[int, int, int, int]:
        """
        將連續觀測 obs 轉為 (i, j, k, l) 這種離散索引 tuple
        流程：
        1. 將 obs 夾在 [obs_low, obs_high] 範圍內避免越界
        2. 算出在此範圍內的比例 ratio ∈ [0, 1]
        3. 根據比例把它映射到對應的 bin index
        """
        # 1. clip 到指定範圍
        clipped = np.clip(obs, self.obs_low, self.obs_high)

        # 2. 計算相對位置比例 ratio
        # ratio = (value - low) / (high - low)
        ratio = (clipped - self.obs_low) / (self.obs_high - self.obs_low)

        # 3. 將 [0,1] 區間映射到 [0, n_bins)，再轉成整數 index
        discrete = (ratio * self.n_bins).astype(int)

        # 最後再確保 index 在合法範圍 [0, n_bins-1] 內
        discrete = np.clip(discrete, 0, self.n_bins - 1)

        # 回傳 tuple[int, int, int, int] 才能拿來 index Q-table
        return tuple(int(x) for x in discrete)

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """
        方便取得 (bins_x, bins_xdot, bins_theta, bins_thetadot)
        直接拿來建立 Q-table 的前四維大小
        """
        return tuple(int(x) for x in self.n_bins)


class QLearningCartPoleAgent(Agent):
    """使用 Q-learning 的 CartPole Agent（搭配 StateDiscretizer 做離散化）"""

    def __init__(
        self,
        action_space: gym.Space,
        discretizer: StateDiscretizer,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.0005,
    ):
        """
        參數說明：
        - action_space: Gym 的動作空間，用於知道動作數量（Discrete.n）
        - discretizer: 狀態離散化工具
        - learning_rate: Q-learning 的學習率 alpha
        - discount_factor: 折扣因子 gamma
        - epsilon_start: epsilon-greedy 中初始的探索機率
        - epsilon_min: epsilon 最低值
        - epsilon_decay: 每一集結束後，epsilon 減少的量
        """
        super().__init__()
        self.action_space = action_space
        self.discretizer = discretizer

        # 建立 Q-table
        # 維度 = (bins_x, bins_xdot, bins_theta, bins_thetadot, n_actions)
        # 初始為 0，表示對所有 state-action 沒有偏好
        self.q_table = np.zeros(discretizer.shape + (action_space.n,), dtype=np.float32)

        # Q-learning 相關超參數
        self.alpha = learning_rate      # 學習率
        self.gamma = discount_factor    # 折扣因子

        # epsilon-greedy 相關參數
        self.epsilon = epsilon_start    # 目前的探索機率
        self.epsilon_min = epsilon_min  # 探索機率下限
        self.epsilon_decay = epsilon_decay  # 每集結束時的 epsilon 衰減量

        # 建立一個獨立的隨機數產生器
        self.rng = np.random.default_rng()

    def _get_q_values(self, state_discrete: Tuple[int, int, int, int]) -> np.ndarray:
        """
        給定離散狀態 (i,j,k,l)，回傳該狀態下所有動作的 Q 值向量：
        Q(s, ·) ∈ R^{n_actions}
        """
        return self.q_table[state_discrete]

    def select_action(self, state: np.ndarray) -> int:
        """
        使用 epsilon-greedy 策略選擇動作：
        以 epsilon 的機率隨機選擇 (exploration)
        否則選擇 Q 值最大的動作 (exploitation)

        只有在 self.training=True（訓練模式）時才使用 epsilon-greedy
        測試模式 (training=False) 會完全 greedy
        """
        # 先將連續狀態離散化
        state_d = self.discretizer.discretize(state)

        # 取出此離散狀態下所有動作的 Q 值
        q_values = self._get_q_values(state_d)

        # 訓練模式 + 丟出來的隨機數 < epsilon => 做探索
        if self.training and self.rng.random() < self.epsilon:
            # 探索：隨機取樣一個動作
            return int(self.action_space.sample())
        else:
            # 利用：選擇 Q 值最大的動作（若有多個最大值，np.argmax 回傳第一個 index）
            return int(np.argmax(q_values))

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        在每一步 (s, a, r, s', done) 後，用 Q-learning 規則更新 Q-table：

        Q(s,a) ← Q(s,a) + α [ r + γ max_{a'} Q(s',a') - Q(s,a) ]

        若 done=True，表示 s' 是終止狀態，則目標只會是 r（不再看未來回報）
        """
        # 若是測試模式，直接不更新 Q-table
        if not self.training:
            return

        # 先將 s, s' 轉成離散狀態 index
        s_d = self.discretizer.discretize(state)
        ns_d = self.discretizer.discretize(next_state)

        # 取出下一狀態的所有 Q 值，並找出最優 Q 值
        best_next_q = np.max(self._get_q_values(ns_d))

        # 建立 TD target:
        # 若 done=True => 代表已終止，不再有未來折扣報酬，所以 target = r
        # 若 done=False => target = r + gamma * max_a' Q(s', a')
        td_target = reward + (0.0 if done else self.gamma * best_next_q)

        # 目前狀態行為對應的 Q 值 Q(s,a)
        current_q = self.q_table[s_d + (action,)]

        # TD 誤差 = 目標值 - 目前估計
        td_error = td_target - current_q

        # Q-learning 更新
        self.q_table[s_d + (action,)] = current_q + self.alpha * td_error

    def end_episode(self) -> None:
        """
        每一集結束後呼叫，用來做一些 per-episode 操作
        在這裡用來衰減 epsilon，使得訓練中越到後期越偏向 exploitation
        """
        if self.training:
            # 線性衰減 epsilon，直到不低於 epsilon_min 為止
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)


class Trainer:
    """負責統一訓練與評估 Agent 的流程"""

    def __init__(
        self,
        env_id: str,
        agent: Agent,
        max_steps_per_episode: int = 500,
    ):
        """
        - env_id: Gym 環境名稱，例如 "CartPole-v1"
        - agent: 任意實作了 Agent 介面的智能體（Q-learning, DQN, Random...）
        - max_steps_per_episode: 每一集最多步數上限（避免無窮迴圈）
        """
        self.env_id = env_id
        self.agent = agent
        self.max_steps = max_steps_per_episode

    def _run_episodes(
        self,
        num_episodes: int,
        training: bool,
        render: bool = False,
    ) -> List[float]:
        """
        核心迴圈：執行 num_episodes 集，依照 training flag 決定：
        - training=True: 會進行學習 (agent.learn)
        - training=False: 只測試，不更新參數

        render=True 時會建立能顯示畫面的環境（可能較慢、且需 GUI 支援）。
        """
        # 如需要 human render，就在建立環境時傳入 render_mode
        env_kwargs = {}
        if render:
            env_kwargs["render_mode"] = "human"

        # 建立 Gym 環境
        env = gym.make(self.env_id, **env_kwargs)

        # 紀錄每一集的總 reward
        episode_rewards: List[float] = []

        # 跑 num_episodes 集
        for ep in range(num_episodes):
            # env.reset() 會回傳 (obs, info)，這裡只需 obs
            state, _ = env.reset()

            # 告知 agent 新的一集開始，並指定這一集是訓練或測試模式
            self.agent.begin_episode(training=training)

            total_reward = 0.0  # 累計此集的總 reward

            # 每集最多跑 self.max_steps 步，避免永遠不終止
            for t in range(self.max_steps):
                # 由 agent 根據目前狀態選擇動作
                action = self.agent.select_action(state)

                # 對環境執行動作，取得下一狀態與回饋
                # 新版 Gymnasium step 回傳為:
                # obs, reward, terminated, truncated, info
                # terminated: 環境自然終止 (成功或失敗)
                # truncated: 因時間或其他限制被截斷
                next_state, reward, terminated, truncated, _ = env.step(action)

                # done = True 表示這一集應該結束
                done = bool(terminated or truncated)

                # 如果是訓練模式，用這個轉移做學習
                self.agent.learn(state, action, reward, next_state, done)

                # 累加 reward
                total_reward += reward

                # 往下一狀態前進
                state = next_state

                # 若已終止，本集結束
                if done:
                    break

            # 通知 agent 一集結束
            self.agent.end_episode()

            # 記錄本集 total reward
            episode_rewards.append(total_reward)

            # 若是評估模式，印出每集結果
            if not training:
                print(f"[Eval] Episode {ep+1}/{num_episodes}, reward = {total_reward}")

        # 關閉環境
        env.close()
        return episode_rewards

    def train(self, num_episodes: int) -> List[float]:
        """
        執行訓練模式，會更新 agent 的參數
        回傳每一集的 total reward
        """
        rewards = self._run_episodes(num_episodes, training=True, render=False)
        print(f"Training finished. Last episode reward = {rewards[-1]:.1f}")
        return rewards

    def evaluate(self, num_episodes: int, render: bool = False) -> List[float]:
        """
        執行評估模式，不更新 agent 的參數
        選擇 render=True 來可視化 agent 的表現
        """
        rewards = self._run_episodes(num_episodes, training=False, render=render)
        avg_reward = np.mean(rewards)
        print(f"Evaluation finished. Avg reward over {num_episodes} episodes = {avg_reward:.2f}")
        return rewards


def plot_rewards(
    rewards: List[float],
    window: int = 50,
    filename: str = "cartpole_training.png",
) -> None:
    """
    畫出每集 total reward 以及移動平均曲線

    - rewards: 長度為 N 的 list，rewards[i] 是第 i+1 集的 total reward
    - window: 移動平均的窗口大小
    - filename: 圖片輸出檔名（PNG）
    """
    # 轉成 numpy array 方便計算
    rewards = np.array(rewards, dtype=np.float32)

    # episodes = [1, 2, ..., N] 對應 X 軸
    episodes = np.arange(1, len(rewards) + 1)

    # moving_avg[i] = 從 max(0, i-window+1) 到 i 的平均 reward
    moving_avg = np.zeros_like(rewards)
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        moving_avg[i] = np.mean(rewards[start : i + 1])

    # 建立圖表
    plt.figure(figsize=(8, 4))

    # 每集 reward
    plt.plot(episodes, rewards, label="Reward per episode", alpha=0.4)

    # 移動平均曲線
    plt.plot(episodes, moving_avg, label=f"Moving average (window={window})")

    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("CartPole-v1 training performance")
    plt.legend()
    plt.tight_layout()

    # 儲存成 PNG 檔
    plt.savefig(filename)
    print(f"Training curve saved to {filename}")


if __name__ == "__main__":
    # 使用的環境 ID
    ENV_ID = "CartPole-v1"

    # 建立離散化器，指定每一維的 bins 數量
    discretizer = StateDiscretizer(n_bins=(6, 6, 12, 12))

    # 先建一個暫時的環境，為了讀取 action_space 資訊
    tmp_env = gym.make(ENV_ID)
    action_space = tmp_env.action_space  # 這裡在 CartPole 是 Discrete(2)
    tmp_env.close()
    
    # 學習
    agent = QLearningCartPoleAgent(
        action_space=action_space,
        discretizer=discretizer,
        learning_rate=0.1,      # Q-learning 的 α
        discount_factor=0.99,   # 折扣因子 γ
        epsilon_start=1.0,      # 一開始 epsilon=1.0，完全隨機探索
        epsilon_min=0.05,       # 最低探索機率 5%
        epsilon_decay=0.0005,   # 每一集 epsilon -= 0.0005
    )

    # 建立 Trainer，負責跑完整的訓練 / 評估流程
    trainer = Trainer(env_id=ENV_ID, agent=agent, max_steps_per_episode=500)

    # ========= 訓練階段 =========
    train_episodes = 3000  # 訓練總集數
    training_rewards = trainer.train(num_episodes=train_episodes)

    # 畫出訓練過程的 reward 曲線
    plot_rewards(training_rewards, window=50, filename="cartpole_training.png")

    # ========= 評估階段 =========
    # 評估時不再學習，只是用目前學到的 Q-table 做 greedy 行為
    # 若 render=True，會跳出視覺畫面
    eval_rewards = trainer.evaluate(num_episodes=20, render=True)
    print(f"平均評估 reward = {np.mean(eval_rewards):.2f}")
