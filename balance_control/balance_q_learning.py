u"""OpenAI gymのCarPole-v0をQ-Learningでやってみる

環境から取れる状態の情報は位置、速度、角度、角速度のみの組み合わせ。

#角度6通り * 角速度6通りの36パターンで試す。
#位置と横方向の速度も取れるが、状態数が爆発する。
状態は位置4通り * 速度4通り * 角度6通り * 角速度6通りの576パターン。
状態数が増えるほど学習に時間がかかり、しきい値も決めないといけない。

行動空間は状態によらず、左か右へ全力移動(env.action_space)。
移動しないというのはgymの都合で存在しない。

角度6通り * 角速度6通りでの学習が終わると、右か左にちょっとずつ動き続け、端に到達して終了するようになる。
角度角速度 + 位置速度で学習させると、角度のみより生存step数は小さい値で収束し、真ん中あたりで倒れる。
※しきい値とかハイパーパラメータを吟味すればもっと伸びるだろうが、これが難しい。

→報酬関数を調整したらかなりいい感じになった。
gymのenvが返す報酬がずっと1だったので、位置と角度に応じて適当な報酬を設定したら200step（gymの上限）まで生きるようになった。
"""
import enum
import time
import random

import gym

env = gym.make('CartPole-v0')
print(env.action_space)

STATE_PATTERN = list(range(4 * 4 * 6 * 6))


def get_state_no(observation):
    u"""角度と角速度から適当に状態番号をふる
    
    ここの振り方がかなり適当なので、改善の余地はたくさんあるはず
    Args:
        observation(tuple): 観測結果（x, x_dot, theta, theta_dot)
    Returns:
        (int): 状態番号
    """
    x, x_dot, angle, angle_rate = observation
    x_threshold = 0.04
    if x < -x_threshold:
        x_no = 0
    elif x < 0:
        x_no = 1
    elif x < x_threshold:
        x_no = 2
    else:
        x_no = 3

    x_dot_threshold = 0.2
    if x_dot < -x_dot_threshold:
        x_dot_no = 0
    elif x_dot < 0:
        x_dot_no = 1
    elif x_dot < x_dot_threshold:
        x_dot_no = 2
    else:
        x_dot_no = 3

    angle_threshold1 = 0.03
    angle_threshold2 = 0.06
    if angle < -angle_threshold2:
        angle_no = 0
    elif angle < -angle_threshold1:
        angle_no = 1
    elif angle < 0:
        angle_no = 2
    elif angle < angle_threshold1:
        angle_no = 3
    elif angle < angle_threshold2:
        angle_no = 4
    else:
        angle_no = 5

    angle_rate_threshold_1 = 0.15
    angle_rate_threshold_2 = 0.3
    if angle_rate < -angle_rate_threshold_2:
        angle_rate_no = 0
    elif angle_rate < -angle_rate_threshold_1:
        angle_rate_no = 1
    elif angle_rate < 0:
        angle_rate_no = 2
    elif angle_rate < angle_rate_threshold_1:
        angle_rate_no = 3
    elif angle_rate < angle_rate_threshold_2:
        angle_rate_no = 4
    else:
        angle_rate_no = 5

    return x_no * 6 * 6 * 2 + x_dot_no * 6 * 6 + angle_no * 6 + angle_rate_no
    # return angle_no * 6 + angle_rate_no


def get_reward(observation):
    x, _, angle, _ = observation
    loss = abs(x) + abs(angle) * 5
    return -loss


class Action(enum.Enum):
    u"""エージェントが取りうる行動"""
    u"""左に全力"""
    ACTION1 = 0

    u"""右に全力"""
    ACTION2 = 1


class Agent(object):
    u"""エージェント"""
    def __init__(self):
        self.action_value = {}  # Q値
        self.gamma = 0.95  # 割引率γ
        self.alpha = 0.15  # 学習率α
        self.epsilon = 0.15  # 探索率ε
        # Q関数を初期化
        for state in STATE_PATTERN:
            for action in Action:
                # Q関数の値は適当に初期化
                self.action_value[(state, action)] = random.random() * 10

    def decide_action(self, state_no, policy=False, greedy=False):
        u"""方策に応じて行動を選択する"""
        if policy:
            # 方策に応じた行動を決定する
            return self._policy()
        elif greedy:
            # 最適の行動を選択する(greedy)
            return self._greedy(state_no)
        else:
            if random.random() < self.epsilon:
                # ε-greedyアルゴリズムにより、self.epsilonの確率で探索行動を取る
                # 確率的に適当に行動を選択する
                return self._explore()
            else:
                # 最適の行動を選択する(greedy)
                return self._greedy(state_no)

    @staticmethod
    def _explore():
        if random.random() < 0.5:
            return Action.ACTION1
        else:
            return Action.ACTION2

    @staticmethod
    def _policy():
        if random.random() < 0.5:
            return Action.ACTION1
        else:
            return Action.ACTION2

    def _greedy(self, state_no):
        max_q_value = float('-inf')
        max_action = None
        for action in Action:
            q_value = self.action_value[(state_no, action)]
            if q_value > max_q_value:
                max_q_value = q_value
                max_action = action
        return max_action

    def update_action_value(self, state_no, action, reward, next_state_no):
        u"""Q関数を更新する"""
        next_action = self.decide_action(next_state_no, greedy=True)
        # Q-learningアルゴリズムによってQ関数を更新
        # Q(St, At) ← (1 - α)Q(St, At) + α(Rt+1 + γ * max_a{Q(St+1, At+1))}
        self.action_value[(state_no, action)] = \
            (1 - self.alpha) * self.action_value[(state_no, action)] + \
            self.alpha * (reward + self.gamma * self.action_value[(next_state_no, next_action)])


def train(agent, needs_render=False):
    steps_log = []
    for i_episode in range(10000):
        observation = env.reset()
        t = 0
        for t in range(200):
            state_no = get_state_no(observation)
            # 環境の値に応じてエージェントが行動を選択する
            action = agent.decide_action(state_no)
            # エージェントの行動に応じて環境の状態が変わる
            observation, _reward, done, info = env.step(action.value)
            # 選んだ行動と次の状態、報酬が決まったので行動価値関数を更新する(=学習)
            reward = get_reward(observation)
            next_state_no = get_state_no(observation)
            agent.update_action_value(state_no, action, reward, next_state_no)
            if done:
                # print("Episode finished after {} time steps".format(t+1))
                break
        if i_episode > 0 and i_episode % 100 == 0:
            saikin = steps_log[-100:]
            print('episode: {} / 平均生存step: {}steps'.format(i_episode + 1, sum(saikin) / len(saikin)))
        if needs_render and i_episode in [1, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400]:
            evaluate(agent, 1)
        steps_log.append(t + 1)

    return agent


def evaluate(agent, episode_num=100):
    for i_episode in range(episode_num):
        observation = env.reset()
        t = 0
        for t in range(200):
            env.render()
            time.sleep(0.02)
            state_no = get_state_no(observation)
            # 環境の値に応じてエージェントが行動を選択する
            action = agent.decide_action(state_no, greedy=True)
            # エージェントの行動に応じて環境の状態が変わる
            observation, reward, done, info = env.step(action.value)
            # print(t, get_reward(observation), reward)
            if done:
                break
        print('evaluate: {}step生存'.format(t + 1))

if __name__ == '__main__':
    agent = Agent()
    try:
        # needs_render=Trueにすると学習経過が見られる
        agent = train(agent, needs_render=True)
    except KeyboardInterrupt:
        # Ctrl+Cで学習を中断して結果確認する
        pass
    evaluate(agent, episode_num=100)
