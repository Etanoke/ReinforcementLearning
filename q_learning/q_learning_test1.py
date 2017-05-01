u"""「これからの機械学習」の1.3.5に出てきたQ-Learningアルゴリズムを実装した。
内容はほとんどSarsa版と同じだが、学習時に最適方策のQ値を使って行動価値関数を更新するところが異なる。

状態と行動と報酬関数は1.3.2にある図1.3.1の無人島環境を採用した。
エージェントの行動決定には1.1.4に出てきたε-greedyアルゴリズムを使う。

実行した結果次のようになった。この場合a1(s1) -> a1(s2)を選択するのが良いことを学習している（この環境での最適方策）。
ただしεを増やすとs3でa2を選ぶ（探索行動）確率が上がるため、s1でa2を選ぶ方が良いと学習するようになる。

行動価値関数の学習結果
Episode: 10000
  State.STATE1:
    Action.ACTION1 => 10.399999999999823
    Action.ACTION2 => 8.200000000000154
  State.STATE2:
    Action.ACTION1 => 7.319999999999765
    Action.ACTION2 => 9.000000000000066
  State.STATE3:
    Action.ACTION1 => 12.999999999999979
    Action.ACTION2 => -91.6799999999999
"""
import enum
import random
import os


class State(enum.Enum):
    u"""Environmentが取りうる状態"""
    u"""寝起きしている洞窟"""
    STATE1 = 1

    u"""遠回り"""
    STATE2 = 2

    u"""近道（岩場）"""
    STATE3 = 3

    u"""水場（エピソード終了条件）"""
    STATE4 = 4


class Action(enum.Enum):
    u"""エージェントが取りうる行動"""
    u"""近道の経路を進む行動"""
    ACTION1 = 1

    u"""遠回りの経路を進む行動"""
    ACTION2 = 2


class Agent(object):
    u"""エージェント"""
    def __init__(self):
        self.action_value = {}  # Q値
        self.gamma = 0.8  # 割引率γ
        self.alpha = 0.01  # 学習率α
        self.epsilon = 0.01  # ε-greedy
        # Q関数を初期化
        for state in State:
            for action in Action:
                # Q関数の値は適当に10で初期化
                self.action_value[(state, action)] = 10

    def decide_action(self, state, policy=False, greedy=False):
        u"""方策に応じて行動を選択する"""
        if policy:
            # 方策に応じた行動を決定する
            return self._policy()
        elif greedy:
            # 最適の方策を選択する(greedy)
            return self._greedy(state)
        else:
            if random.random() < self.epsilon:
                # ε-greedyアルゴリズムにより、self.epsilonの確率で探索行動を取る
                # 確率的に適当に行動を選択する
                return self._explore()
            else:
                # 方策に応じた行動を決定する
                return self._policy()

    @staticmethod
    def _explore():
        if random.random() < 0.55:
            return Action.ACTION1
        else:
            return Action.ACTION2

    @staticmethod
    def _policy():
        if random.random() < 0.5:
            return Action.ACTION1
        else:
            return Action.ACTION2

    def _greedy(self, state):
        max_q_value = float('-inf')
        max_action = None
        for action in Action:
            q_value = self.action_value[(state, action)]
            if q_value > max_q_value:
                max_q_value = q_value
                max_action = action
        return max_action

    def update_action_value(self, state, action, reward, next_state):
        u"""Q関数を更新する"""
        next_action = self.decide_action(next_state, greedy=True)
        # Q-learningアルゴリズムによってQ関数を更新
        # Q(St, At) ← (1 - α)Q(St, At) + α(Rt+1 + γ * max_a{Q(St+1, At+1))}
        self.action_value[(state, action)] = \
            (1 - self.alpha) * self.action_value[(state, action)] + \
            self.alpha * (reward + self.gamma * self.action_value[(next_state, next_action)])

    def print_action_value(self, episode):
        u"""現在のQ関数の様子を出力する"""
        message = 'Episode: {}{}'.format(episode, os.linesep)
        for state in State:
            if state == State.STATE4:
                continue
            message += '  {}:{}'.format(state, os.linesep)
            for action in Action:
                message += '    {} => {}{}'\
                    .format(action, self.action_value[(state, action)], os.linesep)
        print(message)


class Environment(object):
    u"""環境"""
    def __init__(self):
        self.state = State.STATE1
        self.agent = Agent()

    def reset_state(self):
        u"""状態をエピソード開始時点（初期状態）にリセットする"""
        self.state = State.STATE1

    def update(self, action):
        u"""エージェントの行動に応じて環境の状態を更新する"""
        # エージェントの行動に対する即時報酬を決定
        reward = self.calculate_reward(action)
        # エージェントの行動によって遷移する次の状態を決定
        next_state = self.get_next_state(action)
        # 状態、行動、報酬、次の状態（、次の行動）によってQ関数を更新する
        self.agent.update_action_value(self.state, action, reward, next_state)
        self.state = next_state
        done = self.is_done_episode()
        return done

    def get_next_state(self, action):
        u"""エージェントの行動によって次の状態を決める"""
        if self.state == State.STATE1:
            if action == Action.ACTION1:
                return State.STATE3
            elif action == Action.ACTION2:
                return State.STATE2
        elif self.state == State.STATE2:
            if action == Action.ACTION1:
                return State.STATE1
            elif action == Action.ACTION2:
                return State.STATE4
        elif self.state == State.STATE3:
            if action == Action.ACTION1:
                return State.STATE4
            elif action == Action.ACTION2:
                return State.STATE1

    def is_done_episode(self):
        u"""エピソード終了判定"""
        return self.state == State.STATE4

    def calculate_reward(self, action):
        u"""報酬を計算する

        Args:
            action(Action): Agentが取った行動

        Returns:
            (int): 報酬
        """
        if self.state == State.STATE1:
            if action == Action.ACTION1:
                return 0
            elif action == Action.ACTION2:
                return +1
        elif self.state == State.STATE2:
            if action == Action.ACTION1:
                return -1
            elif action == Action.ACTION2:
                return +1
        elif self.state == State.STATE3:
            if action == Action.ACTION1:
                return +5
            elif action == Action.ACTION2:
                return -100
        raise ValueError(self.state, action)


def training():
    environment = Environment()
    episode_num = 10000
    for episode in range(1, episode_num + 1):
        # エピソード開始時点で状態をリセットする
        environment.reset_state()
        # 長くても100ステップで終了する（実際にはSTATE4で終了するため、100STEPもいくことは確率的にほぼない）
        for step in range(100):
            # エージェントが取る行動を決定
            action = environment.agent.decide_action(environment.state)
            # エージェントの行動に応じて環境の状態を更新する
            is_done = environment.update(action)
            if is_done:
                break
        # エピソード終了
        if episode % (episode_num // 100) == 0:
            environment.agent.print_action_value(episode)

if __name__ == '__main__':
    training()
