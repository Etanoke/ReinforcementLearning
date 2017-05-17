u"""OpenAI gymのCarPole-v0をQ-Learning（Neural Network版）で学習する"""
import enum
import time
import random

import gym

env = gym.make('CartPole-v0')


def get_reward(observation):
    """環境の値から報酬を計算する"""
    x, _, angle, _ = observation
    loss = abs(x + 1) ** 2 + abs(angle * 5 + 1) ** 2
    return -loss


class NeuralNetwork(object):
    """ニューラルネットワークの学習管理クラス"""
    INPUT_LAYER_NEURONS = 4  # 入力層ニューロン数
    HIDDEN_LAYER_NEURONS = 16  # 隠れ層ニューロン数
    OUTPUT_LAYER_NEURONS = 2  # 出力層ニューロン数 = Action数

    LEARNING_RATE = 1e-3  # 学習率

    def __init__(self):
        # とりあえずバイアス項はなし
        self.params = {
            'W_INPUT': [[random.random() for _ in range(self.HIDDEN_LAYER_NEURONS)]
                        for _ in range(self.INPUT_LAYER_NEURONS)],  # 入力層 - 隠れ層間の重み
            'W_HIDDEN': [[random.random() for _ in range(self.OUTPUT_LAYER_NEURONS)]
                         for _ in range(self.HIDDEN_LAYER_NEURONS)],  # 隠れ層 - 出力層間の重み
        }
        self.output = None

    def forward(self, x_input, should_save_output=False):
        """ネットワークを順伝搬させて出力を計算する
        
        入力層のニューロンは順に1, 2, ..., i, ...
        隠れ層と出力層も同様にj, kと添字をふることにする
        ここではuは入力値と重みの総和、φは任意の活性化関数、yはニューロンの出力とする
        φ_hは隠れ層の活性化関数。ここではReLUを使う
        φ_oは隠れ層の活性化関数。ここでは恒等関数を使う
        """
        # 隠れ層の計算
        # u_j = Σ_i { x_i * w_ij }
        u_hidden = self._poor_dot(x_input, self.params['W_INPUT'])
        # y_j = φ_h(u_j)
        y_hidden = self.relu(u_hidden)  # 活性化関数はReLU

        # 出力層の計算
        # u_k = Σ_j { y_j * w_jk }
        u_output = self._poor_dot(y_hidden, self.params['W_HIDDEN'])
        # y_k = φ_o(u_k)
        # y_output = self.sigmoid(u_output)  # 活性化関数はシグモイド関数
        y_output = u_output  # 活性化関数は恒等関数

        # 誤差逆伝搬で使う出力値
        if should_save_output:
            self.output = {
                'u_hidden': u_hidden,
                'y_hidden': y_hidden,
                'y_output': y_output,
            }
        return y_output

    def back_propagation(self, x_input, target):
        """誤差逆伝搬でネットワークの重みを更新する
        
        誤差関数Eは、出力が連続値であるため自乗平均をとる
        targetは教師信号の値
        E = Σ_k{ (target_k - y_k)^2 } / 2
        
        隠れ層 - 出力層間の重みは次の式で更新する
        ηは学習率とする
        w_jk = w_jk - η * Δw_jk
        Δw_jk = ∂E/∂w_jk
              = ∂E/∂y_k * ∂y_k/∂u_k * ∂u_k/∂w_jk
              = (y_k - target_k) * φ_o'(u_k) * y_j
        ここで
        δ_output_k = (y_k - target_k) * φ_o'(u_k)
        とおいておく
        
        入力層 - 隠れ層間の重みは
        w_ij = w_ij - η * Δw_ij
        Δw_ij = ∂E/∂w_ij
              = Σ_k{ ∂E/∂y_k * ∂y_k/∂u_k * ∂u_k/∂y_j * ∂y_j/∂u_j * ∂u_j/∂x_i }
              = Σ_k{ (y_k - target_k) * φ_h'(u_k) * w_jk * φ'(u_j) * x_i }
              = Σ_k{ δ_output_k * w_jk * φ_h'(u_j) * x_i }
              = Σ_k{ δ_output_k * w_jk } * φ_h'(u_j) * x_i
        """
        if self.output is None:
            return
        # 誤差逆伝搬では順伝搬で計算したニューロン出力値を使う
        u_hidden = self.output['u_hidden']
        y_hidden = self.output['y_hidden']
        y_output = self.output['y_output']

        # 隠れ層 - 出力層間の重みを更新
        # 出力層の活性化関数は恒等関数なので、φ_o'(u_k) = 1
        delta_o = []
        for y_output_k, target_k in zip(y_output, target):
            delta_o.append(y_output_k - target_k)

        for j, y_hidden_j in enumerate(y_hidden):
            for k, delta_o_k in enumerate(delta_o):
                self.params['W_HIDDEN'][j][k] += -self.LEARNING_RATE * y_hidden_j * delta_o_k

        # 入力層 - 隠れ層間の重みを更新
        # φ_h'(u_j)はReLUの微分
        delta_relu = [value > 0 for value in u_hidden]
        # delta_w1_tmpは　Σ_k{ δ_output_k * w_jk } * φ_h'(u_j) までの計算
        delta_w1_dot = self._poor_dot(delta_o, self.params['W_HIDDEN'])
        delta_w1_tmp = []
        for delta_relu_j, delta_w1_dot_j in zip(delta_relu, delta_w1_dot):
            delta_w1_tmp.append(delta_relu_j * delta_w1_dot_j)

        for i, x_input_i in enumerate(x_input):
            for j, delta_w1_j in enumerate(delta_w1_tmp):
                self.params['W_INPUT'][i][j] += -self.LEARNING_RATE * x_input_i * delta_w1_j

    @staticmethod
    def relu(inputs):
        """活性化関数ReLU"""
        return [value if value > 0 else 0 for value in inputs]

    @staticmethod
    def _poor_dot(value_1d, value_2d):
        u"""np.dotの代用。1次元配列と2次元配列のみ受け付ける"""
        outputs = [0] * len(value_2d[0])
        for input_, weight_i in zip(value_1d, value_2d):
            for j, weight in enumerate(weight_i):
                outputs[j] += input_ * weight
        return outputs


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
        self.epsilon = 0.15  # 探索率ε
        self.network = NeuralNetwork()

    def decide_action(self, state, greedy=False, should_save_output=False):
        """方策に応じて行動を選択する"""
        # ランダムに行動を決定するのにはaction_valuesの値が必要ないが、学習のためにはネットワークを順伝搬させておく必要がある
        action_values = self.network.forward(state, should_save_output=should_save_output)
        if not greedy and random.random() < self.epsilon:
            # ε-greedyアルゴリズムにより、self.epsilonの確率で探索行動を取る
            # 確率的に適当に行動を選択する
            return self._explore()
        else:
            # 最適の行動を選択する(greedy)
            return self._greedy(action_values)

    def update_action_value(self, state, decided_action, reward, next_state):
        """Q-Learningアルゴリズムでネットワークを更新する
        
        元のQ-Learningの更新式は
        Q(St, At) ← (1 - η)Q(St, At) + η(Rt+1 + γ * max_a{Q(St+1, At+1))}
        または式変形して
        Q(St, At) ← η( Rt+1 + γ * max_a{Q(St+1, At+1)} - Q(St, At) )
        
        この
        Rt+1 + γ * max_a{Q(St+1, At+1)}
        が教師信号targetとなる
        """
        next_action_values = self.network.forward(next_state, should_save_output=False)
        next_max_action_value = max(next_action_values)
        target = list(self.network.output['y_output'])
        # 実際に選択した行動については報酬から教師信号を計算する
        # 　→選択しなかった行動は更新させない
        target[decided_action.value] = reward + self.gamma * next_max_action_value
        # 誤差逆伝搬でネットワークを更新する
        self.network.back_propagation(state, target)

    @staticmethod
    def _explore():
        """ランダムな行動（探索行動）をとる"""
        if random.random() < 0.5:
            return Action.ACTION1
        else:
            return Action.ACTION2

    @staticmethod
    def _greedy(action_values):
        """行動価値が最も高い行動を選択する"""
        max_action_value = float('-inf')
        max_action = None
        for action in Action:
            action_value = action_values[action.value]
            if action_value > max_action_value:
                max_action_value = action_value
                max_action = action
        return max_action


def train(agent, needs_render=False):
    steps_log = []
    for i_episode in range(10000):
        state = env.reset()
        t = 0
        for t in range(200):
            # env.render()
            # time.sleep(0.02)
            # 環境の値に応じてエージェントが行動を選択する
            action = agent.decide_action(state, should_save_output=True)
            # エージェントの行動に応じて環境の状態が変わる
            last_state = state
            state, _reward, done, info = env.step(action.value)
            # 選んだ行動と次の状態、報酬が決まったので行動価値関数を更新する(=学習)
            reward = get_reward(state)
            agent.update_action_value(last_state, action, reward, state)
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
        state = env.reset()
        t = 0
        for t in range(200):
            env.render()
            time.sleep(0.01)
            # 環境の値に応じてエージェントが行動を選択する
            action = agent.decide_action(state, greedy=True)
            # エージェントの行動に応じて環境の状態が変わる
            state, reward, done, info = env.step(action.value)
            # print(t, get_reward(state), reward)
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
