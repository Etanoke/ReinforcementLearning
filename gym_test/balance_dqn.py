u"""OpenAI gymのCarPole-v0をQ-Learning（Neural Network版）で学習する"""
import enum

import numpy as np


def get_reward(observation):
    """環境の値から"""
    x, _, angle, _ = observation
    loss = abs(x) + abs(angle) * 5
    return -loss


class NeuralNetwork(object):
    """ニューラルネットワークの学習管理クラス"""
    INPUT_LAYER_NEURONS = 4  # 入力層ニューロン数
    HIDDEN_LAYER_NEURONS = 10  # 隠れ層ニューロン数
    OUTPUT_LAYER_NEURONS = 1  # 出力層ニューロン数

    LEARNING_RATE = 1e-4  # 学習率

    def __init__(self):
        # とりあえずバイアス項はなし
        self.params = {
            'W_INPUT': np.random.randn(self.INPUT_LAYER_NEURONS, self.HIDDEN_LAYER_NEURONS),  # 入力層 - 隠れ層間の重み
            'W_HIDDEN': np.random.randn(self.HIDDEN_LAYER_NEURONS, self.OUTPUT_LAYER_NEURONS),  # 隠れ層 - 出力層間の重み
        }

    def forward(self, x_input):
        """ネットワークを順伝搬させて出力を計算する
        
        入力層のニューロンは順に1, 2, ..., i, ...
        隠れ層と出力層も同様にj, kと添字をふることにする
        ここではuは入力値と重みの総和、φは任意の活性化関数、yはニューロンの出力とする
        φ_hは隠れ層の活性化関数。ここではReLUを使う
        φ_oは隠れ層の活性化関数。ここでは恒等関数を使う※1
        
        numpyのndarrayを使って行列演算する
        コメントの数式は1変数ずつの計算を書いているが、コードは層ごとに一括の計算であることに注意
        
        ※1: シグモイドを使おうとも思ったが、Q関数の近似という意味では値域を0〜1に制限したりしないほうが良いのか？わからん
        """
        # 隠れ層の計算
        # u_j = Σ_i {x_i * w_ij }
        u_hidden = np.dot(x_input, self.params['W_INPUT'])
        # y_j = φ_h(u_j)
        y_hidden = self.relu(u_hidden)  # 活性化関数はReLU

        # 出力層の計算
        # u_k = Σ_j {y_j * w_jk }
        u_output = np.dot(y_hidden, self.params['W_HIDDEN'])
        # y_k = φ_o(u_k)
        # y_output = self.sigmoid(u_output)  # 活性化関数はシグモイド
        y_output = u_output  # 活性化関数は恒等関数
        return y_output

    def back_propagation(self, x_input, u_hidden, y_hidden, y_output, target):
        """誤差逆伝搬でネットワークの重みを更新する
        
        誤差関数Eは、出力が連続値であるため自乗平均をとる
        targetは教師信号の値(teacher_signalのほうが良いか？）
        E = Σ_k{ (target_k - y_k)^2 } / 2
        
        隠れ層 - 出力層間の重みは次の式で更新する
        ηは学習率とする
        w_jk = w_jk - η * Δw_jk
        Δw_jk = ∂E/∂w_jk
              = ∂E/∂y_k * ∂y/∂u_k * ∂u_k/∂w_jk
              = (y_k - target_k) * φ_o'(u_k) * y_j
        ここで
        δ_output_k = (y_k - target_k) * φ_o'(u_k)
        とおいておく
        
        入力層 - 隠れ層間の重みは
        w_ij = w_ij - η * Δw_ij
        Δw_ij = ∂E/∂w_ij
              = Σ_k{ ∂E/∂y_k * ∂y/∂u_k * ∂u_k/∂y_j * ∂y_j/∂u_j * ∂u_j/∂x_i }
              = Σ_k{ (y_k - target_k) * φ_h'(u_k) * w_jk * φ'(u_j) * x_i }
              = Σ_k{ δ_output_k * w_jk * φ_h'(u_j) * x_i }
              = Σ_k{ δ_output_k * w_jk } * φ_h'(u_j) * x_i
        """
        # 隠れ層 - 出力層間の重みを更新
        # 出力層の活性化関数は恒等関数なので、φ_o'(u_k) = 1
        delta_o = y_output - target
        delta_w2 = delta_o * y_hidden
        self.params['W_HIDDEN'] += -self.LEARNING_RATE * delta_w2

        # 入力層 - 隠れ層間の重みを更新
        # φ_h'(u_j)はReLUの微分
        delta_relu = u_hidden > 0
        delta_w1 = np.dot(delta_o, self.params['W_HIDDEN']) * delta_relu * x_input
        self.params['W_INPUT'] += -self.LEARNING_RATE * delta_w1 

    @staticmethod
    def relu(inputs):
        """活性化関数ReLU"""
        inputs[inputs < 0] = 0
        return inputs

    @staticmethod
    def sigmoid(inputs):
        """活性化関数シグモイド"""
        return 1.0 / (1.0 + np.exp(-inputs))


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

    def decide_action(self, state_no, greedy=False):
        u"""方策に応じて行動を選択する"""
        if not greedy and np.random.rand() < self.epsilon:
            # ε-greedyアルゴリズムにより、self.epsilonの確率で探索行動を取る
            # 確率的に適当に行動を選択する
            return self._explore()
        else:
            # 最適の行動を選択する(greedy)
            return self._greedy(state_no)

    @staticmethod
    def _explore():
        """ランダムな行動（探索行動）をとる"""
        if np.random.rand() < 0.5:
            return Action.ACTION1
        else:
            return Action.ACTION2

    def _greedy(self, state_no):
        """行動価値が最も高い行動を選択する"""
        max_q_value = float('-inf')
        max_action = None
        for action in Action:
            q_value = self.action_value[(state_no, action)]
            if q_value > max_q_value:
                max_q_value = q_value
                max_action = action
        return max_action
