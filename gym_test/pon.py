""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import cPickle as pickle
import gym

# hyper parameters
H = 200  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
resume = False  # resume from previous checkpoint?
render = False

# model initialization
D = 80 * 80  # input dimensionality: 80x80 grid
if resume:
    model = pickle.load(open('save.p', 'rb'))
else:
    model = {
        'W1': np.random.randn(H, D) / np.sqrt(D),
        'W2': np.random.randn(H) / np.sqrt(H)
    }

grad_buffer = {k: np.zeros_like(v) for k, v in model.iteritems()}  # update buffers that add up gradients over a batch
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.iteritems()}  # rmsprop memory


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            # r[t]が0以外の値を返すのは、勝敗がついた場合のみ。
            # したがってr[t] != 0ならばゲーム（エピソード）の境界のため、エピソード単位でのrunning_addをリセットする。
            # 例として、rは次のような配列になるはず。
            # r = [0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 1]
            running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        # ゲームの決着がついた瞬間がもっとも絶対値の大きい報酬となる。
        # ゲームの終盤から初めにかけてγずつ影響を減衰させる。
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0  # ReLU non linearity
    # 出力層の結果はUPに行く確率の対数とするので、logpという名前
    logp = np.dot(model['W2'], h)
    # シグモイドの活性化関数を通して確率を0〜1の範囲に整形する
    p = sigmoid(logp)
    return p, h  # return probability of taking action 2, and hidden state


def policy_backward(eph, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    # ephもepdlogpも配列で、それぞれ時間ステップごとの値
    # ephは隠れ層の出力値（policy_forwardで計算した値）
    # epdlogpは"教師信号"との差に割引報酬を掛けたもの
    # epdlogpが誤差逆伝搬における∂E/∂yと考えて良いのか？

    # ∂E/∂wjk = ∂E/∂h * ∂h/∂v * ∂v/∂wjk
    # ここで∂h/∂vは活性化関数の微分、活性化関数はReLUを使っているから、1 if v>0 else 0になりそうだけど、ここでは1としているのか？
    # ∂E/∂W2 = ∂E/∂h * 1 * ∂v/∂wjk
    # ここでv = Σj(wjk * hj)なので、∂v/∂wjk = hj
    # ∂E/∂hがepdlogpだとすると
    # ∂E/∂W2 = epdlogp * hj
    # これで合ってるかな・・・
    dW2 = np.dot(eph.T, epdlogp).ravel()
    # epdlogpと隠れ層-出力層間の重みの外積？
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0  # backpro prelu
    # なんでephは引数で渡しているのにepxはグローバルなのか、これがわからない
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}


env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None  # used in computing the difference frame
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
while True:
    if render:
        env.render()

    # preprocess the observation, set input to network to be difference image
    # currentのx(input)
    cur_x = prepro(observation)
    # 1フレーム前の画面と現在の画面との差分を入力とする
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
    # aprob: UPを選択する確率、h: 中間層の出力
    aprob, h = policy_forward(x)
    # action==2: UP / action==3: DOWN
    action = 2 if np.random.uniform() < aprob else 3  # roll the dice!

    # record various intermediates (needed later for backprop)
    # xのstack?もしくは複数形のつもり？エピソードの間の入力と隠れ層出力は誤差逆伝搬で使うため配列に保持する
    xs.append(x)  # observation
    hs.append(h)  # hidden state
    # ラベル、つまりUPかDOWNかのクラス分けか？
    # UP->1 / DOWN->0
    y = 1 if action == 2 else 0  # a "fake label"
    # grad that encourages the action that was taken to be taken
    # (see http://cs231n.github.io/neural-networks-2/#losses if confused)
    # 実際に選ばれた出力と、出力する確率の差
    # あとでゲーム終了時に決まる報酬（1 or -1)を割り引いた値をこれにかける。
    # これにより、勝利時は選んだ方、敗北時は選ばなかった方をより選びやすく更新するはず？
    # 例(割引は無視）: aprob = 0.7, y = 1:
    #  勝利 => (1 - 0.7) * 1 = 0.3
    #  敗北 => (1 - 0.7) * -1 = -0.3
    # これでaprobが負の方向(0 = DOWNの方）に近づく、と理解した
    dlogps.append(y - aprob)

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    # rewardのリスト。変数名は無理に省略しないでほしい
    drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

    if done:  # an episode finished
        episode_number += 1

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs, hs, dlogps, drs = [], [], [], []  # reset array memory

        # compute the discounted reward backwards through time
        # 割引した報酬
        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
        grad = policy_backward(eph, epdlogp)
        for k in model:
            grad_buffer[k] += grad[k]  # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            for k, v in model.iteritems():
                g = grad_buffer[k]  # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
        if episode_number % 100 == 0:
            pickle.dump(model, open('save.p', 'wb'))
        reward_sum = 0
        observation = env.reset()  # reset env
        prev_x = None

    if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
        print('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')
