# 倒立振子制御
OpenAI Gymの[CarPole環境](https://gym.openai.com/envs/CartPole-v0)での倒立振子制御の実装

## [Q-Learningによる制御](https://github.com/Etanoke/ReinforcementLearning/blob/master/balance_control/balance_q_learning.py)
Q-Learningアルゴリズムにより倒立制御を行う。

環境から取得できる情報は次の4変数：
* 横軸座標(x)
* 横軸加速度(x_dot)
* ポール角度(theta)
* ポール角速度(theta_dot)

この状態変数は連続値なので、このままでは行動価値関数のテーブルで表現することができない。
そこでこの変数を適当な閾値で区切って状態番号を振った。

ここではx, x_dot, theta, theta_dotをそれぞれ4, 4, 6, 6の区画に区切った。
状態の組み合わせは4 * 4 * 6 * 6 = 576通り。

ざっくりいうと576通りの状態に対してどの行動を取ると良いか、というのを学習させた。

### 課題
* 状態数をどのくらいの粒度に区切るのか  
    - 状態数が少なすぎるとエージェントが行動決定するための情報が粗くなり、行動の精度が落ちる 
    - 状態数が多すぎると探索に時間がかかる。場合によっては未探索（あるいは探索不足）の状態が増え、最適な行動が取れない  
* 状態を区切る閾値をどう決めるのか。  
    今回は適当に決めたが、この決め方によって性能が大きく変わりそう

### 実行結果
表示しているエピソード分の学習を行ってからgreedy動作をさせた。
![](https://raw.githubusercontent.com/Etanoke/ReinforcementLearning/master/balance_control/images/balance_control_ql.gif)


## [Q-Networkによる制御](https://github.com/Etanoke/ReinforcementLearning/blob/master/balance_control/balance_q_network.py)
上記のQ-Learningでは「状態数をいくつに区切るか」「状態を区切る閾値をどう設定するか」を手動で調整する必要があった。
Q-Networkでは行動価値関数をニューラルネットで近似しすることでテーブル関数を使用しなくても良いようにしてこの問題を解決した。
性能もQ-Learningより改善された。

### 実行結果
表示しているエピソード分の学習を行ってからgreedy動作をさせた。
![](https://raw.githubusercontent.com/Etanoke/ReinforcementLearning/master/balance_control/images/balance_control_qn.gif)
