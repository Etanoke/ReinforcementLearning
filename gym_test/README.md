# OpenAI Gymの導入
## 環境
* Ubuntu 16.04 LTS
* Python 3.5.2

## 構築手順
[参考](https://gym.openai.com/docs)

1. 適当なvirtualenvを作る（ここでは名前をpy35とする）
```
$ virtualenv -p python3.5 py35
```

2. pyenvのactivate
```
$ source py35/bin/activate
```

3. 適当な場所にgymをインストール
```sh
$ git clone https://github.com/openai/gym
$ cd gym
$ pip install -e . # minimal install
```
多分、代わりに `pip install gym` でも問題ない

## 倒立振子を動かしてみる
```
$ source py35/bin/activate
$ python3 gym_test/car_pole_test.py
```
