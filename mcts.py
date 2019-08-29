import random
import math
from collections import Counter, deque
from copy import deepcopy
from sklearn.neural_network import MLPRegressor
import numpy as np
np.set_printoptions(precision=2)

class TreeNode:
    '''
    MonteCarloTreeのノード
    W, N, Qを持つ。
    Qは W/Nで計算できるので省略
    '''
    def __init__(self, parent):
        self.parent = parent
        self.children = {}

        self.W = 0
        self.N = 0

    def expand(self, feasible_actions):
        '''
        Actionの数だけ直下に子を作る
        '''
        for a in feasible_actions:
            self.children[a] = TreeNode(self)
            
    def has_children(self):
        '''子ノードがあるか？'''
        return len(self.children) > 0

    def ensure_children(self, feasible_actions): 
        '''子ノードがなければ作る'''
        if not self.has_children():
            self.expand(feasible_actions)

    def backup(self, reward):
        '''現在のノードから根まで報酬と通過回数を伝搬させる'''
        self.N += 1
        self.W += reward
        if self.parent:
            self.parent.backup(reward)

    def is_leaf(self):
        '''このノードが末端の葉か？'''
        return not self.has_children()


class MonteCarloTree:
    '''
    MonteCarloTreeを扱うクラス
    '''
    def __init__(self, policy_fn):
        self._root = TreeNode(None)

        self.c = 5
        self.policy_fn = policy_fn

    def move(self, action):
        '''
        ツリーの中を移動する
        移動した側の部分木をルートに設定し、それ以外は捨てる
        (メモリ量の節約のためだと思われる。MCTSはゲーム開始時に初期化され、ゲーム中は更新され続けるが、不要な部分は捨てていく)
        '''
        next_root = self._root.children[action]
        self._root = next_root

    def select_deterministically(self):
        '''
        現在のMCTから導かれる行動を決定的に導く
        '''
        action_node = max(self._root.children.items(),
            key=lambda action_node: action_node[1].N
        )

        return action_node[0]

    def select_stochastically(self, temperature):
        '''
        現在のMCTから導かれる行動を確率的に導く
        temperatureは温度を表す
        温度は探索と知識利用のバランスを調節する値である
        温度が高いほど探索側に傾く
        '''
        action_nodes = list(self._root.children.items())
        actions, nodes = zip(*action_nodes)
        weights = [n.N ** (1/temperature) for n in nodes] 
        if sum(weights) == 0:
            raise Exception('call playout before select')
        return random.choices(actions, weights=weights)[0]

    def _playout(self, node, env, obs, verbose=False):
        '''
        MCTを一つ育てる        
        '''
        done = False
        reward = 0
        value = 0
        while not node.is_leaf():
            probs, value = self.policy_fn(obs)
            action_ucts = MonteCarloTree.get_ucts(node, probs, self.c)
            action = max(action_ucts, key=lambda au: au[1])[0]
            if verbose:
                print('obs: ', end='')
                env.render()
                print('children:', node.children)
                print('probs:', probs)
                print('action_ucts:', action_ucts)
                print('action:', ['left', 'right'][action])

            node = node.children[action]
            obs, reward, done, info = env.step(action)

        if not done:
            feasible_actions = env.feasible_actions()
            node.ensure_children(feasible_actions)
            node.backup(value)
        else:
            node.backup(reward)

        return node, env, obs

    def playout(self, orig_env, num_playout=200, verbose=False):
        '''
        現在の状態から
        num_playoutの数だけシミュレートを行いMCTを育てる
        '''
        node = self._root
        env, obs = orig_env.copy()
        for i in range(num_playout):
            if verbose:
                print('playout:', i)
            self._playout(node, env, obs, verbose)
            env, obs = orig_env.copy()
            node = self._root

    @staticmethod
    def get_ucts(node, probs, c_u):
        '''
        UCT(Upper Confidence Tree)
        の手法で、子ノードに対して値を計算する
        https://ja.wikipedia.org/wiki/%E3%83%A2%E3%83%B3%E3%83%86%E3%82%AB%E3%83%AB%E3%83%AD%E6%9C%A8%E6%8E%A2%E7%B4%A2#UCT_(Upper_Confidence_Tree)
        '''
        actions, child_nodes = zip(*node.children.items())

        def calc_uct(N_p, W, N, P):
            q = W / N if N != 0 else 0
            u = c_u * P * math.sqrt(math.log(N_p + 1) / (1+N))
            return q + u

        ucts = [calc_uct(node.N, child.W, child.N, p) for child, p in zip(child_nodes, probs)]
        return list(zip(actions, ucts))

    def get_probability(self):
        '''
        現在の状態から可能な行動に対して
        子ノードのNを使ってsearch probabilityを求める
        '''
        children = self._root.children
        total = sum([n.N for n in children.values()])
        result = [0] * 2
        for a, n in children.items():
            result[a] = n.N / total
        return result

    def get_value(self):
        '''Qを求める'''
        return self._root.W / self._root.N

    def count(self):
        '''MCTが持つノード数を求める'''
        def count_recursive(node):
            return sum(count_recursive(child) for child in node.children.values()) + 1
        return count_recursive(self._root)

    def __str__(self):
        '''MCTを描画する'''
        def print_recursive(action, node, depth):
            if node.N == 0:
                return ''
            result_str = ''
            if depth > 0:
                result_str += '|  ' * (depth-1) + '|--'
            result_str += f'[a:{action}, N:{node.N}, W:{node.W}]\n'
            for child_action, child_node in node.children.items():
                result_str += print_recursive(child_action, child_node, depth+1)
            return result_str 
        return print_recursive(None, self._root, 0)


class Action:
    left = 0
    right = 1


class TestEnv:
    '''
    MCTをテストするために、簡単な探索問題を作る
    [0 2 0 0 0 0 0 0 1 0]
    のような状態から1を左右に動かして2に近づけていく問題
    2のところにくれば'1'の報酬を与える
    一度の試行中に同じ状態に二回以上来た場合は失敗とし報酬は'0'とする
    初期状態の2,1の場所はランダムに決まる
    '''
    player = 1
    goal = 2
    n = 10
    n_actions = 2

    def __init__(self):
        self.reset()

    def reset(self):
        n = self.n
        field = [0] * n

        self.player_pos = random.randint(0, n-1)
        self.goal_pos = random.randint(0, n-1)
        while self.player_pos == self.goal_pos:
            self.goal_pos = random.randint(0, n-1)
        field[self.player_pos] = self.player
        field[self.goal_pos] = self.goal

        self.obs = field
        self.obs_counter = Counter()
        return field

    def check_obs(self):
        key = hash(str(self.obs))
        self.obs_counter[key] += 1
        return self.obs_counter[key] > 1

    def step(self, action):

        obs = self.obs[:]
        player_pos = self.player_pos
        goal_pos = self.goal_pos

        if action == Action.right:
            next_player_pos = player_pos+1
        if action == Action.left:
            next_player_pos = player_pos-1

        obs[next_player_pos] = self.player
        obs[player_pos] = 0
        self.player_pos = next_player_pos
        self.obs = obs

        if next_player_pos == goal_pos:
            return obs, 1, True, {}

        return obs, 0, self.check_obs(), {}

    def render(self):
        print(self.obs)

    def feasible_actions(self):
        '''
        実行可能な行動のリストを返す
        '''
        if self.player_pos == 0:
            return [Action.right]
        if self.player_pos == self.n-1:
            return [Action.left]
        return [Action.left, Action.right]

    def copy(self):
        '''MCTSでシミュレーションするために環境をコピーする'''
        new_env = TestEnv()
        new_env.obs = deepcopy(self.obs)
        new_env.player_pos = deepcopy(self.player_pos)
        new_env.goal_pos = deepcopy(self.goal_pos)
        new_env.obs_counter = deepcopy(self.obs_counter)
        
        return new_env, new_env.obs


class PolicyNet:
    '''
    状態を行動価値と状態価値に変換する関数
    単純な3層NeuralNetworkで構築する
    10 -> 100 -> 3
    '''
    def __init__(self, env):
        self.reg = MLPRegressor((env.n, 100, env.n_actions+1), max_iter=1000, activation='relu')
        self.reg.fit(
            np.random.normal(size=(10, env.n)),
            np.random.rand(10, env.n_actions+1)
        )
        self.data = deque(maxlen=1000)

    def get_policy_fn(self):
        '''
        状態を行動価値と状態価値に変換する関数を返す
        '''
        def policy_fn(obs):
            X = np.array([obs])
            result = list(self.reg.predict(X)[0])
            probs = result[:-1]
            value = result[-1]
            probs = (probs - np.min(probs) + 1) / np.sum(probs - np.min(probs) + 1)
            return probs, value
        return policy_fn

    def update_batch(self, obs_list, prob_list):
        '''
        状態のリストと(行動価値と状態価値)の一覧を使って
        NeuralNetworkを学習する
        '''
        for obs, prob in zip(obs_list, prob_list):
            self.data.append((obs, prob))
        
        if len(self.data) < 5:
            return
        sampled = random.sample(self.data, min(500, len(self.data)))
        obs_list, prob_list = zip(*sampled)
        X = np.array(obs_list)
        y = np.array(prob_list)
        print(np.concatenate([X, y], axis=1)[:20,:])
        print('before, score:', self.reg.score(X, y))
        self.reg.fit(X, y)
        print('after, score:', self.reg.score(X, y))

if __name__ == '__main__':

    env = TestEnv()
    policy_net = PolicyNet(env)

    mct = MonteCarloTree(policy_net.get_policy_fn())

    obs = env.reset()
    try:
        num_done = 0
        obs_list = []
        prob_list = []
        while True:
            mct.playout(env, num_playout=10, verbose=False)
            prob = mct.get_probability()
            value = mct.get_value()

            action = mct.select_stochastically(2.0)

            mct.move(action)

            obs_list.append(obs)
            prob_list.append(prob)

            obs, reward, done, info = env.step(action)
            if done:
                prob_list = [np.array(prob + [reward]) for prob in prob_list]
                policy_net.update_batch(obs_list, prob_list)
                obs_list = []
                prob_list = []
                num_done += 1
                obs = env.reset()
                mct = MonteCarloTree(policy_net.get_policy_fn())

            if num_done > 100:
                break
    except KeyboardInterrupt:
        pass 

    total_reward = 0
    num_iterate = 10
    for _ in range(num_iterate):
        obs = env.reset()
        done = False
        mct = MonteCarloTree(policy_net.get_policy_fn())
        while not done:
            mct.playout(env, verbose=False)
            prob = mct.get_probability()
            print('obs:', obs)
            action = mct.select_deterministically()
            mct.move(action)
            obs, reward, done, info = env.step(action)
            total_reward += reward
    print(total_reward / 10)
