# MCTStest

Test Monte Carlo Tree Search with small problem

小さい問題をMonte Carlo Tree Searchで解いてみる

## Dependencies

python >= 3.6.5
scikit-learn==0.20.1

## Run

```
python mct.py
```

The mean rewards with 10 plays will be displayed as follows.

```
obs: [0, 0, 2, 0, 0, 0, 1, 0, 0, 0]
obs: [0, 0, 2, 0, 0, 1, 0, 0, 0, 0]
obs: [0, 0, 2, 0, 1, 0, 0, 0, 0, 0]
obs: [0, 0, 2, 1, 0, 0, 0, 0, 0, 0]
obs: [0, 0, 0, 0, 0, 0, 1, 2, 0, 0]
obs: [2, 0, 0, 1, 0, 0, 0, 0, 0, 0]
obs: [2, 0, 1, 0, 0, 0, 0, 0, 0, 0]
obs: [2, 1, 0, 0, 0, 0, 0, 0, 0, 0]
obs: [2, 0, 0, 0, 0, 1, 0, 0, 0, 0]
obs: [2, 0, 0, 0, 1, 0, 0, 0, 0, 0]
obs: [2, 0, 0, 1, 0, 0, 0, 0, 0, 0]
obs: [2, 0, 1, 0, 0, 0, 0, 0, 0, 0]
obs: [2, 1, 0, 0, 0, 0, 0, 0, 0, 0]
obs: [0, 0, 0, 0, 1, 0, 0, 2, 0, 0]
obs: [0, 0, 0, 0, 0, 1, 0, 2, 0, 0]
obs: [0, 0, 0, 0, 0, 0, 1, 2, 0, 0]
obs: [0, 0, 0, 0, 0, 1, 2, 0, 0, 0]
obs: [0, 2, 0, 0, 0, 0, 0, 1, 0, 0]
obs: [0, 2, 0, 0, 0, 0, 1, 0, 0, 0]
obs: [0, 2, 0, 0, 0, 1, 0, 0, 0, 0]
obs: [0, 2, 0, 0, 1, 0, 0, 0, 0, 0]
obs: [0, 2, 0, 1, 0, 0, 0, 0, 0, 0]
obs: [0, 2, 1, 0, 0, 0, 0, 0, 0, 0]
obs: [0, 0, 0, 0, 0, 0, 1, 0, 0, 2]
obs: [0, 0, 0, 0, 0, 0, 0, 1, 0, 2]
obs: [0, 0, 0, 0, 0, 0, 0, 0, 1, 2]
obs: [0, 0, 0, 1, 2, 0, 0, 0, 0, 0]
obs: [0, 2, 0, 0, 1, 0, 0, 0, 0, 0]
obs: [0, 2, 0, 1, 0, 0, 0, 0, 0, 0]
obs: [0, 2, 1, 0, 0, 0, 0, 0, 0, 0]
1.0
```

# 強化学習とは？

状態, 行動 -> 環境 -> 次の状態, 報酬
状態 -> エージェント -> 行動

+ 最適な行動をとるエージェントを学習させるための方法

+ どういう時に使えるか？
    + 上記のフレームに当てはめられる時
        + 全ての行動に対して報酬を与えられる環境を作ることができる
    + 環境が状況に応じて変化する
        + 教師データを作ることができない時でも大丈夫
    + 機械学習で解けて強化学習で解けないもの
        + 地価のデータから家の価格を予測する
            + 行動によって状態を変えられないため
    + 強化学習で解けて機械学習で解けないもの
        + 倒立振子を立てる
            + 正解を定義するのが難しい
    + 両方で解けるもの
        + ECサイトのランキング
            + ランキングアルゴリズムを変更する(行動)とユーザーのクリック率(状態)が変わる
            + クリック数の多さで正解を定義できる
+ 学習する時には、探索(explore)と知識利用(exploit)のバランスを上手くとることで、学習が収束するまでの時間を短くすることができる

# Monte Carlo Tree Searchについて

+ 使われるシーン
    + 離散: アクションが離散的である
    + 決定的: Actionの結果が決定的である
    + ゲーム
    + 完全情報: 全ての情報を知ることができる
        + 不完全情報は例えばポーカー・麻雀などのように、全てのカードや牌の情報を知ることができないもの
        + Wikipediaの情報によればポーカーなどでも使われた実績があるようである

+ 目的
    + 次の行動を決めるために用いられる
    + メリットは、探索と知識利用のバランスを考えながら次の一手を決めることができること

+ 注意
    + AlphaZeroが使っている手法と、Wikipediaに載っている手法はやや異なる気がする

# Monte Carlo Tree Searchとは？

[CheetSheet](https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0)


