# Spatial Transformer Networks
[Spatial Transformer Networks](https://arxiv.org/abs/1506.02025)とは，画像の空間的補正(アフィン変換など)のパラメータを予測するネットワークを全体のネットワークに組み込むことで、画像補正のパラメータを学習的に獲得し，適切に予測することができる手法で，NIPS2015にて採択された．
[STNの概要説明動画](https://www.youtube.com/watch?v=Ywv0Xi2-14Y)
## 1. 構造
- 入力層 -> 40x40
- 平滑層 40x40 -> 1600
- 全結合層(act=tanh) 1600 -> 20
- ドロップアウト層 20 -> 20
- 全結合層(act=tanh) 20 -> 6 New
- STF層 40x40, 20 -> 40x40
- 畳み込み層(n_filter=16, size=3x3, strides(2,2), act=relu, padding='same') 40x40 -> 16x40x40
- 畳み込み層(n_filter=16, size=3x3, strides(2,2), act=relu, padding='same') 40x40 -> 16x40x40
- 平滑層 16x40x40 -> 25600
- 全結合層(act=relu) 25600 -> 1024
- 全結合層(act=identity) 1024 -> 10
- 出力層 10 ->

<div align="center">
<figure>
<figcaption>図1 STNを組み込んだ識別器の構造概要</figcaption>
<img src="https://github.com/hrichii/spatial_transformer_net/blob/master/images/architecture.jpg" width=100%>
</figure>
</div>

## 2. 実行環境
- Windows 10
- NVIDIA GeForce GTX 1060 3GB
- python==3.6.7
- tensorflow-gpu==1.14.0
- CUDA==10.0
- cuDNN==7.6.0

## 3. 入力
MNISTの手書き文字を使用(70,000images × 28pixel × 28pixel)
 - 訓練データ　 55,000 imgaes
 - 検証データ　 5,000 imgaes  
 - テストデータ 10,000 imgaes



## 4. 訓練
### 4.1. ゼロパディング
28x28のグレースケール画像の周囲にゼロパディングをし，40x40の画像に変換

### 4.2. データ拡張
データ拡張と言っても，ここではデータ数を増やすわけではなく，手書き数字画像を歪めてより難易度を上げ，STNの空間的補正能力を際立たせる目的で全元画像それぞれに以下の条件の範囲でランダムに変換させている．
- 回転(-30°～30°)
- 剪断歪み(-0.05~0.05)
- 上下左右移動(-0.25~0.25)
- 拡縮(0.95~1.05倍)
<div align="center">
<figure>
<figcaption>図2 データ拡張の条件毎の出力結果</figcaption>
<img src="https://github.com/hrichii/spatial_transformer_net/blob/master/images/data_augumentation.jpg" width=100%>
</figure>
</div>


### 4.3. 誤差関数
クロスエントロピー誤差関数

### 4.4. パラメータ最適化手法 Adam
- 学習率 0.0001
- 浮動小数点数b1 0.9
- 浮動小数点数b2 0.999
- 微小量 e 1e-08

### 4.5. 諸条件
 - バッチサイズ 500
 - クラス数 10
 - エポック数 20
 - イテレーション数 2200 = 20x55,000/500

## 5. 評価と結果
### 5.1. 学習の変遷
<div align="center">
<figure>
<figcaption>図3 エポック毎の損失のグラフ</figcaption>
<img src="https://github.com/hrichii/spatial_transformer_net/blob/master/images/loss_history.jpg" width=100%>
</figure>
</div>

### 5.2. テストデータを用いた評価
<div align="center">
<figure>
<figcaption>図4 テストデータを入力しSTN部分から出力された予測画像</figcaption>
<img src="https://github.com/hrichii/spatial_transformer_net/blob/master/images/prediction.jpg" width=100%>
</figure>
</div>

## 参考文献

[【論文】Spatial Transformer Networks (NIPS 2015) をできる限り省略せずに読んでいく](https://qiita.com/nkato_/items/125bd2e7c0af582aa32e)
