論文：https://arxiv.org/abs/1711.11279

# Testing with Concept Activation Vectors

## Summary

- ニューラルネットワークモデルの判断根拠を示す手法
- 従来のピクセルごとに重要度を算出するような手法ではなく、予測クラスの概念（色、性別、人種など）の重要度を示す
  - 各データ点に対する説明（≒ローカル）ではなく、各クラスに対する説明（≒グローバル）を生成するため、人間にわかりやすい説明性を持つ。MLモデルの専門知識がなくても説明を理解することができる
- 解釈したい既存モデルに対して、再学習や変更の必要はない
- 入力空間上にデータ点が置かれたときに、人が解釈しやすいような概念空間にデータ点を持ってくるイメージ
- （図1が理解できればokな気がする）

---

## abstract

- deep learningモデルの解釈は、サイズや複雑さ、不透明な内部状態などによって課題である
- 画像分類のような多くのシステムでは、高レベルの概念ではなく、低レベルの概念で動作する
- これらの課題に対処するため、Concept Activation Vectors(CAVs)を導入する
- ユーザーが定義した概念が分類結果に対してどれくらい重要かを定量化するために、方向微分を使用する

## 1.Introduction

- 問題提起
    - ニューラルネットワークなどの最新の機械学習モデルの動作を理解することは重要であり、大きな課題。正確な予測を確保しながらも、解釈可能性は重要
    - 既存のアプローチ：解釈可能性に対する自然なアプローチは、特徴量の観点から表すこと
        - 例①：ロジスティック回帰分類では係数の重みを各特徴の重要度として解釈する
        - 例②：顕著性マップは、1次導関数に基づいたピクセル単位の重要度の重みをもって解釈する
    - 主要な難点：ほとんどのMLモデルがピクセル単位で解釈が必要になること
        - 人間が容易に理解できる高レベルの概念に対応していない
        - モデルの内部が理解できていないように見える場合がある
- どうやって解決するのか
  - MLモデルの状態を、入力データに対応する基底ベクトル <img src="https://latex.codecogs.com/gif.latex?em" /> にまたがるベクトル空間 <img src="https://latex.codecogs.com/gif.latex?Em" />として表現し、人間が解釈可能な概念に対応する暗黙のベクトル <img src="https://latex.codecogs.com/gif.latex?eh" /> にまたがるベクトル空間 <img src="https://latex.codecogs.com/gif.latex?Eh" /> を考える
  - この観点から、MLモデルの「解釈」は関数  <img src="https://latex.codecogs.com/gif.latex?g:&space;Em\rightarrow&space;Eh" /> として見ることができる（入力空間上にデータ点が置かれたときに、人が解釈しやすいように概念空間にデータ点を持ってくる関数 <img src="https://latex.codecogs.com/gif.latex?g" /> ）
    - <img src="https://latex.codecogs.com/gif.latex?g" /> が線形の場合、**線形解釈可能性**と呼ぶ
    - 入力空間 <img src="https://latex.codecogs.com/gif.latex?Em" /> のいくつかを説明できず、<img src="https://latex.codecogs.com/gif.latex?Eh" /> で考えられるすべての概念をカバーできないこともある
- Concept Activation Vectors (CAV)の概念
  - 概念画像とランダムな反例の間で線形分類器をトレーニングし、決定境界に直交するベクトルを取得することにより、CAVを導出する（図1見た方が早い）
- 主な結果
  - たとえば、シマウマを認識するML画像モデルと、「縞模様」を定義する新しいユーザー定義のサンプルセット（概念画像）がある場合、TCAVは「シマウマ」予測に対する縞模様の概念の影響を単一の数値として定量化できる
  - さらに、モデルの出力クラスと有意な相関関係が示されるか統計テストを実施（3.2章で詳述）
  
- ![image](https://user-images.githubusercontent.com/36921282/71402601-ada9cb00-2670-11ea-8ecd-92d97d0aff14.png)
- 図1：
    a : 上はユーザー定義の概念画像・下はランダムな反例
    b : 学習済みクラスの学習データセット
    c : 学習済みモデル
    d：概念画像とランダムな反例の線形分類器（<img src="https://latex.codecogs.com/gif.latex?v_C^l" /> は分類境界に直行する法線ベクトル）
    e：概念に対する感度（TCAVは方向微分 <img src="https://latex.codecogs.com/gif.latex?S_{c,&space;k,&space;l}(x)" /> を使用する）
- 目標
  - アクセシビリティ：ユーザーのMLの専門知識がほとんど必要ないこと
  - カスタマイズ：あらゆる概念（性別など）に適応し、トレーニング中の概念に限定されないこと
  - プラグインの準備：MLモデルの再トレーニングや変更がなく機能すること
  - グローバルな定量化：個々のデータ入力を説明するだけでなく、クラス全体またはデータセットを単一の定量的尺度で解釈できること



## 2.Related Work
この章では、1. 既存の解釈方法、2. ニューラルネットワークに固有の方法、3. ニューラルネットワークのローカル線形性を活用する方法の概要を示す

### 2.1. Interpretability methods

解釈可能性を実現する方法2つ

1. 本質的に解釈可能なモデルに限定する
   - メリット：説明がモデルに組み込まれているためシンプルである
   - デメリット：既に高性能なモデルを持っているユーザーにとってはコストがかかる可能性がある
   - ⇒ 説明可能なMLに対する需要の増加に伴い、ネットワークを再トレーニングまたは変更することなく適用できる方法の必要性が高まっている
2. 洞察をもたらす方法でモデルを後処理する（**摂動ベース/感度分析ベース**　※LIME・SHAPなど）
   - 構造によるが、ローカル（≒データ点とその近傍の説明）な解釈になる
     - 説明が特定のデータ点とその近傍のみに当てはまる場合（≒ローカルな説明）、クラス内のすべての入力に対してではなくなり、矛盾する可能性がある
     - たとえば、同じクラスの2つのデータ点に対して矛盾する説明がされ、ユーザーの信頼が低下する場合がある
   - TCAVは、解釈するために人間が関係する概念に向かってデータ点を摂動するので、一種のグローバル摂動法
     - TCAVは、単一のデータポイントだけでなく、各クラスに当てはまる説明（≒グローバルな説明）を生成する

### 2.2. Interpretability methods in neural networks

TCAVの目標は、ニューラルネットワークモデルのような高次元の <img src="https://latex.codecogs.com/gif.latex?Em" /> を解釈すること
顕著性マップは、画像分類の最も一般的な手法である

- 図8に示すように、特定の画像の各ピクセルがその分類にとってどれほど重要かを示すマップを生成する
- 図8 ![image](https://user-images.githubusercontent.com/36921282/71402630-c4502200-2670-11ea-89d9-3bc3a1345a6d.png)
- 顕著性マップは、いくつかの制限がある
  - (1) 顕著性マップは1つの画像（≒ローカルな説明）に対するものであるため、人間はクラス全体の結論を引き出すために、各画像を見なければならない
    - たとえば、2つの異なる猫の写真の2つの顕著性マップを考えてみる
      - 1つの画像の猫に対しては耳が重要であるという結果が得られたとする。一方、別の画像の猫の耳は重要でないという結果だった場合、「猫」の予測において耳がどれほど重要か評価できないでしょ？
  - (2) マップが関心を寄せる概念をユーザーが制御できない（カスタマイズの欠如）

### 2.3. Linearity in neural network and latent dimensions

- ニューロンの線形結合が有意義な情報をエンコードする可能性があることを示す多くの研究がある。単純な線形分類器を介して意味のある方向を効率的に学習できることを示している
- 潜在的な次元を人間の概念にマッピングすることもGANなどの文脈で研究され、属性固有の画像が生成される
- 生成モデルの文脈で、潜在的な次元で概念ベクトルを使用するアイデアも検討されている
- このアイデアを拡張し、学習した方向に沿って方向導関数を計算し、モデルの予測における各方向の重要性を収集
- TCAVのフレームワークを使用して、ユーザーにとって意味のある概念をテストし、各クラスのグローバルな説明を作成できる

## 3. Methods

この章では、アイデアと方法について説明する

（a）方向導関数を使って、さまざまなユーザー定義の概念に対するMLモデル予測の感度を定量化する方法

（b）相対的な重要性の最終定量的説明（  <img src="https://latex.codecogs.com/gif.latex?TCAV_Q" /> ）を計算する方法

- モデルの再学習や変更を行わずに、予測クラスに各概念を適用する
- 入力 <img src="https://latex.codecogs.com/gif.latex?x\in&space;R^n" /> と  <img src="https://latex.codecogs.com/gif.latex?m" /> 個のニューロンを持つフィードフォワード層  <img src="https://latex.codecogs.com/gif.latex?l" /> を持つニューラルネットワークモデルを検討する

### 3.1. User-defined Concepts as Sets of Examples

この方法の最初のステップは、関心のある概念を定義すること
- このとき、解釈したいモデルのラベルや学習データだけを使った概念に制限しないこと（この手法の主な利点だから）

- MLの専門家ではない人でも概念を定義、探索、改良できる柔軟性がある

### 3.2. Concept Activation Vectors (CAVs)

線形解釈可能性のアプローチに従う
1. レイヤー <img src="https://latex.codecogs.com/gif.latex?l" /> の出力を抽出する
2.  <img src="https://latex.codecogs.com/gif.latex?l" /> の出力から、概念画像群と説明したい画像群を分類する線形分類器を学習させ、分類境界に対する法線ベクトル <img src="https://latex.codecogs.com/gif.latex?v_C^l" /> を得る（＝「Concept Activation Vectors」、（図1の赤い矢印））

### 3.3. Directional Derivatives and Conceptual Sensitivity

顕著性マップなどの方法は、ピクセルなどの個々の入力特徴量に対するロジットの勾配を使用して計算する

<img src="https://latex.codecogs.com/gif.latex?h_k(x)" /> はクラス <img src="https://latex.codecogs.com/gif.latex?k" /> のデータ点 <img src="https://latex.codecogs.com/gif.latex?x" /> のロジット、<img src="https://latex.codecogs.com/gif.latex?x_{a,b}" /> は <img src="https://latex.codecogs.com/gif.latex?x" /> の座標 <img src="https://latex.codecogs.com/gif.latex?(a,b)" />

![image](https://user-images.githubusercontent.com/36921282/71403419-0b3f1700-2673-11ea-94b5-e6c9ca442477.png)

顕著性マップでは導関数を使って、ピクセル <img src="https://latex.codecogs.com/gif.latex?(a,b)" /> の大きさの変化に対する出力クラス <img src="https://latex.codecogs.com/gif.latex?k" /> の感度を測定している

本手法では、CAVと方向導関数を使用することにより、活性化レイヤー <img src="https://latex.codecogs.com/gif.latex?l" /> で概念方向に向かう入力の変化に対するML予測の感度を測定する（式(1) 参照）

レイヤー <img src="https://latex.codecogs.com/gif.latex?l" /> での入力 <img src="https://latex.codecogs.com/gif.latex?x" /> における、コンセプト <img src="https://latex.codecogs.com/gif.latex?C" /> に対するクラス <img src="https://latex.codecogs.com/gif.latex?k" /> の「概念感度」は方向として計算できる。
導関数 <img src="https://latex.codecogs.com/gif.latex?S_{C, k, l}(x)" /> は下記のように定義する。

![image](https://user-images.githubusercontent.com/36921282/71403494-3fb2d300-2673-11ea-9422-768f8337d5c5.png)


この <img src="https://latex.codecogs.com/gif.latex?S_{C, k, l}(x)" /> は、モデルの任意の層における概念に対するモデル予測の感度を定量的に測定できる

ピクセルごとの顕著性マップとは異なり、入力全体または入力セットで計算された概念ごとのスカラー量になる。



### 3.4. Testing with CAVs (TCAV)

CAVを使ったテストでは、方向導関数を使用して、入力のクラス全体でMLモデルの概念感度を計算する

<img src="https://latex.codecogs.com/gif.latex?k" /> ：教師あり学習のクラスラベル
<img src="https://latex.codecogs.com/gif.latex?X_k" /> ：クラス <img src="https://latex.codecogs.com/gif.latex?k" /> のデータセット
TCAVスコアを次のように定義

![image](https://user-images.githubusercontent.com/36921282/71403532-5e18ce80-2673-11ea-85f8-577fef7a990a.png)


- <img src="https://latex.codecogs.com/gif.latex?l" /> 層の活性化ベクトルが概念 <img src="https://latex.codecogs.com/gif.latex?C" /> によって正の影響を受けた入力の割合、https://latex.codecogs.com/gif.latex?TCAV_{Q_{C,k,l}}\in&space;[0,&space;1] 
- <img src="https://latex.codecogs.com/gif.latex?TCAV_{Q_{C,k,l}}" /> は <img src="https://latex.codecogs.com/gif.latex?S_{C, k, l}" /> の符号のみに依存する（負の影響は無視する）
- <img src="https://latex.codecogs.com/gif.latex?TCAV_Q" /> を使用すると、ラベル内のすべての入力について、概念的な感度をグローバルに簡単に解釈できる



### 3.5. Statistical significance testing

割愛（簡単な統計的有意性テストをやりました）

### 3.6. TCAV extensions: Relative TCAV

- 意味的に近い概念（たとえば、茶髪と黒髪）は、多くの場合直交とは程遠いCAVを生成する
- 関連する概念間の相対的な比較は良い解釈ツールで、細かい区別をするために有益に使用される可能性がある
- ここで、アナリストは2つの異なる概念 <img src="https://latex.codecogs.com/gif.latex?C" /> と <img src="https://latex.codecogs.com/gif.latex?D" /> を表す2つの入力セットを選択する
  - <img src="https://latex.codecogs.com/gif.latex?f_l(P_C)" /> と <img src="https://latex.codecogs.com/gif.latex?f_l(P_D)" /> で分類器をトレーニングすると、ベクトル https://latex.codecogs.com/gif.latex?v_{C,&space;D}^l\in&space;R^m が得られる
  - <img src="https://latex.codecogs.com/gif.latex?x" />が概念 <img src="https://latex.codecogs.com/gif.latex?C" /> または <img src="https://latex.codecogs.com/gif.latex?D" /> により関連しているかどうかを測定できる
- たとえば、相対CAVは画像認識に適用される場合があり、「点線」、「縞模様」、および「メッシュ」テクスチャの概念は内部表現として存在し、相関または重複していると仮定できる
  - 3つの正の例セット <img src="https://latex.codecogs.com/gif.latex?C" /> または <img src="https://latex.codecogs.com/gif.latex?P_{dot}" /> 、<img src="https://latex.codecogs.com/gif.latex?P_{stripe}" />、および <img src="https://latex.codecogs.com/gif.latex?P_{mesh}" /> が与えられた場合、それぞれに対して、補数による負の入力セットを構築することにより、相対CAVを導き出すことができる（たとえば、ストライプのhttps://latex.codecogs.com/gif.latex?\{P_{dot}\cup&space;P_{mesh}\}）

## 4. Results

4.1: CAVが人間が解釈可能な概念を学んでいるかの検証

​	4.1.1：さまざまな概念との類似性に基づいて画像をソート

​	4.1.2：活性化最大化手法であるディープドリームを使用

4.2：GoogleNetとInception V3に対して、TCAVを使用。①TCAVの有用性、②バイアス、③概念が学習される場所を示す

4.3：真のラベルのキャプションを入れたデータセットを作成し、 モデルがどこを重要視しているかを検証（キャプションを見ているか、画像を見ているか）。TCAVは真のラベルを厳密に追跡し、顕著性マップはこの真の値を人間に伝えることができないことを示した

4.4：応用してみた。TCAVを適用して、糖尿病性網膜症（DR）を予測するモデルの解釈を支援。TCAVは、ドメイン専門家の知識と分岐したモデルについて洞察を提供した

### 4.1. Validating the learned CAVs

最初のステップは、学習したCAVが目的の概念と整合していることを確認すること

1. CAVを使って画像を並べ替える
2. 視覚的に確認するためのアクティベーション最大化手法を使って、各CAVを最大限アクティブにするパターンを学習する

#### 4.1.1. SORTING IMAGES WITH CAVs

 <img src="https://latex.codecogs.com/gif.latex?v_C^l" />セット間のコサイン類似度を計算した（ソートされる画像は、CAVの学習には使用されていない）

- CAVを使用して、概念との関係に関して画像を並べ替えることができる
- データセットのバイアスも明らかにできる

![image](https://user-images.githubusercontent.com/36921282/71402707-f06ba300-2670-11ea-9ba0-e0b8e894a177.png)

図2：
（左）「CEO」概念を使って、ストライプの最もよく似た写真とそうでない写真
（右）「女性モデル」概念を使用って、ネクタイの最もよく似た写真とそうでない写真

- 左上の3つの画像はピンストライプで、CEOが着用するネクタイやスーツに関連している可能性がある
- 右上の3つの画像はすべてネクタイの女性を示している
  - CAVが類似性ソーターとして機能し、学習画像のバイアスを明らかにするために使えることを示唆している

#### 4.1.2. EMPIRICAL DEEP DREAM

任意の概念画像に対するCAVが最大になるパターンを最適化し、CAVが関心のある根本的な概念を反映していることを示す。

- Deep DreamやLucidなどのアクティベーション最大化手法で行われるように、最適化の開始はランダム画像

![image](https://user-images.githubusercontent.com/36921282/71402749-04afa000-2671-11ea-9041-482aee5afcee.png)

図3.ニットテクスチャ、コーギー、およびシベリアンハスキーの概念を使用したディープドリーム（拡大）

- TCAVを使用して、レイヤー内の興味深い方向を識別および視覚化できることを示唆している（？）
- AppendixにすべてのレイヤーとCAVの結果がある



### 4.2. Insights and biases: TCAV for widely used image classifications networks

GoogleNetとInception V3にTCAVを適用して、下記を確認

1）TCAVの有用性

2）バイアス

3）概念が学習される場所



#### 4.2.1. GAINING INSIGHTS USING TCAV

色、テクスチャ、オブジェクト、性別、人種など、さまざまな種類のCAVを試行
（これらの概念はいずれもクラスラベルのセットには含まれていない）

GoogleNetは全レイヤー、Inception V3は最後の3層の出力から学習したCAVでTCAVの結果を示す

![image](https://user-images.githubusercontent.com/36921282/71402774-12652580-2671-11ea-807e-96b7b8334139.png)

図4：グラフの色が各レイヤーのTCAV

- 直観的に納得できる結果が出た
  - 消防車クラスの赤の概念の重要性
  - シマウマクラスの縞模様の概念
  - ダンベルクラスの腕の概念
  - 犬ぞりクラスのシベリアンハスキーの概念
- 加えて、バイアスの確認もできる（このネットワークが性別と人種に敏感であるという疑念）
  - 「ラグビーボール」クラスにおいて、「白人」の概念が関連している
  - 「ピンポンボール」クラスにおいて、「東アジア人」の概念が関連している
  - 「エプロン」クラスにおいて、「女性」の概念が関連している
- 人種の概念は最終層に近づくにつれて強い信号を示すが、テクスチャ（ストライプなど）の概念は最初のレイヤーで <img src="https://latex.codecogs.com/gif.latex?TCAV_Q" /> に影響する

- 場合によっては、CAVを学習に使う画像は少数で十分だった
  - 「ダンベル」クラスでは概念画像を30枚収集した。少数にもかかわらず、「腕」の概念が他の概念よりも重要であると正常に識別した。この発見は、DeepDreamの結果と一致していて、TCAVは定性的な発見に対する定量的な確認を可能にする



#### 4.2.2. TCAV FOR WHERE CONCEPTS ARE LEARNED

線形分類器を学習して各概念を分離するのがCAVの学習プロセス
この線形分類器のパフォーマンスを使って、各概念が学習されるレイヤーの下限を取得できる

![image](https://user-images.githubusercontent.com/36921282/71402805-26108c00-2671-11ea-8c6e-ba22e10c1652.png)

図5：各レイヤーでのCAVの精度。グラフの横にいくほどレイヤーが深くなる。グラフの色は概念
単純な概念（色など）は、下位層で高いパフォーマンスで、抽象的・複雑な概念（人、オブジェクトなど）は上位層で高いパフォーマンスを実現する

- 各層でどの程度粗い・細かい部分の特徴を捉えているかがわかる

### 4.3. A controlled experiment with ground truth

#### 実験概要

- この実験の目標は、TCAVの定量的結果と顕著性マップの評価を比較すること
- 画像にノイズの多いキャプションが書き込まれた3つの任意のクラス（ゼブラ、タクシー、キュウリ）のデータセットを作成した（図6参照）
  - ノイズパラメータ https://latex.codecogs.com/gif.latex?p\in&space;[0,&space;1.0] は、画像キャプションが画像クラスと一致する確率を制御する。 ノイズがない場合（ <img src="https://latex.codecogs.com/gif.latex?p = 0" /> ）、キャプションは常に画像ラベルと一致する。 <img src="https://latex.codecogs.com/gif.latex?p = p = .3" /> では、各写真の30％の確率で、正しいキャプションがランダムな単語に置き換えられる
- 4つのネットワークを学習する：p = 0, p = 0.3, p = 1.0, キャプションなし
  - 分類タスクでは、画像かキャプションまたはその両方を学習するケースがある。各ネットワークがどのコンセプトに注意を払ったかを概算するために、キャプションなしの画像でネットワークのパフォーマンスをテストした
    - 画像を注視している場合、キャプションがなくてもパフォーマンスは高いまま。そうでない場合パフォーマンスは低下する



![image](https://user-images.githubusercontent.com/36921282/71402824-37599880-2671-11ea-84c6-99a4d74576f9.png)



図6：通常の学習画像と、タクシーとキュウリクラスのキャプション付きの画像



#### 4.3.1. QUANTITATIVE EVALUATION OF TCAV

![image](https://user-images.githubusercontent.com/36921282/71402841-44768780-2671-11ea-81a4-4052c6f8bda2.png)

図7

- タクシークラス（図7左）
  - どのモデルも高い精度で、タクシーを判断する際にキャプションを使っていないことがわかる
- きゅうりクラス（図7右）
  - キャプションのノイズが少ないモデル（0%, 30%）では、ほとんどキャプションを見ている（赤線）。そのためAccuracyが低い
  - ノイズが多いorキャプションがないモデル（100%, no captions）では、キャプションではなく画像を見ている。そのためAccuracyが高い
  - <img src="https://latex.codecogs.com/gif.latex?TCAV_Q" /> とAccuracyが密接に関連している



#### 4.3.2. EVALUATION OF SALIENCY MAPS WITH HUMAN SUBJECTS

顕著性マップは、画像ベースのネットワークの解釈可能性方法として使用される（2章参照）
人間の被験者実験を通して、顕著性マップが人間に伝えられる情報を定量的に評価する

![image](https://user-images.githubusercontent.com/36921282/71402855-4fc9b300-2671-11ea-94fd-0c065bc28abd.png)

図8：
異なるノイズパラメーター <img src="https://latex.codecogs.com/gif.latex?p" /> （行）および異なる顕著性マップメソッド（列）
ネットワークがキャプションよりも画像に多くの注意を払っていることは顕著性マップから明らかではない

生成された顕著性マップを使用して50人の対して実験を行った。（簡単にするため、4つのノイズレベルのうち2つ（0％と100％のノイズ）、および2つのタイプの顕著性マップを評価した）

各ワーカーは、モデルが重要視しているのが「画像」か「キャプション」かポイントをつけた。

![image](https://user-images.githubusercontent.com/36921282/71402868-5a844800-2671-11ea-8bf8-3daeecb1bee1.png)

図9：実験結果

- タクシークラスでは、キャプションよりも画像の方が重要であるという事実があった
- ただし、顕著性マップを見ると、人間はキャプションの概念をより重要であると認識しているか（ノイズ0％のモデル）、違いを認識していない（ノイズ100％のモデル）
- 対照的に、TCAVの結果は、画像の概念がより重要であることを正しく示している
- 全体的に、非常に自信があると評価された正解の割合は、不正解の割合と類似しており、顕著性マップが誤解を招く可能性があることを示している



### 4.4. TCAV for a medical application

TCAVは、眼底画像から治療可能であるが視力を脅かす状態である糖尿病性網膜症（DR）を予測する現実の問題に適用している。 結果について医療専門家と相談した。

詳細は割愛するが、TCAVは「専門家がモデルの予測に同意できない場合、エラーの解釈と修正に役立つ場合がある」ということを言っている

![image](https://user-images.githubusercontent.com/36921282/71402883-66700a00-2671-11ea-924b-f535704163a5.png)

図10：

上：DRレベル4の画像とTCAVの結果。 <img src="https://latex.codecogs.com/gif.latex?TCAV_Q" /> は、このレベルに関連する機能では高く（緑色）、無関係な概念では低い（赤色）

中：DRレベル1（軽度）TCAV結果

下：HMA機能は、DRレベル1よりもDRレベル2でより頻繁に表示される



- モデルは、DRレベルを予測する（レベル0（DRなし）～レベル4（増殖性））
- 医師のDRレベルの診断は、微小動脈瘤（MA）や汎網膜レーザー瘢痕（PRP）などの一連の診断概念を評価することに依存
- TCAVを使用して、モデルに対するこれらの概念の重要性をテストした。

結果

- 一部のDRレベルでは、TCAVは正しい診断概念が重要であると特定した
  - 図10（上）に示すように、TCAVスコアは、DRレベル4に関連する概念では高く、診断以外の概念では低くなった
- 一方、DRレベル1の場合、TCAVの結果は医師の経験則とは異なる場合があった（図10下）
  - たとえば、動脈瘤（HMA）は、より高いDRレベルの診断であるにもかかわらず、比較的高いTCAVスコアだった（図10の下）
  - この発見と一致して、モデルは多くの場合、レベル1（軽度）をレベル2（中程度）として過剰予測していた。 このことを考えると、医師は、「モデルにレベル1のHMAの重要性を強調しないように伝えたい」と述べた



## 5.Conclusion and Future Work

- TCAVは、ディープラーニングモデルの内部状態の線形解釈を作成するステップ
- 概念画像をに対する感度をみることで、ピクセル単位ではなくより概念的に解釈することができる
- 画像分類に焦点をあててきたが、オーディオ、ビデオ、シーケンスなどに適用すると、新しい洞察が得られる場合がある
