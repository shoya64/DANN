# LSTM_DANNコード



## 修正点

* ~~バッチサイズが安定しない~~ 
  * ~~バッチサイズ作成時点でサイズ変わる~~ 
  * ~~dataでは問題なくてもmodelに挿入するとX_inputがおかしい~~

```python
torch.Size([100, 50, 310])
torch.Size([100])
torch.Size([100])
X_input
torch.Size([100, 50, 310])
lstm_out
torch.Size([100, 50, 128])
X_input
torch.Size([7, 50, 310])
lstm_out
torch.Size([7, 50, 128])
```

* ~~学習しない~~

  

  ```python
  1 loss: 54.997, training_accuracy: 0.47059, test_accuracy: 0.96000
  2 loss: 54.590, training_accuracy: 0.34876, test_accuracy: 1.00000
  3 loss: 54.840, training_accuracy: 0.42249, test_accuracy: 0.00000
  4 loss: 52.610, training_accuracy: 0.38309, test_accuracy: 0.24000
  5 loss: 56.069, training_accuracy: 0.29479, test_accuracy: 0.00000
  6 loss: 53.615, training_accuracy: 0.40179, test_accuracy: 0.50000
  7 loss: 54.603, training_accuracy: 0.33571, test_accuracy: 0.00000
  8 loss: 57.825, training_accuracy: 0.33692, test_accuracy: 0.00000
  9 loss: 55.786, training_accuracy: 0.40139, test_accuracy: 1.00000
  10 loss: 54.397, training_accuracy: 0.26292, test_accuracy: 1.00000
  11 loss: 52.121, training_accuracy: 0.48718, test_accuracy: 0.00000
  12 loss: 54.358, training_accuracy: 0.48571, test_accuracy: 0.00000
  13 loss: 59.542, training_accuracy: 0.31381, test_accuracy: 0.77000
  14 loss: 54.645, training_accuracy: 0.35476, test_accuracy: 1.00000
  15 loss: 53.459, training_accuracy: 0.31810, test_accuracy: 0.00000
  16 loss: 56.626, training_accuracy: 0.32314, test_accuracy: 0.90000
  17 loss: 51.292, training_accuracy: 0.42238, test_accuracy: 0.95000
  18 loss: 57.218, training_accuracy: 0.24490, test_accuracy: 1.00000
  19 loss: 55.330, training_accuracy: 0.42612, test_accuracy: 0.00000
  20 loss: 53.044, training_accuracy: 0.51326, test_accuracy: 0.34000
  21 loss: 54.706, training_accuracy: 0.26177, test_accuracy: 0.00000
  22 loss: 54.725, training_accuracy: 0.47736, test_accuracy: 0.45000
  23 loss: 54.205, training_accuracy: 0.26613, test_accuracy: 0.00000
  24 loss: 58.570, training_accuracy: 0.21190, test_accuracy: 0.55000
  25 loss: 53.167, training_accuracy: 0.49905, test_accuracy: 0.00000
  ```

  * ~~test_accuracy は別で出す，training_accuracyだけ取りあえず~~
  * ~~BCEWithLogitsLossとCrossEntropyLossの扱い方の問題かも~~
  * ~~Confusion matrixもひどい~~

  ![image-20200602172439030](/Users/furukawashouya/Library/Application Support/typora-user-images/image-20200602172439030.png)

  

*　入力を変えよう

  *　生データを入れる
    *　時系列どうしよう
  *　EMD使ってみる

*　コメントつける

*　~~Confusion matrix の総データ数変わるのなんでだろ~~

*　LSTM層に入力する前に圧縮してみる

  *　時系列そのままで
  *　9の時系列から　→ 意味なさそう

*　窓の掛け方の調節で全結合層の代わりができる？

* 過学習する
  * 正則化
  * 訓練データ増やす
  * 

* 生のEEGデータを入れるなら，EEGデータはなんでもよいのではないか
  * データ数を増やせる
  * 映画を見るときの脳波ほど限定的ではない
  * ドメインラベル，ここでは個人の識別ができるのではないか
* 他の脳波データを用いて事前学習を行っておいてドメインの識別に使用する

# 修正中

* ミニバッチの作り方を変更した．mkbatch
* 

# 結果 #

* 状況ごとの結果

  * バッチサイズ32，エポック数100，dropout=0.001，α = 1.0

  | ![image-20200608122808533](/Users/furukawashouya/Library/Application Support/typora-user-images/image-20200608122808533.png) | ![image-20200608122839759](/Users/furukawashouya/Library/Application Support/typora-user-images/image-20200608122839759.png) |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  |                                                              |                                                              |

  ![image-20200608122903488](/Users/furukawashouya/Library/Application Support/typora-user-images/image-20200608122903488.png)

  | ![image-20200608122915118](/Users/furukawashouya/Library/Application Support/typora-user-images/image-20200608122915118.png) | ![image-20200608122922017](/Users/furukawashouya/Library/Application Support/typora-user-images/image-20200608122922017.png) |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  |                                                              |                                                              |

  * バッチサイズ32，バッチサイズ，dropout=0.01, α=1.0

  

  | ![image-20200608124723695](/Users/furukawashouya/Library/Application Support/typora-user-images/image-20200608124723695.png) | ![image-20200608124730402](/Users/furukawashouya/Library/Application Support/typora-user-images/image-20200608124730402.png) |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  |                                                              |                                                              |

  ![image-20200608125004076](/Users/furukawashouya/Library/Application Support/typora-user-images/image-20200608125004076.png)

  

  | ![image-20200608125016077](/Users/furukawashouya/Library/Application Support/typora-user-images/image-20200608125016077.png) | ![image-20200608125022267](/Users/furukawashouya/Library/Application Support/typora-user-images/image-20200608125022267.png) |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  |                                                              |                                                              |

  

  * バッチサイズ64，エポック数50，dropout=0.001，α=1.0

  | ![image-20200608214810430](/Users/furukawashouya/Library/Application Support/typora-user-images/image-20200608214810430.png) | ![image-20200608214821493](/Users/furukawashouya/Library/Application Support/typora-user-images/image-20200608214821493.png) |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | ![image-20200608214844847](/Users/furukawashouya/Library/Application Support/typora-user-images/image-20200608214844847.png) |                                                              |

  | ![image-20200608215121686](/Users/furukawashouya/Library/Application Support/typora-user-images/image-20200608215121686.png) | ![image-20200608215131117](/Users/furukawashouya/Library/Application Support/typora-user-images/image-20200608215131117.png) |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  |                                                              |                                                              |

  * バッチサイズ64，エポック数50，dropout=0.001，α=0.8

  | ![image-20200608222825205](/Users/furukawashouya/Library/Application Support/typora-user-images/image-20200608222825205.png) | ![image-20200608222832541](/Users/furukawashouya/Library/Application Support/typora-user-images/image-20200608222832541.png) |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | ![image-20200608222845581](/Users/furukawashouya/Library/Application Support/typora-user-images/image-20200608222845581.png) |                                                              |

  | ![image-20200608222856593](/Users/furukawashouya/Library/Application Support/typora-user-images/image-20200608222856593.png) | ![image-20200608222904660](/Users/furukawashouya/Library/Application Support/typora-user-images/image-20200608222904660.png) |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  |                                                              |                                                              |

  * バッチサイズ64，エポック数50，dropout = 0.001，α=0.7，LSTM層：64

  | ![image-20200609114847753](/Users/furukawashouya/Library/Application Support/typora-user-images/image-20200609114847753.png) | ![image-20200609114854236](/Users/furukawashouya/Library/Application Support/typora-user-images/image-20200609114854236.png) |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  |                                                              |                                                              |

  | ![image-20200609115128019](/Users/furukawashouya/Library/Application Support/typora-user-images/image-20200609115128019.png) | ![image-20200609115137803](/Users/furukawashouya/Library/Application Support/typora-user-images/image-20200609115137803.png) |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  |                                                              |                                                              |

  









