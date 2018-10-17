# mnist
## 数据输出
input_data.py: MNIST数据集下载与解压
## 模型选取
+ mnist.py: 构建mnist网络，主函数在fully_connected_feed.py中运行
+ mnist_softmax.py: MNIST机器学习入门
+ mnist_deep.py: 深入MNIST
+ fully_connected_feed.py: TensorFlow运作方式入门
+ mnist_with_summaries.py: Tensorboard训练过程可视化
## 运行结果
### fully_connected_feed
```
/opt/modules/anaconda3/envs/mnist/bin/python /home/lee/PycharmProjects/mnist/fully_connected_feed.py
Extracting Mnist_data/train-images-idx3-ubyte.gz
Extracting Mnist_data/train-labels-idx1-ubyte.gz
Extracting Mnist_data/t10k-images-idx3-ubyte.gz
Extracting Mnist_data/t10k-labels-idx1-ubyte.gz
Step 0: loss = 2.31 (0.043 sec)
Step 100: loss = 2.12 (0.003 sec)
Step 200: loss = 1.82 (0.003 sec)
Step 300: loss = 1.46 (0.002 sec)
Step 400: loss = 1.22 (0.002 sec)
Step 500: loss = 0.88 (0.002 sec)
Step 600: loss = 1.00 (0.002 sec)
Step 700: loss = 0.74 (0.005 sec)
Step 800: loss = 0.71 (0.002 sec)
Step 900: loss = 0.59 (0.003 sec)
Training Data Eval:
  Num examples: 55000  Num correct: 46679  Precision @ 1: 0.8487
Validation Data Eval:
  Num examples: 5000  Num correct: 4283  Precision @ 1: 0.8566
Test Data Eval:
  Num examples: 10000  Num correct: 8588  Precision @ 1: 0.8588
Step 1000: loss = 0.65 (0.010 sec)
Step 1100: loss = 0.55 (0.116 sec)
Step 1200: loss = 0.45 (0.002 sec)
Step 1300: loss = 0.55 (0.003 sec)
Step 1400: loss = 0.49 (0.003 sec)
Step 1500: loss = 0.31 (0.002 sec)
Step 1600: loss = 0.53 (0.002 sec)
Step 1700: loss = 0.44 (0.003 sec)
Step 1800: loss = 0.27 (0.002 sec)
Step 1900: loss = 0.33 (0.002 sec)
Training Data Eval:
  Num examples: 55000  Num correct: 49171  Precision @ 1: 0.8940
Validation Data Eval:
  Num examples: 5000  Num correct: 4502  Precision @ 1: 0.9004
Test Data Eval:
  Num examples: 10000  Num correct: 8992  Precision @ 1: 0.8992

Process finished with exit code 0
```
### mnist_softmax
```
/opt/modules/anaconda3/envs/mnist/bin/python /home/lee/PycharmProjects/mnist/mnist_softmax.py
Extracting Mnist_data/train-images-idx3-ubyte.gz
Extracting Mnist_data/train-labels-idx1-ubyte.gz
Extracting Mnist_data/t10k-images-idx3-ubyte.gz
Extracting Mnist_data/t10k-labels-idx1-ubyte.gz
0.9161

Process finished with exit code 0
```
### mnist_deep
```
/opt/modules/anaconda3/envs/mnist/bin/python /home/lee/PycharmProjects/mnist/mnist_deep.py
Extracting Mnist_data/train-images-idx3-ubyte.gz
Extracting Mnist_data/train-labels-idx1-ubyte.gz
Extracting Mnist_data/t10k-images-idx3-ubyte.gz
Extracting Mnist_data/t10k-labels-idx1-ubyte.gz
step 0, train accuracy 0.06
step 100, train accuracy 0.9
step 200, train accuracy 1
step 300, train accuracy 0.9
step 400, train accuracy 0.98
step 500, train accuracy 0.96
step 600, train accuracy 1
step 700, train accuracy 0.94
step 800, train accuracy 1
step 900, train accuracy 1
step 1000, train accuracy 1
step 1100, train accuracy 0.96
step 1200, train accuracy 0.98
step 1300, train accuracy 1
step 1400, train accuracy 1
step 1500, train accuracy 1
step 1600, train accuracy 0.98
step 1700, train accuracy 0.94
step 1800, train accuracy 0.94
step 1900, train accuracy 1
step 2000, train accuracy 0.98
step 2100, train accuracy 0.98
step 2200, train accuracy 0.98
step 2300, train accuracy 1
step 2400, train accuracy 0.98
step 2500, train accuracy 1
step 2600, train accuracy 0.98
step 2700, train accuracy 1
step 2800, train accuracy 0.94
step 2900, train accuracy 1
step 3000, train accuracy 0.98
step 3100, train accuracy 1
step 3200, train accuracy 0.96
step 3300, train accuracy 0.96
step 3400, train accuracy 0.96
step 3500, train accuracy 0.98
step 3600, train accuracy 0.96
step 3700, train accuracy 1
step 3800, train accuracy 1
step 3900, train accuracy 0.98
step 4000, train accuracy 0.98
step 4100, train accuracy 0.98
step 4200, train accuracy 1
step 4300, train accuracy 1
step 4400, train accuracy 0.98
step 4500, train accuracy 0.98
step 4600, train accuracy 1
step 4700, train accuracy 0.98
step 4800, train accuracy 1
step 4900, train accuracy 0.96
step 5000, train accuracy 1
step 5100, train accuracy 0.98
step 5200, train accuracy 0.96
step 5300, train accuracy 0.98
step 5400, train accuracy 0.98
step 5500, train accuracy 1
step 5600, train accuracy 1
step 5700, train accuracy 1
step 5800, train accuracy 1
step 5900, train accuracy 1
step 6000, train accuracy 0.96
step 6100, train accuracy 0.98
step 6200, train accuracy 1
step 6300, train accuracy 1
step 6400, train accuracy 1
step 6500, train accuracy 1
step 6600, train accuracy 1
step 6700, train accuracy 0.98
step 6800, train accuracy 0.98
step 6900, train accuracy 1
step 7000, train accuracy 1
step 7100, train accuracy 1
step 7200, train accuracy 1
step 7300, train accuracy 1
step 7400, train accuracy 1
step 7500, train accuracy 1
step 7600, train accuracy 1
step 7700, train accuracy 0.98
step 7800, train accuracy 1
step 7900, train accuracy 1
step 8000, train accuracy 1
step 8100, train accuracy 1
step 8200, train accuracy 1
step 8300, train accuracy 1
step 8400, train accuracy 1
step 8500, train accuracy 1
step 8600, train accuracy 1
step 8700, train accuracy 1
step 8800, train accuracy 0.98
step 8900, train accuracy 1
step 9000, train accuracy 1
step 9100, train accuracy 1
step 9200, train accuracy 1
step 9300, train accuracy 1
step 9400, train accuracy 0.98
step 9500, train accuracy 0.98
step 9600, train accuracy 0.98
step 9700, train accuracy 1
step 9800, train accuracy 1
step 9900, train accuracy 1
step 10000, train accuracy 1
step 10100, train accuracy 1
step 10200, train accuracy 1
step 10300, train accuracy 0.98
step 10400, train accuracy 1
step 10500, train accuracy 1
step 10600, train accuracy 1
step 10700, train accuracy 1
step 10800, train accuracy 1
step 10900, train accuracy 1
step 11000, train accuracy 1
step 11100, train accuracy 1
step 11200, train accuracy 1
step 11300, train accuracy 1
step 11400, train accuracy 1
step 11500, train accuracy 1
step 11600, train accuracy 1
step 11700, train accuracy 1
step 11800, train accuracy 1
step 11900, train accuracy 1
step 12000, train accuracy 1
step 12100, train accuracy 1
step 12200, train accuracy 1
step 12300, train accuracy 1
step 12400, train accuracy 1
step 12500, train accuracy 1
step 12600, train accuracy 1
step 12700, train accuracy 0.96
step 12800, train accuracy 1
step 12900, train accuracy 1
step 13000, train accuracy 0.98
step 13100, train accuracy 1
step 13200, train accuracy 1
step 13300, train accuracy 1
step 13400, train accuracy 1
step 13500, train accuracy 1
step 13600, train accuracy 1
step 13700, train accuracy 1
step 13800, train accuracy 1
step 13900, train accuracy 1
step 14000, train accuracy 1
step 14100, train accuracy 1
step 14200, train accuracy 1
step 14300, train accuracy 1
step 14400, train accuracy 1
step 14500, train accuracy 0.98
step 14600, train accuracy 1
step 14700, train accuracy 1
step 14800, train accuracy 1
step 14900, train accuracy 1
step 15000, train accuracy 1
step 15100, train accuracy 1
step 15200, train accuracy 1
step 15300, train accuracy 1
step 15400, train accuracy 1
step 15500, train accuracy 1
step 15600, train accuracy 1
step 15700, train accuracy 0.98
step 15800, train accuracy 0.98
step 15900, train accuracy 1
step 16000, train accuracy 1
step 16100, train accuracy 1
step 16200, train accuracy 1
step 16300, train accuracy 1
step 16400, train accuracy 1
step 16500, train accuracy 0.98
step 16600, train accuracy 1
step 16700, train accuracy 1
step 16800, train accuracy 1
step 16900, train accuracy 1
step 17000, train accuracy 1
step 17100, train accuracy 1
step 17200, train accuracy 1
step 17300, train accuracy 1
step 17400, train accuracy 1
step 17500, train accuracy 1
step 17600, train accuracy 1
step 17700, train accuracy 1
step 17800, train accuracy 1
step 17900, train accuracy 1
step 18000, train accuracy 1
step 18100, train accuracy 1
step 18200, train accuracy 1
step 18300, train accuracy 0.96
step 18400, train accuracy 1
step 18500, train accuracy 1
step 18600, train accuracy 1
step 18700, train accuracy 1
step 18800, train accuracy 1
step 18900, train accuracy 1
step 19000, train accuracy 1
step 19100, train accuracy 1
step 19200, train accuracy 1
step 19300, train accuracy 1
step 19400, train accuracy 0.98
step 19500, train accuracy 1
step 19600, train accuracy 1
step 19700, train accuracy 1
step 19800, train accuracy 1
step 19900, train accuracy 1
```
### mnist_with_summaries
```
/opt/modules/anaconda3/envs/mnist/bin/python /home/lee/PycharmProjects/mnist/mnist_with_summaries.py
Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
Extracting Mnist_data/train-images-idx3-ubyte.gz
Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
Extracting Mnist_data/train-labels-idx1-ubyte.gz
Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Extracting Mnist_data/t10k-images-idx3-ubyte.gz
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Extracting Mnist_data/t10k-labels-idx1-ubyte.gz
WARNING:tensorflow:Passing a `GraphDef` to the SummaryWriter is deprecated. Pass a `Graph` object instead, such as `sess.graph`.
Accuracy at step 0: 0.098
Accuracy at step 10: 0.7404
Accuracy at step 20: 0.8041
Accuracy at step 30: 0.814
Accuracy at step 40: 0.8421
Accuracy at step 50: 0.8381
Accuracy at step 60: 0.8518
Accuracy at step 70: 0.8403
Accuracy at step 80: 0.8066
Accuracy at step 90: 0.8624
Accuracy at step 100: 0.871
Accuracy at step 110: 0.8864
Accuracy at step 120: 0.8973
Accuracy at step 130: 0.8957
Accuracy at step 140: 0.8817
Accuracy at step 150: 0.8612
Accuracy at step 160: 0.8835
Accuracy at step 170: 0.8527
Accuracy at step 180: 0.8993
Accuracy at step 190: 0.9012
Accuracy at step 200: 0.8909
Accuracy at step 210: 0.9038
Accuracy at step 220: 0.8846
Accuracy at step 230: 0.8921
Accuracy at step 240: 0.9017
Accuracy at step 250: 0.901
Accuracy at step 260: 0.8994
Accuracy at step 270: 0.9067
Accuracy at step 280: 0.9004
Accuracy at step 290: 0.9079
Accuracy at step 300: 0.9014
Accuracy at step 310: 0.8989
Accuracy at step 320: 0.8703
Accuracy at step 330: 0.8248
Accuracy at step 340: 0.882
Accuracy at step 350: 0.8976
Accuracy at step 360: 0.8746
Accuracy at step 370: 0.9066
Accuracy at step 380: 0.9081
Accuracy at step 390: 0.9
Accuracy at step 400: 0.9033
Accuracy at step 410: 0.9054
Accuracy at step 420: 0.9042
Accuracy at step 430: 0.9066
Accuracy at step 440: 0.9057
Accuracy at step 450: 0.9088
Accuracy at step 460: 0.9051
Accuracy at step 470: 0.9035
Accuracy at step 480: 0.9108
Accuracy at step 490: 0.908
Accuracy at step 500: 0.9093
Accuracy at step 510: 0.8299
Accuracy at step 520: 0.9018
Accuracy at step 530: 0.9133
Accuracy at step 540: 0.9085
Accuracy at step 550: 0.8733
Accuracy at step 560: 0.9076
Accuracy at step 570: 0.9124
Accuracy at step 580: 0.9151
Accuracy at step 590: 0.9124
Accuracy at step 600: 0.9111
Accuracy at step 610: 0.8914
Accuracy at step 620: 0.9126
Accuracy at step 630: 0.915
Accuracy at step 640: 0.9153
Accuracy at step 650: 0.9189
Accuracy at step 660: 0.9155
Accuracy at step 670: 0.9129
Accuracy at step 680: 0.9139
Accuracy at step 690: 0.9042
Accuracy at step 700: 0.9146
Accuracy at step 710: 0.9175
Accuracy at step 720: 0.9151
Accuracy at step 730: 0.916
Accuracy at step 740: 0.9154
Accuracy at step 750: 0.9136
Accuracy at step 760: 0.9147
Accuracy at step 770: 0.9113
Accuracy at step 780: 0.9042
Accuracy at step 790: 0.9001
Accuracy at step 800: 0.9
Accuracy at step 810: 0.905
Accuracy at step 820: 0.9037
Accuracy at step 830: 0.9006
Accuracy at step 840: 0.8941
Accuracy at step 850: 0.9166
Accuracy at step 860: 0.9145
Accuracy at step 870: 0.914
Accuracy at step 880: 0.917
Accuracy at step 890: 0.906
Accuracy at step 900: 0.9
Accuracy at step 910: 0.9138
Accuracy at step 920: 0.9064
Accuracy at step 930: 0.9146
Accuracy at step 940: 0.9089
Accuracy at step 950: 0.9155
Accuracy at step 960: 0.9167
Accuracy at step 970: 0.9174
Accuracy at step 980: 0.9174
Accuracy at step 990: 0.9123

Process finished with exit code 0
```
