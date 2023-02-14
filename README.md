# DeepLabV3plus_fusion
## 基于DeepLabV3 Plus的语义分割模型改进
## 基于Encoder-Decoder和ASPP的卷积神经网络结构

### Backbone: "Xception, MobileNet"
### Neck: ASPP module
### Head: deepLabV3plus
### low_feature dim = 256


### 梯度下降优化器Optimizer：
#### Bubbliiiing
SGD: 7e-3
Adam: 5e-4

#### WZMIAOMIA
SGD: 1e-4
Adam: 5e-4

#### shadousheng
SGD:
B=32
lr=0.1 * 32 / 256
momentum=0.9
weight_decay=1e-4


AdamW:
B=32
lr=4e-3 * 32 / 64
weight_decay=0.05
eps=1e-8
betas=(0.9, 0.999)

B=16
lr=5e-4 * 16 / 64
weight_decay=0.05
eps=1e-8
betas=(0.9, 0.999)


## 模型version更新的地方
（1）stage1的expanded_rate更改为3