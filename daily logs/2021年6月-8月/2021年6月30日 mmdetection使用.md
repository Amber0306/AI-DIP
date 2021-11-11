# 1.MM Detection学习

## 1.1源码结构

训练命令

```sh
python demo/image_demo.py demo/demo.jpg configs/mask_rcnn/mask_rcnn_r101_fpn_1x_coco.py checkpoints/mask_rcnn_r101_fpn_1x_coco_20200204.pth
```

r101层数多于r50

configs用来加载模型，checkpoints这里指明说明的权重。

### 1.1.1问题

pycharm运行命令

```shell
./pycharm.sh
sh pycharm.sh
```

## 1.2魔改源码

暂且推后

## 1.3labelme数据集生成coco数据集

### bug1问题描述

```shell
conda install -p
albumentations>=0.32 -y
```

json.decoder.JSONDecodeError: Unterminated string starting at: line 790799 column 19

### 解决方案

未解决，放弃下载对应包

原因可能是命令格式不对？

## 1.4训练自己的数据并可视化训练结果

### 1.4.1处理数据

使用labelme2coco.py处理数据，并拷贝到data文件夹下

### 1.4.2修改datasets

修改configs/_base_/datasets中的coco_detection.py coco_instance.py coco_instance)semantic.py中的img_scale 共六处需要修改

### 1.4.3修改models

修改configs/_base_/models/cascade_rcnn_r50_fpn.py

修改num_classes，不需要添加背景

### 1.4.4修改class_names

修改mmdet/core/evaluation/class_names.py中的coco_classes()函数

修改mmdet/datesets/coco.py中CocoDataset类中的CLASSES换成自己的label

### 1.4.5tools文件夹

功能性的api

#### 训练 train.py

```python
python tools/train.py configs/mask_rcnn/mask_rcnn_r101_fpn_2x_coco.py --gpus 1 --work-dir work_dir_gzq/test01
```

ctrl+c杀死

打开工作目录里的配置文件

参数：

```python
optimizer 优化器
lr 学习率
total_epochs 训练轮数
checkpoint_config interval=4 每四个轮次保存一个模型
log_config interval=5 每五次迭代保存一次日志
```

再次执行

```shell
python tools/train.py work_dir_gzq/test01/mask_rcnn_r101_fpn_2x_coco.py --gpus 1
```

如果遇到out of memory 可以调小img_scale

#### 分析日志analyze_logs.py

执行命令

```shell
python tools/analysis_tools/analyze_logs.py plot_curve work_dir/test01/2020.log.json --keys acc 
```

```shell
python tools/analyze_logs.py plot_curve work_dir/test01/2020.log.json --keys loss_cls loss_bbox loss_mask --out out.pdf
```

#### 测试test.py

执行命令

```shell
python tools/test.py work_dir_gzq/test01/mask_coco.py work_dir_gzq/test01/epoch_24.pth --show
```

```shell
python tools/test.py work_dir_gzq/test01/mask_coco.py work_dir_gzq/test01/epoch_24.pth --out myresult.pkl 
```

