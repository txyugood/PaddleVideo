# PYSKL: Towards Good Practices for Skeleton Action Recognition（基于Paddle复现STGCN++）
## 1.简介
本文作者提出了一个原始的GCN模型ST-GCN++。仅对原始ST-GCN进行简单修改。ST-GCN++重新设计了空间模块和时间模块，ST-GCN++就获得了与具有复杂注意机制的SOTA识别性能。同时，计算开销也减大大的减少了。


## 2.复现精度
在NTU60-HRNET数据集上的测试效果如下表。

| NetWork | epochs | opt  | batch_size | dataset | top1 acc |
| --- | --- | ---  | --- | --- | --- |
| STGCN++ | 16 | SGD  | 128 | UCF-101 | 97.56% |

## 3.数据集
数据集下载地址:

[https://aistudio.baidu.com/aistudio/datasetdetail/167195](https://aistudio.baidu.com/aistudio/datasetdetail/167195)



## 4.环境依赖
PaddlePaddle == 2.3.2 
## 5.快速开始
### 训练：
```shell
cd PaddleVideo
pip install -r requirements.txt
ln -s path/to/ntu60_hrnet.pkl data/ntu60_hrnet.pkl
nohup python -u main.py --validate -c configs/recognition/stgcn_plusplus/stgcn_plusplus_ntucs.yaml --seed 9999 > train.log &
tail -f train.log
```
validate: 开启验证

-c: 模型配置路径

seed: 随机种子


### 测试：
 
使用最优模型进行评估.

最优模型下载地址：


链接: https://pan.baidu.com/s/1X8-M1IzEQqu1s_wtYtLFyw 

提取码: s8eo 



```shell
python -u main.py --test
-c
configs/recognition/stgcn_plusplus/stgcn_plusplus_ntucs.yaml
--weights path/to/STGCN_PlusPlus_best.pdparams
```

test: 测试模式

-c: 模型配置

weights: 模型权重路径

测试结果

```shell
[09/07 23:45:20] [TEST] Processing batch 18929/18932 ...
[09/07 23:45:20] [TEST] Processing batch 18930/18932 ...
[09/07 23:45:20] [TEST] Processing batch 18931/18932 ...
[09/07 23:45:20] [TEST] finished, avg_acc1= 0.9755968451499939, avg_acc5= 0.9996830821037292
```


### 模型导出
模型导出可执行以下命令：

```shell
python3.7 tools/export_model.py -c configs/recognition/stgcn_plusplus/stgcn_plusplus_ntucs.yaml --save_name inference -p=path/to/STGCN_PlusPlus_best.pdparams -o=./output/STGCN_PlusPlus/
```

参数说明：

-c: 模型配置路径

save_name:导出的静态图文件名

-p: 动态图模型权重路径

-o: 输出结果保存路径

### Inference推理

可使用以下命令进行模型推理。该脚本依赖auto_log, 请参考下面TIPC部分先安装auto_log。infer命令运行如下：

```shell
python tools/predict.py
--config configs/recognition/stgcn_plusplus/stgcn_plusplus_ntucs.yaml --use_gpu=True --model_file=./output/STGCN_PlusPlus/inference.pdmodel --params_file=./output/STGCN_PlusPlus/inference.pdiparams --batch_size=1 --input_file=./data/stdgcn_plusplus_data/example_ntu60_skeleton.pkl 
```

参数说明:

use_gpu:是否使用GPU

model_file: 模型结构文件路径，由export_model.py脚本导出。

params_file: 模型权重文件路径，由export_model.py脚本导出。

batch_size: 批次大小

input_file: 输入文件路径




### TIPC基础链条测试

该部分依赖auto_log，需要进行安装，安装方式如下：

auto_log的详细介绍参考[https://github.com/LDOUBLEV/AutoLog](https://github.com/LDOUBLEV/AutoLog)。

```shell
git clone https://gitee.com/Double_V/AutoLog
cd AutoLog/
pip3 install -r requirements.txt
python3 setup.py bdist_wheel
pip3 install ./dist/auto_log-1.2.0-py3-none-any.whl
```


```shell
bash test_tipc/prepare.sh test_tipc/configs/STGCN_PlusPlus/train_infer_python.txt 'lite_train_lite_infer'

bash test_tipc/test_train_inference_python.sh test_tipc/configs/STGCN_PlusPlus/train_infer_python.txt 'lite_train_lite_infer'
```

测试结果如截图所示：

<img src=./test_tipc/data/tipc_result.png></img>


## 6.模型信息

| 信息 | 描述 |
| --- | --- |
|模型名称| STGCN++ |
|框架版本| PaddlePaddle==2.3.2|
|应用场景| 骨骼识别 |
