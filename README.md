# SkinCancer
用于皮肤癌和黑色素瘤分类。同时也是基于Pytorch用做**分类、回归以及多任务学习**的高可复用的模版。
## 1 结构与模块解析
![](https://jerrymazeyu.oss-cn-shanghai.aliyuncs.com/2020-05-16-084052.png)

## 2 如何复用？
### 2.1 根据指定格式导入数据放在/data目录中
### 2.1 修改Config中的内容
其中dataset_name、classes需要修改，还有定义好custom_data的dataloader名字，若自定义了模型则需要将model_name改成model实例的名字，同时需要修改trainer，如果修改了loss的话则还需要修改criterion、loss_alpha等
### 2.2 Trainer部分有一些可视化部分需要重载
这在代码中有写到
### 2.3 在main中根据示例修改
#### 2.3.1 定义sym_dict是先验认知的json文件
#### 2.3.2 img_call_back(imgpath, img)
接受两个参数，imgpath是图像的绝对路径，可以对其进行处理，img是经过resize等操作后的Tensor，注意：最后返回的也应该是一个Tensor！
#### 2.3.3 get_label(imgpath, classid)
前面是对于图像的回调，这是对于label的回调，使用方法是一样的，若不做更改直接return classid即可
#### 2.2.4 定义custon_data
将参数传入
#### 2.2.5 若定义过nn后需要在这里进行实例化，注意和config中的需要对应
#### 2.2.6 定义Trainer
需要继承BasicTrainer,_evaluate(self, opts, labels)其中opts是model的输出、label是得到的label(一个batch经过get_label)，需要给出一个衡量的指标，比如准确率，f1-score等，只有在val数据集上满足了这样的条件后才可以更新参数
#### 2.2.7 将参数传入ce中，训练
### 2.4 bash utils/show.sh查看可视化
