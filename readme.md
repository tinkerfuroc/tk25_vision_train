# 视觉模型微调
用于Robocup物品识别

## Requirements
Python 3.10，安装requriements.txt中内容。

## 构造数据集
目前至支持使用Realsense相机实时采样。

保存的路径在`.env`文件中修改。开始采样前确保`DATASET_DIR`文件夹为空（否则会混进前一次构造的脏数据）。

1. 在`resources/ontology.json`中输入希望识别的类和其GroudingDINO的prompt。格式为：
    ```json
    {"<GroudingDINO prompt>" : "label"}
    ```

2. 使用USB将Realsense连接到电脑。

3. 进入`yolo_tuning`文件夹

4. 使用`python -m create_dataset`开始构造数据集

5. 对于出现的每张照片，使用`上下箭头`选中不同的bounding box，并按`d`删除不正确的框。没问题后按`s`保存。如果一张图片全部错误或者没有识别出来，按`空格`跳过。
    * 若每次的错误或漏识别都非常多，更改GroundingDINO的prompt
    * 保存前确保没有错标的bounding box，不要污染数据源！！！
    * 每种物品至少在数据集里出现几百次（四位数就更好了）。

6. 标定完后按`q`结束。

## 训练
第一次使用时会从huggingface上下载初始权重，确保能连上（看看梯子）。

1. 进入`yolo_tuning`文件夹

2. 使用`python -m tune_YOLOv11`开始训练
    * 训练太慢可以将第148行`model = YOLO('yolo11s.pt')`的权重名称呢该改成`yolo11n.pt`(nano)。训练很快也可以调成medium(m)，large(l)，但是s对于Robocup应该够用。

3. 训练结束后将出现的`yolo_finetuned_best.pt`拷走就好。训练情况在`CHECKPOINT_DIR`文件夹中查看。

## 测试训练结果
连上realsense，进入`yolo_tuning`文件夹，使用`python -m test_new_model`即可开始实时测试新权重的识别能力。