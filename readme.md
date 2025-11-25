# YOLO Finetuning
For identification of competition items

## Requirements
Python 3.10，use `pip install -r requirements.txt`

Download 'sam_vit_b_01ec64.pth' to .

If you use the conda yml file, make sure to install LangSAM and SAM2 manually from github.

## Dataset construction
Currently only supports live sampling through a Realsense camera.

You may change the directory of saved samples in `.env`. Ensure the folder under `DATASET_DIR` is empty (otherwise old files will be mixed into your new data).


1. Enter the labels and their corresponding GroundingDINO prompts in `resources/ontology.json`：
    ```json
    {"<GroudingDINO prompt>" : "label"}
    ```

2. Connect Realsense camera to computer using USB cable。

3. cd into `yolo_tuning` and activate your conda environment
    ```
    conda activate visionTrain
    ```

4 If you wish to train regular YOLO (bounding box only), use `python -m create_dataset`, otherwise, for YOLO-seg, use `python -m create_dataset_seg` to start construction your dataset

5. An OpenCV window should pop up, follow the instructions shown in terminal for a smooth dataset creation process!

6. 标定完后按`q`结束。

## Training

1. cd into `yolo_tuning`

2. use `python -m prepare_dataset` to split the dataset into YOLO-appropriate format

2. Use `python -m tune_YOLOv11` or `python -m tune_YOLOv11_seg` to start training

3. The finishe best segmentation shall be saved to `yolo_finetuned_best.pt` or `yolo_seg_finetuned_best.pt`

## Testing the result of your training
Plug in realsense, cd into `yolo_tuning`, and use `python -m test_new_model` or `python -m test_new_model_seg` to test your newly trained model live!