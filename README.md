# HGRR
The implementation of our paper "Hue Guidance Network for Single Image Reflection Removal".

## Implementation

1. Prepare the training data as follows:
    ```
      |--training_data
           |--JPEGImages  # VOC2012 images for synthesis the reflection data 
           |--real_train  # real world reflection images and corresponding GT from previous methods, eg, real, nature 
               |--blended   #   reflection images
               |--transmission_layer   #  ground truth (GT)
           
    ```
2. training phase

      ```python train_HGRR.py --name train_DB_wboosting --hyper --inet HGRR_wboosting1128 --lambda_vgg 0.02 --lambda_newloss 50 --lambda_newloss_H 20```

3. evaluation phase

      ``` python test_HGRR.py --name test -r --inet HGRR_wboosting1128 --icnn_path best.pt --hyper``` 


## Results & Pre-trained model
[Results of Reflection Benchmarks](https://drive.google.com/drive/folders/1JPeI9K-KxUbLk9WNlPIJsdWgrMmW2kPl?usp=sharing)

[Pre-trained model](https://drive.google.com/drive/folders/1JPeI9K-KxUbLk9WNlPIJsdWgrMmW2kPl?usp=sharing)

## Concat
If you have any problem, please feel free to contact me (zyr@mail.ustc.edu.cn).

## Acknowledgments
* Our codes are inspired by [ERRNet](https://github.com/Vandermode/ERRNet)
