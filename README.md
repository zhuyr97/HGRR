# HGRR
The implementation of our paper "Hue Guidance Network for Single Image Reflection Removal".

## Implementation
* Training Phase

   ```python train_HGRR.py --name train_DB_wboosting --hyper --inet HGRR_wboosting1128 --lambda_vgg 0.02 --lambda_newloss 50 --lambda_newloss_H 20```

* Testing Phase

   ``` python test_HGRR.py``` 


## Download Results
[Results of Reflection Benchmarks](https://drive.google.com/drive/folders/1JPeI9K-KxUbLk9WNlPIJsdWgrMmW2kPl?usp=sharing)

## Concat
If you have any problem, please feel free to contact me (zyr@mail.ustc.edu.cn).

## Acknowledgments
* Our codes are inspired by [ERRNet](https://github.com/Vandermode/ERRNet)
