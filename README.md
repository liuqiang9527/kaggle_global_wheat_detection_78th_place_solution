# kaggle_global_wheat_detection_78th_place_solution
Based on efficientdet-pytorch and yolov5

This is my first solo silver medal. I will keep my solution simple and short.
Special thank @rwightman for amazing repo (https://github.com/rwightman/efficientdet-pytorch), thank @ultralytics for amazing repo (https://github.com/ultralytics/yolov5.git) too.

Unfortunately, yolov5 is banned in this competition because of its licene

First of all, heavy data augmentation is used, such as random size cropping, horizontal filpping, vertical flipping, ToGray and etc.
Additionally, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, Blur, CLAHE, Sharpen, Emboss, RandomBrightnessContrast, HueSaturationValue will also be useful due to some discussion.
**Notice, Custom mosaic data augmentation and mixup is useful besides simple data augmentation above.**

Mosaic is a data augmentation method that combines 4 training images into one for training (instead of 2 in CutMix). Instead of randomly cropping a piece of the image, I create custom mosaic augmentation as bellow to keep the border information:
- First, we get a random point in 2048 * 2048 region, this point can split the region.
- secondly, we get 4 random images and pad them into 4 splitted region, the boxes should also be padded into the corresponding position.

Notice that the implementations in gwd_traing.py and dateset.py is different

We also should pay attention to the implementation of mixup, because the wheat images have differences in color, for example, there are yellow wheat images and green wheat images, so we can't add the weighted pixels easily.

Fortunately, I write the implementation via cv2.addWeighted, then, we can avoid some awful pixel, it's very useful.

Additionally, I have used 5 folds, stratified-kfold, splitted by source(usask_1, arvalis_1, arvalis_2…).

TTA(test time augmentation), wbf, pseudo labeling also contributes lots in accurancy.

**Pseudo labeling**
Pseudo labeling step1: train EfficientDet-d5 10 epochs on trainset + hidden testset (output of **ensembling**) with mixup, load checkpoint from base
Pseudo labeling step2: continue train EfficientDet-d5 6 epochs on trainset + hidden testset (output of pseudo labeling step1) with mixup, load checkpoint from pseudo labeling step1
