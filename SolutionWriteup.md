## Deciding Deep Learning Framework
* It's emphasized that 'generalizability' is the key goal of this competition. Given the small amount of training set and the diversity, we concluded that we'll need a lot of data augmentation in data pipeline. Furthermore, easy in debugging and integration with existing Python modules would be important to us. We decided to use PyTorch for this competion. 

## Starting from Semantic Segmentation
* We started from UNet, and did lots of experiments with regarding to data augmentation (random crop, flip, rotate, resize, color jitter, elasitc distortion, color invert, clahe, and gaussian noise). We trained a simplifed UNet from scratch without external dataset, optimized with Binary Cross-Entropy (BCE) loss function. It could reach LB 0.34. We then optimize for Jaccard/IoU + BCE, it helped to reach LB 0.36.
* Read from the winning solution writeup of Kaggle Carvana Image Masking Challenge, we expected to get better LB score with transfer learning. We picked up VGG16 as the pretrained encoder, but unfortunately, we didn't see the improvement at that time.
* Note: 
    - color invert, clahe, and gaussian noise didn't help in our experiments
    - we revisit transfer learning in post-competition

## Toward Instance Segmentation
* Our first step toward instance segmentation is trying to leverage computer vision morphology algorithm. By using watershed with peak local max, it could reach LB 0.41
* After reviewing the visualized prediction, we found the model performed poorly in histology images, so we added TCGA dtaset to training set, and it could reach LB 0.44
* Inspired by DCAN (Deep Contour-Aware Networks for Accurate Gland Segmentation), we revise the UNet model to be multitasked. One head is semantic prediction, another head is contour prediction. However, a naive (semantic - contour) as instance segmentation prediction is no good, so we instead use (semantic - contour) as the marker of watershed algorithm, and it brought us to LB 0.45
* We revisited the UNet paper, and thought the 'weight map' to force the network to learn the border pixels should be helpful. We then implemented a weight map, which emphasizes on borders & centeriods (especially for small cells), and it brought us to LB 0.47. [HERE: An illustration image]  
* We compared the partitioning result of watershed and random walker for touching objects, and felt the result of random walker is more natural from our perspective. After changed post-processing to random walker, it brought us to LB 0.49
* Reviewed aforementioned markers (semantic - contour) visually, there are always some poor cases and it's hard to design rules for them. Therefore, we decided to let the model learn and predict the markers by itself. The third head was added to predict markers, whose ground truths are derived from shrinked cell masks. And to further conquer the data imbalance issue, we implemented Focal loss in addition to aforementioned Jaccard/IoU loss function and weight map mechanism. It brought us to LB 0.50.
* By adding synthesized images of touching/overlapping cells to training set, it could reach LB 0.52    

## Ensemble
* We found several stage 1 test images have just 3 to 5 big cells, and our model didn't perform well for all these cases. To address this, we tried to expand the receptive field of encoder part of UNet via dilated convolution. We found dilated and non-dilated encoder seems to be complement each other, so we ensemble these two models by averaging their pixel-wise prediction of three heads. 
* We implemented Test Time Augmentation (TTA) with horizontal flip, vertical flip, horizontal and vertical flip. However, an implementation error onr eflect padding & flip pipelinea at that time made us drop TTA at that time.
* We also added some training examples from external cell-tracking dataset, and the ensemble brought us to LB 0.534 

## Struggled in Stage 2
* Surprised at the huge difference between stage1 train/test and stage 2 test dataset, we add additional BBBC018, BBBC020, and stage 1 test to further train the models.
* We made a mistake of no objective local cross validataion to judge overfitting, but relied on human judgement on visualized predictions (A discussion point later on)
* We picked up the final submissions which were trained longer, but it turned out that we chose the worse ones. 
* Here we actually stuck in no validataion set similar to test set (Andrew Ng had a great talk about Nuts and Bolts of Applying Deep Learning in this regard), and we don't
know the way to improve our model and result in such a short stage 2 period in this competition.
* One of the biggest challenge for us is to figure out what we did wrong and how to improve it in the next one, so we are eager to learn from others how to do this appropriately. 

## Post-Competition & Late Submissions
* Read some nice writeups of top winning solutions (UNet-based), we found we had many common technical approaches. We then tried to experiment with those we haven't tried in the competition: 
- Data Augmentation: channel shuffle and rgb/gray color space transform
- Implement TTA correctly, which turns out help the score a lot (0.569 -> 0.580)
- Transfer learning with deep neural network (Resnet)
- Shared decoder vs separated decoders for multitasks, and ensemble them for final prediction.
* We used stage 1 test as validation set (totally isolated from training set), and saw the high variance of LB private score (0.56 ~ 0.623) along the training epochs. [HERE A figure] 

The figure hints the data distribution of stage 1 test set is not similar to stage 2 test set, it's somehow like a lottory here since we can't have a reasonable local validation set for stage 2 test set (with tons of on-purposed posion images) in such a short period? We are really appreciate of any sharing on practical and applicable strategy/approach to this dilemma in Kaggle competition.