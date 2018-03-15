# Description

Kaggle 2018 Data Science Bowl: find the nuclei in divergent images to advance medical discovery

## Development Plan:

* Platform and framework
  - [x] PyTorch
  - [x] macOS
  - [x] Ubuntu
* Explore model architecture
  - [x] UNet
    - [x] Contour Aware model (2 tasks)
    - [ ] Contour Aware Marker model (3 tasks)
    - [ ] Boundaries detection for adjacent nuclei only?
  - [x] DCAN (switch to multitask UNet judged bt experimental results)
    - [x] Training efficiency for contour detection
  - [ ] Mask RCNN
  - [ ] Dilated Convolution
  - [x] Dropout
  - [x] Batch normalization
  - [x] Transfer learning
  - [x] Score function
  - Cost functions
    + [x] binary cross entropy
    + [x] pixel wise IoU, regardless of instances
    + [x] loss weight per distance of instances's boundary 
    + [x] Focal loss (attention on imbalance loss)
    + [ ] Distance transform based weight map
    + [ ] Shape aware weight map
* Hyper-parameter tunning
  - [ ] Learning rate
  - [ ] Input size
  - [ ] Confidence level threshold
  - [ ] Evaluate performance of mean and std of channels  
* Data augmentation
  - [x] Random crop
  - [x] Random horizontal and vertical flip
  - [x] Random aspect resize
  - [x] Random color adjustment
  - [x] Random color invert
  - [x] Random elastic distortion
  - [x] Contrast limited adaptive histogram equalization
  - [x] Random rotate
* Public dataset extension
  - [ ] ... 
* Pre-process 
  - [x] Input normalization
  - [x] Binarize label
  - [x] Cross-validation split
  - [x] Verify training data whether png masks aligned with cvs mask. 
  - [x] Blacklist mechanism to filter noisy label(s)
  - [x] Annotate edge as soft label, hint model less aggressive on nuclei edge
  - [x] Whitelist configure option of sub-category(s) for training / validation 
  - Prediction datafeed (aka. arbitrary size of image prediction)
    + [x] Resize and regrowth 
    + [x] Origin image size with border padding (black/white constant color)
    + [x] Origin image size with border padding (replicate border color)
    + [ ] Tile-based with overlap
  - [ ] Convert input data to CIELAB color space instead of RGB
  - [ ] Use [color map algorithm](https://stackoverflow.com/questions/42863543/applying-the-4-color-theorem-to-list-of-neighbor-polygons-stocked-in-a-graph-arr) to generate ground truth of limited label (4-), in order to prevent cross-talking 
* Post-process
  - [x] Segmentation group by scipy watershed algorithm
  - [x] Segmentation group by scipy watershed algorithm with contour-based markers
  - [ ] Segmentation group by scipy watershed algorithm with model predicting markers
  - [ ] Fill hole inside each segment group
  - [ ] ...
* Computation performance
  - [x] CPU
  - [x] GPU 
  - [x] Multiple subprocess workers (IPC) 
  - [x] Cache images
  - [ ] Redundant extra contour loop in dataset / preprocess (~ 50% time cost)
  - [ ] Parallel CPU/GPU pipeline, queue or double buffer
* Statistics and error analysis
  - [x] Mini-batch time cost (IO and compute)
  - [x] Mini-batch loss
  - [x] Mini-batch IOU
  - [x] Visualize prediction result
  - [x] Visualize log summary in TensorBoard
  - [x] Running length output
  - [x] Graph visualization
  - [x] Enhance preduction color map to distinguish color of different nucleis
  - [x] Visualize overlapping of original and prediction nucleis
  - [x] Statistics of per channel data distribution, particular toward alpha
 

## Setup development environment

* Install Python 3.6 (conda recommanded)
* Install [PyTorch](http://pytorch.org/)
    ```
    conda install pytorch torchvision -c pytorch
    ```

* Install dependency python packages
    ```
    $ conda install --file requirements.txt
    ```

## Prepare data

* [Download](https://www.kaggle.com/c/data-science-bowl-2018) and uncompress to `data` folder as below structure,

```
.
├── README.md
├── data
    ├── stage1_test
    │   ├── 0114f484a16c152baa2d82fdd43740880a762c93f436c8988ac461c5c9dbe7d5
    │   └── ...
    └── stage1_train
        ├── 00071198d059ba7f5914a526d124d28e6d010c92466da21d4a04cd5413362552
        └── ...
```

* (Optional) prepare V4 dataset
    - Download [V2](https://drive.google.com/open?id=1UyIxGrVzzo7IUXRJnDRpqT3C_rXOe1s1) and uncompress to `data` folder
    - Download [TCGA no overlap](https://drive.google.com/open?id=1YB_jnDfLpZhnIj0b3wRLiiDrCtb9zNxo) and uncompress to `data` folder
    - Split TCGA to proper scale and prefered data distribution
    ```
    $ cd data
    $ python3 ../split.py external_TCGA_train --step 200 --width 256
    $ mv external_TCGA_train_split/* stage1_train/
    ```

* (Optional) prepare V6 dataset
    - Git clone [lopuhin Github](https://github.com/lopuhin/kaggle-dsbowl-2018-dataset-fixes) and move `stage1_train` in `data` folder
    - Download [TCGA no overlap](https://drive.google.com/open?id=1YB_jnDfLpZhnIj0b3wRLiiDrCtb9zNxo) and uncompress to `data` folder
    - Split TCGA to proper scale and prefered data distribution
    ```
    $ cd data
    $ python3 ../split.py external_TCGA_train --step 200 --width 256
    $ mv external_TCGA_train_split/* stage1_train/
    ```

## Hyper-parameter tunning and dataset filter

* Create a ` config.ini ` file to overwrite any setting in config_default.ini
    ```
    [param]
    weight_map = True
    model = caunet
    category = Flouresence

    [contour]
    detect = True

    [valid]
    pred_orig_size = True
    ```

* Configure whitelist of sub-category, eg. ` Flouresence ` in ` config.ini `
    - Open [Test](https://docs.google.com/spreadsheets/d/11Ykxp7uW763WXvhNoK_LvhnDV7Q6yT7Hr0f-vDAd-mo/edit?usp=drive_web&ouid=111494078798053745646) and [Train](https://docs.google.com/spreadsheets/d/1Yw-x8T4p2oChaWlLem1yyE7qN6XU3tnedj6rCCqLg4M/edit#gid=537769059) Google sheet 
    - Download CSV format (File > Download as > Common-separated values)
    - Code would check column ` discard ` and ` category ` in csv file(s)
    - Rename and copy to `data` folder as below
        ```
        .
        ├── README.md
        ├── data
            ├── stage1_test.csv <--- rename to this
            ├── stage1_test
            │   ├── 0114f484a16c152baa2d82fdd43740880a762c93f436c8988ac461c5c9dbe7d5
            │   └── ...
            ├── stage1_train.csv <--- rename to this
            └── stage1_train
                ├── 00071198d059ba7f5914a526d124d28e6d010c92466da21d4a04cd5413362552
                └── ...
        ```

## Command line usage

* Train model
    ```
    $ python3 train.py
        usage: train.py [-h] [--resume] [--no-resume] [--epoch EPOCH] [--lr LEARN_RATE]

        Grand new training ...
        Training started...
        // [epoch #][step #] CPU second (io second)     Avg.  batch  (epoch)    Avg. batch (epoch)
        Epoch: [1][0/67]    Time: 0.928 (io: 0.374)	    Loss: 0.6101 (0.6101)   IoU: 0.000 (0.000)	
        Epoch: [1][10/67]   Time: 0.140 (io: 0.051)	    Loss: 0.4851 (0.5816)   IoU: 0.000 (0.000)
        ...
        Epoch: [10][60/67]  Time: 0.039 (io: 0.002)	    Loss: 0.1767 (0.1219)   IoU: 0.265 (0.296)
        Training finished...
        ...
    
    // automatically save checkpoint every 10 epochs
    $ ls checkpoint
        current.json   ckpt-10.pkl
    ```

* Evaluate on test dataset, will show side-by-side images on screen. Specify ` --save ` to save as files
    ```
    $ python3 valid.py
    ```

* Evaluate on train dataset with ground truth, will show side-by-side images & IoU on screen. Specify ` --save ` to save as files
    ```
    $ python3 valid.py --dataset train
    ```

* Generate running length encoding of test dataset
    ```
    $ python3 valid.py --csv
    ```

## Jupyter Notebook running inside Docker container

* ssh to docker host, say ` myhost `
* Launch notebook container, expose port ` 8888 `
    ```
    $ cd ~/Code/DSB2018
    $ docker run --runtime=nvidia -it --rm --shm-size 8G -v $PWD:/mnt -w /mnt -p 8888:8888 rainbean/pytorch
        ...
        Copy/paste this URL into your browser when you connect for the first time,
        to login with a token:
            http://localhost:8888/?token=8dae8f258f5c127feff1b9b6735a7cd651c6ce6f1246263d
    ```

* Open browser to url, remember to change hostname from ` localhost ` to real hostname/ip ` myhost `
* Create new notebook tab (New > Python3)
* Train model
    ```Jupyter Notebook
    In [1]: %run train.py --epoch 10

    Loading checkpoint './checkpoint/ckpt-350.pkl'
    Training started...
    Epoch: [350][59/270]	Time: 0.206 (io: 0.085)		Loss: 0.5290 (0.5996)	IoU(Semantic): 0.344 (0.263)
    ```

* Evaluate side-by-side prediction in notebook cell
    ```Jupyter Notebook
    In [2]: %matplotlib inline
    In [3]: %run valid.py
    ```

* Generate csv result
    ```Jupyter Notebook
    In [4]: %run valid.py --csv
    ```

## Benchmark 

| Score | Data | Model | Cost Fn. | Epoch | Marker | Watershed | TP | Learn Rate | CV | Width | PO | Crop | Flip | Invert | Jitter | Distortion | Clahe | Edge Soft Label | 
| ----- | ---- | -----  | --------- | ---- | ------- | - | -- | ----------- | --- | --- | - | - | - | - | - | - | - | - |
| 0.334 | Orig | UNet   | BCE       | 600  |         |   | .5 | 1e-4 > 3e-5 | 10% | 256 |   | V | V |   | V |   |   |   |
| 0.344 | Orig | UNet   | IOU+BCE   | 600  |         |   | .5 | 1e-4 > 3e-5 | 10% | 256 |   | V | V |   | V | V |   |   |
| (TBA) | Orig | UNet   | IOU+BCE   | 600  |         |   | .5 | 1e-4 > 3e-5 |  0% | 256 |   | V | V |   | V | V |   |   |
| 0.326 | v2   | UNet   | IOU+BCE   | 600  |         |   | .5 | 1e-4 > 3e-5 |  0% | 256 |   |   |   |   |   |   |   |   |
| 0.348 | v2   | UNet   | IOU+BCE   | 300  |         |   | .5 | 1e-4        | 10% | 256 |   | V | V |   | V | V |   |   |
| 0.361 | v2   | UNet   | IOU+BCE   | 600  |         |   | .5 | 1e-4 > 3e-5 |  0% | 256 |   | V | V |   | V | V |   |   |
| 0.355 | v2   | UNet   | IOU+BCE   | 600  |         |   | .5 | 1e-4 > 3e-5 |  0% | 256 |   | V | V |   | V | V | V |   |
| 0.350 | v2   | UNet   | IOU+BCE   | 1200 |         |   | .5 | 1e-4 > 3e-6 |  0% | 512 |   | V | V |   | V | V |   |   |
| 0.353 | v2   | UNet   | IOU+BCE   | 600  |         |   | .5 | 1e-4 > 3e-5 |  0% | 256 |   | V | V |   | V | V |   | V |
| 0.413 | v2   | UNet   | IOU+BCE   | 600  | cluster | V | .5 | 1e-4 > 3e-5 |  0% | 256 |   | V | V |   | V | V |   |   |
| 0.421 | v3   | UNet   | IOU+BCE   | 400  | cluster | V | .5 | 1e-4 > 3e-5 |  0% | 256 |   | V | V |   | V | V |   |   |
| 0.437 | v3   | UNet   | IOU+BCE   | 900  | cluster | V | .5 | 1e-4        |  0% | 256 |   | V | V |   | V | V |   |   |
| 0.460 | v4   | CAUNet | IOU+WBCE  | 900  | contour | V | .5 | 1e-4        |  0% | 256 |   | V | V |   | V | V |   |   |
| 0.447 | v4   | CAUNet | IOU+BCE   | 1800 | contour | V | .5 | 1e-4        |  0% | 256 |   | V | V |   | V | V |   |   |
| 0.465 | v4   | CAUNet | IOU+WBCE  | 1800 | contour | V | .5 | 1e-4        |  0% | 256 |   | V | V |   | V | V |   |   |
| 0.459 | v5   | CAUNet | IOU+WBCE  | 1200 | contour | V | .5 | 1e-4        |  0% | 256 |   | V | V |   | V | V |   |   |
| 0.369 | v5   | CAUNet | IOU+WBCE  | 1200 | contour | V | .8 | 1e-4        |  0% | 256 |   | V | V |   | V | V |   |   |
| 0.477 | v5   | CAUNet | IOU+WBCE  | 1200 | contour | V | .3 | 1e-4        |  0% | 256 |   | V | V |   | V | V |   |   |
| 0.464 | v6   | CAUNet | IOU+WBCE  | 1800 | contour | V | .5 | 1e-4        |  0% | 256 |   | V | V |   | V | V |   |   |
| 0.476 | v6   | CAUNet | IOU+WBCE  | 1800 | contour | V | .3 | 1e-4        |  0% | 256 |   | V | V |   | V | V |   |   |
| 0.457 | v6   | CAUNet | IOU+WBCE  | 1800 | contour | V | .2 | 1e-4        |  0% | 256 |   | V | V |   | V | V |   |   |
| 0.473 | v6   | CAUNet | IOU+WBCE  | 1800 | contour | V | .35| 1e-4        |  0% | 256 |   | V | V |   | V | V |   |   |
| 0.467 | v6   | CAUNet | IOU+WBCE  | 1800 | contour | V | .3 | 1e-4        |  0% | 256 | V | V | V |   | V | V |   |   |
| 0.465 | v6   | CAUNet | IOU+WBCE  | 5000 | contour | V | .5 | 1e-4        |  0% | 256 |   | V | V |   | V | V |   |   |
| 0.480 | v6   | CAUNet | IOU+WBCE  | 5000 | contour | V | .3 | 1e-4        |  0% | 256 |   | V | V |   | V | V |   |   |
| 0.461 | v6   | CAUNet | IOU+WBCE  | 5000 | contour | V | .3 | 1e-4        |  0% | 256 | V | V | V |   | V | V |   |   |
| 0.458 | v6   | CAUNet | IOU+WBCE  | 5000 | contour | V | .5 | 1e-4        |  0% | 256 | V | V | V |   | V | V |   |   |
| 0.435 | Orig | CAUNet | IOU+WBCE  | 1800 | contour | V | .5 | 1e-4        |  0% | 256 |   | V | V |   | V | V |   |   |

<!---
# regression to be addressed, March 15.
| 0.427 | v6   | CAUNet | IOU+Focal | 600  | contour | V | .5 | 1e-4        |  0% | 256 | V | V | V |   | V | V |   |   |
| 0.429 | v6   | CAUNet | IOU+Focal | 600  | contour | V | .3 | 1e-4        |  0% | 256 | V | V | V |   | V | V |   |   |
| 0.424 | v6   | CAUNet | IOU+Focal | 600  | contour | V | .5 | 1e-4        |  0% | 256 |   | V | V |   | V | V |   |   |
| 0.360 | v6   | CAUNet | IOU+Focal | 600  | contour | V | .3 | 1e-4        |  0% | 256 |   | V | V |   | V | V |   |   |
| 0.480 | v6   | CAUNet | IOU+WBCE  | 5000 | contour | V | .3 | 1e-4        |  0% | 256 |   | V | V |   | V | V |   |   |
--->

Note:
- Dataset (training): 
    * V1: original kaggle
    * V2: Feb 06, modified by Jimmy and Ryk
    * V3: V2 + TCGA 256
    * V4: V2 + TCGA 256 (Non overlapped)
    * V5: V2 + TCGA 256 (Non overlapped) + Feb. labeled test set
    * V6: [lopuhin Github](https://github.com/lopuhin/kaggle-dsbowl-2018-dataset-fixes) + TCGA 256 (Non overlapped)
- Score is public score on kaggle site
- Zero CV rate means all data were used for training, none reserved
- Adjust learning rate per 300 epoch
- Cost Function:
    * BCE: pixel wise binary cross entropy
    * WBCE: pixel wise binary cross entropy with weight
    * IOU: pixel wise IoU, no instance weight
- TP: threshold of prediction probability 
- PO (predict origin size): true to keep original test image in prediction phase, otherwise resize as training width

## Known Issues

* Error: multiprocessing.managers.RemoteError: AttributeError: Can't get attribute 'PngImageFile'  
    ```
    Reproduce rate: 
        1/10  
    Root cause: 
        PyTorch subprocess workers failed to communicate shared memory.  
    Workaround: 
        Ignore and issue command again
    ```

* Train freeze or runtime exception running in docker container
    ```
    Root cause: 
        PyTorch subprocess worker require enough shared memory for IPC communication.  
    Fix: 
        Assign --shm-size 8G to reserve enough shared memory
    Example: 
        $ docker run --runtime=nvidia -it --rm --shm-size 8G -v $PWD:/mnt -w /mnt rainbean/pytorch python3 train.py
    ```

## Transform effect demo

* Random elastic distortion  

    ![elastic_distortion](docs/elastic_distortion.jpeg) 

* Random color invert  

    ![color_invert](docs/color_invert.jpeg) 

* Random color jitter  

    ![color_jitter](docs/color_jitter.jpeg) 

* Clahe color equalize

    ![color_equalize](docs/clahe_color_adapthist_equalize.jpeg) 

* Image border padding: constant vs replicate. Used in origin size prediction.

    ![color_equalize](docs/image_border_padding.jpg) 

## Learning curve

* Comparison of cost functions 

    ![learn_curve](docs/learn_curve.jpg)  

