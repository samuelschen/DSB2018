# Description

Kaggle 2018 Data Science Bowl: find the nuclei in divergent images to advance medical discovery

## Development Plan:

* Platform and framework
  - [x] PyTorch
  - [x] macOS
  - [x] Ubuntu
* Explore model architecture
  - [x] UNet
  - [x] Dropout
  - [x] Batch normalization
  - [ ] Transfer learning
  - [ ] Cost function
  - [ ] Score function
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
  - [ ] Random rotate
* Public dataset extension
  - [ ] ... 
* Pre and post-process 
  - [x] Input normalization
  - [x] Binarize label
  - [x] Cross-validation split
  - [ ] Use [color map algorithm](https://stackoverflow.com/questions/42863543/applying-the-4-color-theorem-to-list-of-neighbor-polygons-stocked-in-a-graph-arr) to generate ground truth of limited label (4-), in order to prevent cross-talking 
  - [ ] Verify training data whether png masks aligned with cvs mask. 
  - [ ] Blacklist mechanism to filter noisy label(s)
  - [ ] Arbitrary image size handling, eg. brige & up-sample size alignment
* Computation performance
  - [x] CPU
  - [x] GPU 
  - [x] Multiple subprocess workers (IPC) 
  - [x] Pretech / cache images
* Statistics and error analysis
  - [x] Mini-batch time cost (IO and compute)
  - [x] Mini-batch loss
  - [x] Mini-batch IOU
  - [x] Visualize prediction result
  - [x] Visualize log summary in TensorBoard
  - [x] Running length output
  - [ ] Graph visualization
  - [ ] Enhance preduction color map to distinguish color of different nucleis
  - [ ] Visualize overlapping of original and prediction nucleis
  - [ ] Statistics of per channel data distribution, particular toward alpha
 

## Setup development environment

* Install Python 3.5 and pip
* Install [PyTorch](http://pytorch.org/)
    ```
    // macOS
    $ pip3 install http://download.pytorch.org/whl/torch-0.3.0.post4-cp35-cp35m-macosx_10_6_x86_64.whl 
    $ pip3 install torchvision 
    // Ubuntu
    $ pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl 
    $ pip3 install torchvision
    ```

* Install dependency python packages
    ```
    $ pip3 install -r requirements.txt
    ```

## Prepare data

[Download](https://www.kaggle.com/c/data-science-bowl-2018) and uncompress to `data` folder as below structure,

```
.
├── README.md
├── config.py
├── data
    ├── noisy_label.txt
    ├── stage1_test
    │   ├── 0114f484a16c152baa2d82fdd43740880a762c93f436c8988ac461c5c9dbe7d5
    │   └── ...
    └── stage1_train
        ├── 00071198d059ba7f5914a526d124d28e6d010c92466da21d4a04cd5413362552
        └── ...
```

## Command line usage

* Train model
    ```
    $ python3 train.py
        usage: train.py [-h] [--resume] [--no-resume] [--cuda] [--no-cuda] [--epoch EPOCH]

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

* Evaluate test dataset, will show side-by-side images on screen
    ```
    $ python3 valid.py
    ```

* Generate running length encoding of test dataset
    ```
    $ python3 valid.py --csv
    ```

## Known Issues

* Error: multiprocessing.managers.RemoteError: AttributeError: Can't get attribute 'PngImageFile'  
    ```
    Reproduce rate: 1/10  
    Root cause: PyTorch subprocess workers failed to communicate shared memory.  
    Workaround: Ignore and issue command again
    ```

## Transform effect demo

* Random elastic distortion  

    ![elastic_distortion](../../../docs/img/tranform/elastic_distortion.jpeg) 

* Random color invert  

    ![color_invert](../../../docs/img/tranform/color_invert.jpeg) 

* Random color jitter  

    ![color_jitter](../../../docs/img/tranform/color_jitter.jpeg) 

* 