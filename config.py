# configure hyperparameters
n_epoch = 300
n_ckpt_epoch = 10
n_batch = 10
n_worker = 4
print_freq = 10
learn_rate = 0.0001
cv_ratio = 0.1
cv_seed = 666 # change it if different shuffle cv required 
width = 256 # model input size
cuda = True
threshold = 0.5 # possibility gating threshold
model_name = 'unet_iou_loss' # a name for log description
# data augmentation config
mean = [0.5, 0.5, 0.5] # per RGB channels
std  = [0.5, 0.5, 0.5]
label_to_binary = True
color_invert = False
color_jitter = True
elastic_distortion = True