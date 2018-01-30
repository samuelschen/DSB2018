# configure hyperparameters
mean = [0.5, 0.5, 0.5, 0.5] # per RGBA channels
std  = [0.5, 0.5, 0.5, 0.5]
n_epoch = 10
n_batch = 10
n_worker = 4
print_freq = 10
learn_rate = 0.001
cv_ratio = 0.1
cv_seed = 666 # change it if different shuffle cv required 
width = 128 # model input size
cuda = True
threshold = 0.5 # possibility gating threshold
model_name = 'unet' # a name for log description