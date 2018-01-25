# configure hyperparameters
mean = [0.5, 0.5, 0.5, 0.5] # per RGBA channels
std  = [0.5, 0.5, 0.5, 0.5]
n_epoch = 10
n_batch = 10
n_worker = 4
print_freq = 10
learn_rate = 0.001
cv_ratio = 0.1
width = 128 # model input size
cuda = True