# GAN Cubism Art
Generation of Cubism Art using Keras

 Run using Spell platform and V100 GPU

Logs:

  Spell run python art_gan.py -t V100 -m uploads/art_gan/cubism_data.npy
✨ Preparing uncommitted changes…
Enumerating objects: 5, done.
Counting objects: 100% (5/5), done.
Delta compression using up to 4 threads
Compressing objects: 100% (3/3), done.
Writing objects: 100% (3/3), 433 bytes | 433.00 KiB/s, done.
Total 3 (delta 1), reused 0 (delta 0)
💫 Casting spell #6…
✨ Stop viewing logs with ^C
🌟 Machine_Requested… Run created -- waiting for a V100 machine.
⭐ Machine_Requested… Run created -- waiting for a V100 machine.
🌟 Machine_Requested… Run created -- waiting for a V100 machine.
⭐ Machine_Requested… Run created -- waiting for a V100 machine.
✨ Machine_Requested… done
✨ Building… done
✨ Mounting… done
✨ Run is running
Using TensorFlow backend.
2020-09-27 15:59:20.519089: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-09-27 15:59:20.551068: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2200000000 Hz
2020-09-27 15:59:20.553340: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ae0c9fa240 executing computations on platform Host. Devices:
2020-09-27 15:59:20.553373: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): <undefined>, <undefined>
2020-09-27 15:59:20.560612: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcuda.so.1
2020-09-27 15:59:20.821423: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1005] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-09-27 15:59:20.822401: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ae0e502760 executing computations on platform CUDA. Devices:
2020-09-27 15:59:20.822438: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): Tesla V100-SXM2-16GB, Compute Capability 7.0
2020-09-27 15:59:20.822934: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1005] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-09-27 15:59:20.823679: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1640] Found device 0 with properties: 
name: Tesla V100-SXM2-16GB major: 7 minor: 0 memoryClockRate(GHz): 1.53
pciBusID: 0000:00:04.0
2020-09-27 15:59:20.830089: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcudart.so.10.0
2020-09-27 15:59:20.908838: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcublas.so.10.0
2020-09-27 15:59:20.949046: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcufft.so.10.0
2020-09-27 15:59:20.963610: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcurand.so.10.0
2020-09-27 15:59:21.064055: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcusolver.so.10.0
2020-09-27 15:59:21.121868: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcusparse.so.10.0
2020-09-27 15:59:21.295569: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcudnn.so.7
2020-09-27 15:59:21.295806: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1005] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-09-27 15:59:21.296856: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1005] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-09-27 15:59:21.297580: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1763] Adding visible gpu devices: 0
2020-09-27 15:59:21.298918: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcudart.so.10.0
2020-09-27 15:59:21.301131: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1181] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-09-27 15:59:21.301162: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1187]      0 
2020-09-27 15:59:21.301169: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1200] 0:   N 
2020-09-27 15:59:21.301329: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1005] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-09-27 15:59:21.302149: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1005] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-09-27 15:59:21.302916: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1326] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 15060 MB memory) -> physical GPU (device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:04.0, compute capability: 7.0)
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_2 (Dense)              (None, 4096)              413696    
_________________________________________________________________
reshape_1 (Reshape)          (None, 4, 4, 256)         0         
_________________________________________________________________
up_sampling2d_1 (UpSampling2 (None, 8, 8, 256)         0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 8, 8, 256)         590080    
_________________________________________________________________
batch_normalization_5 (Batch (None, 8, 8, 256)         1024      
_________________________________________________________________
activation_1 (Activation)    (None, 8, 8, 256)         0         
_________________________________________________________________
up_sampling2d_2 (UpSampling2 (None, 16, 16, 256)       0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 16, 16, 256)       590080    
_________________________________________________________________
batch_normalization_6 (Batch (None, 16, 16, 256)       1024      
_________________________________________________________________
activation_2 (Activation)    (None, 16, 16, 256)       0         
_________________________________________________________________
up_sampling2d_3 (UpSampling2 (None, 32, 32, 256)       0         
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 32, 32, 256)       590080    
_________________________________________________________________
batch_normalization_7 (Batch (None, 32, 32, 256)       1024      
_________________________________________________________________
activation_3 (Activation)    (None, 32, 32, 256)       0         
_________________________________________________________________
up_sampling2d_4 (UpSampling2 (None, 64, 64, 256)       0         
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 64, 64, 256)       590080    
_________________________________________________________________
batch_normalization_8 (Batch (None, 64, 64, 256)       1024      
_________________________________________________________________
activation_4 (Activation)    (None, 64, 64, 256)       0         
_________________________________________________________________
up_sampling2d_5 (UpSampling2 (None, 128, 128, 256)     0         
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 128, 128, 256)     590080    
_________________________________________________________________
batch_normalization_9 (Batch (None, 128, 128, 256)     1024      
_________________________________________________________________
activation_5 (Activation)    (None, 128, 128, 256)     0         
=================================================================
Total params: 3,369,216
Trainable params: 3,366,656
Non-trainable params: 2,560
_________________________________________________________________
WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

2020-09-27 15:59:26.306444: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcublas.so.10.0
2020-09-27 15:59:26.950416: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcudnn.so.7
0 epoch, Discriminator accuracy: 54.16666865348816, Generator accuracy: 33.33333432674408
100 epoch, Discriminator accuracy: 8.33333358168602, Generator accuracy: 100.0
200 epoch, Discriminator accuracy: 25.0, Generator accuracy: 100.0
300 epoch, Discriminator accuracy: 4.16666679084301, Generator accuracy: 100.0
400 epoch, Discriminator accuracy: 8.33333358168602, Generator accuracy: 100.0
500 epoch, Discriminator accuracy: 8.33333358168602, Generator accuracy: 100.0
600 epoch, Discriminator accuracy: 16.66666716337204, Generator accuracy: 100.0
700 epoch, Discriminator accuracy: 12.5, Generator accuracy: 100.0
800 epoch, Discriminator accuracy: 12.5, Generator accuracy: 100.0
900 epoch, Discriminator accuracy: 12.5, Generator accuracy: 100.0
1000 epoch, Discriminator accuracy: 12.5, Generator accuracy: 100.0
1100 epoch, Discriminator accuracy: 12.5, Generator accuracy: 100.0
1200 epoch, Discriminator accuracy: 0.0, Generator accuracy: 100.0
1300 epoch, Discriminator accuracy: 16.66666716337204, Generator accuracy: 100.0
1400 epoch, Discriminator accuracy: 4.16666679084301, Generator accuracy: 100.0
1500 epoch, Discriminator accuracy: 12.5, Generator accuracy: 100.0
1600 epoch, Discriminator accuracy: 20.83333283662796, Generator accuracy: 100.0
1700 epoch, Discriminator accuracy: 8.33333358168602, Generator accuracy: 100.0
1800 epoch, Discriminator accuracy: 12.5, Generator accuracy: 100.0
1900 epoch, Discriminator accuracy: 4.16666679084301, Generator accuracy: 100.0
2000 epoch, Discriminator accuracy: 12.5, Generator accuracy: 100.0
2100 epoch, Discriminator accuracy: 29.16666567325592, Generator accuracy: 100.0
2200 epoch, Discriminator accuracy: 8.33333358168602, Generator accuracy: 100.0
2300 epoch, Discriminator accuracy: 4.16666679084301, Generator accuracy: 100.0
2400 epoch, Discriminator accuracy: 20.83333283662796, Generator accuracy: 100.0
2500 epoch, Discriminator accuracy: 12.5, Generator accuracy: 100.0
2600 epoch, Discriminator accuracy: 12.5, Generator accuracy: 100.0
2700 epoch, Discriminator accuracy: 12.5, Generator accuracy: 100.0
2800 epoch, Discriminator accuracy: 20.83333283662796, Generator accuracy: 100.0
2900 epoch, Discriminator accuracy: 16.66666716337204, Generator accuracy: 100.0
3000 epoch, Discriminator accuracy: 29.16666567325592, Generator accuracy: 100.0
3100 epoch, Discriminator accuracy: 8.33333358168602, Generator accuracy: 100.0
3200 epoch, Discriminator accuracy: 16.66666716337204, Generator accuracy: 100.0
3300 epoch, Discriminator accuracy: 8.33333358168602, Generator accuracy: 100.0
3400 epoch, Discriminator accuracy: 4.16666679084301, Generator accuracy: 100.0
3500 epoch, Discriminator accuracy: 12.5, Generator accuracy: 100.0
3600 epoch, Discriminator accuracy: 16.66666716337204, Generator accuracy: 100.0
3700 epoch, Discriminator accuracy: 4.16666679084301, Generator accuracy: 100.0
3800 epoch, Discriminator accuracy: 25.0, Generator accuracy: 100.0
3900 epoch, Discriminator accuracy: 20.83333283662796, Generator accuracy: 100.0
4000 epoch, Discriminator accuracy: 12.5, Generator accuracy: 100.0
4100 epoch, Discriminator accuracy: 12.5, Generator accuracy: 100.0
4200 epoch, Discriminator accuracy: 16.66666716337204, Generator accuracy: 100.0
4300 epoch, Discriminator accuracy: 12.5, Generator accuracy: 100.0
4400 epoch, Discriminator accuracy: 20.83333283662796, Generator accuracy: 100.0
4500 epoch, Discriminator accuracy: 8.33333358168602, Generator accuracy: 100.0
4600 epoch, Discriminator accuracy: 8.33333358168602, Generator accuracy: 100.0

4700 epoch, Discriminator accuracy: 20.83333283662796, Generator accuracy: 100.0
4800 epoch, Discriminator accuracy: 16.66666716337204, Generator accuracy: 100.0
4900 epoch, Discriminator accuracy: 8.33333358168602, Generator accuracy: 100.0
5000 epoch, Discriminator accuracy: 12.5, Generator accuracy: 100.0
5100 epoch, Discriminator accuracy: 16.66666716337204, Generator accuracy: 100.0
5200 epoch, Discriminator accuracy: 12.5, Generator accuracy: 100.0
5300 epoch, Discriminator accuracy: 12.5, Generator accuracy: 100.0
5400 epoch, Discriminator accuracy: 12.5, Generator accuracy: 100.0
5500 epoch, Discriminator accuracy: 25.0, Generator accuracy: 100.0
5600 epoch, Discriminator accuracy: 4.16666679084301, Generator accuracy: 100.0
5700 epoch, Discriminator accuracy: 12.5, Generator accuracy: 100.0
5800 epoch, Discriminator accuracy: 4.16666679084301, Generator accuracy: 100.0
5900 epoch, Discriminator accuracy: 4.16666679084301, Generator accuracy: 100.0
6000 epoch, Discriminator accuracy: 4.16666679084301, Generator accuracy: 100.0
6100 epoch, Discriminator accuracy: 12.5, Generator accuracy: 100.0
6200 epoch, Discriminator accuracy: 8.33333358168602, Generator accuracy: 100.0
6300 epoch, Discriminator accuracy: 12.5, Generator accuracy: 100.0
6400 epoch, Discriminator accuracy: 12.5, Generator accuracy: 100.0
6500 epoch, Discriminator accuracy: 4.16666679084301, Generator accuracy: 100.0
6600 epoch, Discriminator accuracy: 4.16666679084301, Generator accuracy: 100.0
6700 epoch, Discriminator accuracy: 16.66666716337204, Generator accuracy: 100.0
6800 epoch, Discriminator accuracy: 12.5, Generator accuracy: 100.0
6900 epoch, Discriminator accuracy: 8.33333358168602, Generator accuracy: 100.0
7000 epoch, Discriminator accuracy: 12.5, Generator accuracy: 100.0
7100 epoch, Discriminator accuracy: 12.5, Generator accuracy: 100.0
7200 epoch, Discriminator accuracy: 8.33333358168602, Generator accuracy: 100.0
7300 epoch, Discriminator accuracy: 12.5, Generator accuracy: 100.0
7400 epoch, Discriminator accuracy: 16.66666716337204, Generator accuracy: 100.0
7500 epoch, Discriminator accuracy: 16.66666716337204, Generator accuracy: 100.0
7600 epoch, Discriminator accuracy: 16.66666716337204, Generator accuracy: 100.0
7700 epoch, Discriminator accuracy: 12.5, Generator accuracy: 100.0
7800 epoch, Discriminator accuracy: 8.33333358168602, Generator accuracy: 100.0
7900 epoch, Discriminator accuracy: 20.83333283662796, Generator accuracy: 100.0
8000 epoch, Discriminator accuracy: 8.33333358168602, Generator accuracy: 100.0
8100 epoch, Discriminator accuracy: 16.66666716337204, Generator accuracy: 100.0
8200 epoch, Discriminator accuracy: 8.33333358168602, Generator accuracy: 100.0
8300 epoch, Discriminator accuracy: 4.16666679084301, Generator accuracy: 100.0
8400 epoch, Discriminator accuracy: 16.66666716337204, Generator accuracy: 100.0
8500 epoch, Discriminator accuracy: 12.5, Generator accuracy: 100.0
8600 epoch, Discriminator accuracy: 0.0, Generator accuracy: 100.0
8700 epoch, Discriminator accuracy: 4.16666679084301, Generator accuracy: 100.0
8800 epoch, Discriminator accuracy: 29.16666567325592, Generator accuracy: 100.0
8900 epoch, Discriminator accuracy: 20.83333283662796, Generator accuracy: 100.0
9000 epoch, Discriminator accuracy: 8.33333358168602, Generator accuracy: 100.0
9100 epoch, Discriminator accuracy: 8.33333358168602, Generator accuracy: 100.0
9200 epoch, Discriminator accuracy: 16.66666716337204, Generator accuracy: 100.0
9300 epoch, Discriminator accuracy: 8.33333358168602, Generator accuracy: 100.0
9400 epoch, Discriminator accuracy: 12.5, Generator accuracy: 100.0
9500 epoch, Discriminator accuracy: 12.5, Generator accuracy: 100.0
9600 epoch, Discriminator accuracy: 8.33333358168602, Generator accuracy: 100.0
9700 epoch, Discriminator accuracy: 16.66666716337204, Generator accuracy: 100.0
9800 epoch, Discriminator accuracy: 8.33333358168602, Generator accuracy: 100.0
9900 epoch, Discriminator accuracy: 12.5, Generator accuracy: 100.0
✨ Saving… done
✨ Pushing… done
🎉 Total run time: 15m30.559878s
🎉 Run 6 complete
