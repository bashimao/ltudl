# Blaze &amp; Inferno - La Trobe University's Deep Learning System

Most of the active development on Blaze and Inferno is conducted via a private repositiory. However, as we will subsequently go public after the Hadoop Summit in Melbourne this year, you can expect updates at an increasing frequency over the next couple of months.

For the moment we only release the code of Blaze and CUBlaze here. With these two components you are able to build and train complex neural networks on the GPU and CPU of your local machine. In their functionality, Blaze & CUBlaze are not disimilar to Torch or CAFFE. However, of course Blaze has its own distinct flavor. We consider deep learning not only as a ML task that starts at layer 1 of your neural network. Blaze defines primitives that cover the entire processing pipeline. From loading data from hard disk until scoring the model.

Core of my research right now is the Inferno optimization engine. Inferno can parallelize the training of Blaze models efficiently on a Spark cluster. It can cope with mixed hardware setups and and works reasonably well in setups where the network bandwith is limited.

I am currently writing a paper about the Inferno optimizer and the experiences, observations and tricks that we used to make it work well. Once that paper has been accepted, I will make the entire source code available here as well. I am running lots of experiments to fine-tune Inferno on our research cluster right now and a good proportion of the paper is already written. So it shouldn't take to long until I can make verything publicly available ;-). However, until then... The best way to prepare for Inferno is to get familliar with Blaze.

TODO: Add Presentation slides from Hadoop Summit 2016


# Compiling

```
git clone https://github.com/bashimao/ltudl.git
cd ltudl/scripts
./build-demos.sh
```

This will download all dependencies and compile the code. The ImageNet demo and its artefacts can be found in the subdirectory `ltudl/out`.

TODO: Expand compile instructions


# Prequirements

Blaze requires Oracle Java 7 or better to run. OpenJDK seems to work as well. But performance is worse. Using Oracle Java is, therefore, strongly recommended.


## Checking the configuration.

Once you got a JVM up and have loaded the blaze/cublaze jar-files for the first time, you can check the current configuration by running the command: `RuntimeStatus.collect()`

`RuntimeStatus.collect()` returns a Json-Object that you can print in a human readable form as follows:
```
val rs = RuntimeStatus.collect()
StreamEx.writeJson(System.out, rs, pretty = true)
```

## Image Library Support

Computer vision is one of the primary applications of deep learning. By default, Blaze will use AWT to load images. If you're using Oracle Java you should experience a decent image processing performance out of the box. However, especially when working with ImageNet, problems with non-standard images can arise. This includes random crashes due to out-of-memory issues. For performance and compatibility reasons we suggest using OpenCV.

### To make Blaze use OpenCV:
1. Make sure you have OpenCV installed.
```
sudo apt-get install libopencv-dev
```
2. Set the following environment variable.
```
export LTU_IO_IMAGE_DEFAULT_IMPLEMENTATION=OpenCV
```

If issues with loading images still persist, please check the wiki. There you will find a blacklist of images from ImageNet2014, that have known issues. If removing those Images from your dataset also doesn't solve your problem, something is going terribly wrong. (Please post an error report, if possible!)


## CPU Acceleration

Blaze is linked against netlib-java, which in turn will select native BLAS implementations to speed up linear algebra operations. To figure out whether your desired BLAS implementation has been loaded check the following fields in the runtime status output:

```
blasClassName
lapackClassName
LIBBLAS.alternatives
LIBBLAS3.alternatives
LIBLAPACK.alternatives
LIBLAPACK3.alternatives

```

In case that `blasClassName` or `lapackClassName` does not contain the expression native-system, netlib-java was unable to locate your BLAS installation. In that case, make sure that your load library path contains the a path to your BLAS implementation. If you use MKL, you may have to manually add the directory to your `/etc/ld.so.conf`.

```
sudo echo >> /etc/ld.so.conf
sudo echo /<path to MKL>/lib >> /etc/ld.so.conf
sudo echo /<path to MKL>/lib64 >> /etc/ld.so.conf
sudo ldconfig
```

Once MKL is in your load library path, that does not necessarily mean that netlib-java will be able to pick it up. Make sure that you have a native BLAS wrapper installed and register MKL as the primary implementation alternative.
```
sudo apt-get install libblas3
update-alternatives --install /usr/lib/libblas.so     libblas.so     <path to MKL>/lib64/libmkl_rt.so
update-alternatives --install /usr/lib/libblas.so.3   libblas.so.3   <path to MKL>/lib64/libmkl_rt.so
update-alternatives --install /usr/lib/liblapack.so   liblapack.so   <path to MKL>/lib64/libmkl_rt.so
update-alternatives --install /usr/lib/liblapack.so.3 liblapack.so.3 <path to MKL>/lib64/libmkl_rt.so
update-alternatives --config libblas.so
update-alternatives --config libblas.so.3
update-alternatives --config liblapack.so
update-alternatives --config liblapack.so.3

```


## GPU Acceleration

If you want to use CUBlaze make sure that the NVDIA drivers, CUDA and cuDNN are installed in your system and visible in the load library path. For Ubuntu this can be achieved as follows.

### 1. Install NVIDIA Drivers
```
sudo apt-add-repository ppa:graphics-drivers
sudo apt-get update
sudo apt-get install nvidia-352
```

### 2. Install CUDA Toolkit 7.5
```
wget http://developer.download.nvidia.com/compute/cuda/repos/<your OS>/cuda-repo-<your OS>_7.5-18_amd64.deb
Example: wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.5-18_amd64.deb
sudo dpkg -i cuda-repo-<your OS>_7.5-18_amd64.deb
sudo apt-get update
sudo apt-get install cuda
```

### 3. Install cuDNN 5.0
Go to https://developer.nvidia.com/cudnn, register and download cuDNN 5.0.
Extract the files into a directory that you feel comfortable with and make sure `/etc/ld.so.conf` includes that path. Don't forget to call `sudo ldconfig` to make your changes to `/etc/ld.so.conf` visible.

To test whether everything CUBlaze works, just call `CUBlaze.register()` and blaze.RuntimeStatus().collect(). CUBLaze specific settings should now show up there.


# Demos


## MNIST

Before you can run these demos you will have to obtain the MNIST dataset. This dataset is available from here [http://yann.lecun.com/exdb/mnist]. Just download and uncompress the 4 files linked at the top of the website.

By default the demos will look for the dataset at `<working directory>/data`. However, you can override the location by setting the environment variable EXPERIMENT_DATA_PATH.


### Simple MLP

This is the hello world of neural networks. The network is simple mult-layer perceptron. We will create and train a very small network. While doing so we perform online cross validation. While not mandatory it is nice to have. After the training is complete we score it once against the entire test set.

```
export EXPERIMENT_DATA_PATH=<where you've extracted the MNIST files>
./start-app.sh edu.latrobe.demos.mnist.SimpleMLP
```


### Simple ConvNet

Very similar to the multi-level perceptron demo. But this time we use a small convolutional neural network. Using of GPUs is strongly recommended. If your have not yet been able to get your GPU work
along with CUBlaze so far, set EXPERIMENT_FORCE_CUDA="no".

```
export EXPERIMENT_DATA_PATH=<where you've extracted the MNIST files>
./start-app.sh edu.latrobe.demos.mnist.SimpleConvNet
```


# Running ImageNet Demo

This is a full featured demo that trains a 1000 class classifier on ImageNet. It is a little bit rough around the edges. And as soon as I have the time I will improve to provide you with a better experience. However, with some effort you can surely make it work. Feel free to ask if you get stuck.

We have provided a demo that can use various models to train ImageNet with live cross validation. You may select the model you can adjust the environment variable `EXPERIMENT_MODEL_NAME`. Possible values are:
* "ResNet-18"
* "ResNet-18-PreAct"
* "ResNet-34"
* "ResNet-34-PreAct"
* "ResNet-50"
* "ResNet-50-PreAct"
* "ResNet-101"
* "ResNet-101-PreAct"
* "ResNet-152"
* "ResNet-152-PreAct"
* "AlexNet"
* "AlexNet-OWT"
* "AlexNet-OWT-BN"
* "VGG-A"

For the larger ResNets you may want to reduce the batch size using the environment variable `EXPERIMENT_TRAINING_BATCH_SIZE`. 

Preparing the dataset.
To get started you need to obtain the imagenet dataset first.On the ImageNet website you'll want to download the CLSLOC dataset from 2014. Furthermore, you need to grab the following package ("imagenet-clsloc-meta.tar.xz" TODO: Add Link). Once obtained, you want to extract all files into the same subdirectory.

We expect the files arranges as follows:
```
/.../CLSLOC/
           /bbox_train_aggregated
                                 /synset....xml
           /bbox_val_aggregated.xml
           /mean-and-variance-100k.json 
           /mean-and-variance-10k.json
           /meta_clsloc.csv
           /test-extract-no-faulty
                                  /...
                                  /01
                                     /...
                                     /ILSVRC2012_test_....JPEG
                                     /...
                                  /...
           /valid-extract-no-faulty
                                   /...
                                   /01
                                      /...
                                      /ILSVRC2012_val_....JPEG
                                      /...
                                   /...
          
	  /train-extract-no-faulty
                                  /...
                                  /n0...
                                        /n0...._....JPEG
                                  /...
```

We keep the no-faulty in the directory name to indicate that we are running on a slighly modified dataset. But feel free to step into the source code and remove it. We add this postfix actually remove a bunch of files that have known issues with Java AWT. If you intend to use Java AWT, go to the wiki [TODO: Add Link], grab the ImageNet faulty images list, and remove them from your dataset as well.

The program will expect the above mentioned directory structuree at `$HOME/$EXPERIMENT_RELATIVE_DATA_PATH`. So you can adjust the location by modifying the environment variable `EXPERIMENT_RELATIVE_DATA_PATH`.

However, all this has been setup, you are ready to roll.

TODO: Add Instructions how to start demo.




