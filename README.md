# ltudl
Blaze &amp; Inferno - La Trobe University's Deep Learning System

Most of the active development on Blaze and Inferno is conducted via a private repositiory. However, we as we will subsequently go public after the Hadoop Summit in Melbourne this year, you can expect updates at an increasing frequency over the next couple of months.

For the moment we are just releasing the code of Blaze and CUBlaze here. With these two components you will be able to build and train complex neural networks on the GPU and CPU of your local machine. In their functionality, Blaze & CUBlaze are not disimilar to Torch. However, of course Blaze has its own quite distinct flavor. We define deep learning not only as a training task that starts at layer 1 of your neural network. Blaze defines primitives that cover the entire processing pipeline. From loading data from the HDD until scoring the model.

Core of my research right now is the Inferno optimization engine. Inferno can parallelize the training of Blaze models efficiently on a Spark cluster. It can cope with mixed hardware setups and and works reasonably well in setups where the network bandwith is limited.

I am currently writing a paper about the Inferno optimizer about the expeirences, observations and tricks that we use to make Inferno work well. Once that paper has been accepted, I will make the entire source code available here as well. I running lots of experiments to fine-tune Inferno on our research cluster right now and a good proportion of the paper is already written. So it shouldn't take to long until that happens ;-). The best way to prepare for Inferno is to get familliar with Blaze.



# Compiling

```
git clone https://github.com/bashimao/ltudl.git
cd ltudl/scripts
./build-demos.sh
```

This will download all dependencies, compile the code and place the artefacts in the subdirectory ltudl/out. 


# Prequirements

Blaze requires Oracle Java 7 or better to run. OpenJDK seems to work as well. But performance is worse. Using Oracle Java is, therefore, strongly recommended.


## Checking the configuration.

Once you got a JVM up and have loaded the Blaze.jar files for the first time, you can check the current configuration by running the command: `RuntimeStatus.collect()`

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


