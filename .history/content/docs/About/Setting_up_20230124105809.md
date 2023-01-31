---
title: "Setting up the environment for GPU"
weight: 1
# bookFlatSection: false
# bookToc: true
# bookHidden: false
# bookCollapseSection: false
# bookComments: false
# bookSearchExclude: false
---

## Introduction 
Setting up the environment for GPU can be challenging, as every computer has different hardware and software configurations. There are no universal instructions that will work for everyone, but in this chapter, we will discuss how to set up the environment for GPU on the Server and Google Colab. We will also highlight the steps to verify that the GPU is working properly. 


## Setting up the environment for Google Colab GPU 

Google Colab is a cloud-based platform that allows you to run Jupyter notebooks in the cloud, with support GPUs-acceleration. Google Colab is free to use, and you do not need to install any software or setup CUDA manually. It comes with already pre-installed libraries such as pandas or NumPy, so you do not need to run “pip install” by yourself. Here are the steps to set up a GPU for use in Colab:

### How to Start? 

The first step is to create a new notebook in Google Colab. You can do this by going to the [Google Colab website](https://colab.research.google.com/), signing in with your Google account, and creating a new notebook.  

<figure title = "test">
     <center>
     <p><img src="https://github.com/jasoncpit/GPU-Analytics/blob/master/Pictures/Colab_create.png?raw=true">
    <figcaption>
    <b>Figure 1: Create a new notebook in Google Colab. 
    </b> 
    </figcaption>
    </center>
</figure>


### Enabling GPU support in Google Colab 

After creating a new notebook, you need to enable GPU support. To do this, go to the "Runtime" menu and select "Change runtime type." Then, set the "Hardware accelerator" to "GPU." 


<figure title = "test">
     <center>
     <p><img src="https://github.com/jasoncpit/GPU-Analytics/blob/master/Pictures/Colab_run_time.png?raw=true">
    <figcaption>
    <b>Figure 2: Select "Change runtime type" in the "Runtime" menu.
    </b> 
    </figcaption>
    </center>
</figure>

After you change your runtime type, your notebook will restart, which means information from the previous session will be lost. 


<figure title = "test">
     <center>
     <p><img src="https://github.com/jasoncpit/GPU-Analytics/blob/master/Pictures/Colab_setting.png?raw=true">
    <figcaption>
    <b>Figure 3: Select "GPU" in the "Hardware accelerator" menu. 
    </b> 
    </figcaption>
    </center>
</figure>

You may also see the TPU option in the drop-down menu. TPU stands for a Tensor Processing Unit, which is a specialized chip designed to accelerate machine learning workloads. TPU is more powerful than GPU, but it is also more expensive.

### Verify GPU access 

After you enable GPU support, you can verify that the GPU is working properly by running the following code in a code block in your notebook. **nvidia-smi** (NVIDIA System Management Interface) is a tool to query, monitor and configure NVIDIA GPUs. It ships with and is installed along with the NVIDIA driver and it is tied to that specific driver version. It is a tool written using the NVIDIA Management Library (NVML), which you can also find the Python bindings in the PyPI package pynvml. 

```Python
!nvidia-smi
```

<figure title = "test">
     <center>
     <p><img src="https://github.com/jasoncpit/GPU-Analytics/blob/master/Pictures/Colab_smi.png?raw=true">
    <figcaption>
    <b>Figure 4: NVIDIA System Management Interface (nvidia-smi) output.
    </b> 
    </figcaption>
    </center>
</figure>

This outputs a summary table, where I find the following information useful:
- **Name**: The name of the GPU.
- **Memory**: The amount of memory available on the GPU.
- **Utilization**: The percentage of time the GPU is being used.
- **Power usage**: The power usage of the GPU.
- **Processes**: List of processes executing on the GPUs.

In this example, we got a Tesla T4 GPU with 16GB of memory. The GPU is currently being used by the notebook, but we can also see that there is no other process running on the GPU.


There are different ways to verify that the GPU is working properly. For example, you can run the following code to check find out the specification of the GPU.

```Python
import torch
use_cuda = torch.cuda.is_available() #Check if GPU is available
if use_cuda:
    print('__CUDNN VERSION:', torch.backends.cudnn.version()) #Which version of cudnn is being used
    print('__Number CUDA Devices:', torch.cuda.device_count()) #How many GPUs are available
    print('__CUDA Device Name:',torch.cuda.get_device_name(0)) #Get the name of the GPU
    print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9) # Get the total amount of memory available on the GPU
```

### Best practices for using Google Colab 

Google Colab is a great tool for running GPU-accelerated notebooks in the cloud. However, there are some limitations that you should be aware of.

- **Notebooks are not persistent**: When you create a notebook in Google Colab, it is stored in the cloud. However, the notebook is not persistent. This means that if you close the notebook, the notebook will be deleted. If you want to save your notebook, you need to download it to your local machine.
- **Google Colab is not free**: Google Colab is free to use, but they limit the amount of time you can use the GPU. Google Colab will automatically disconnects the notebook if we leave it idle for more than 30 minutes, meaning that your training will be interrupted and you will have to restart it. Nevertheless, you can run the free of charge notebooks for at most 12 hours at a time as explained in the [FAQ](https://research.google.com/colaboratory/faq.html#gpu-time-limit). If you want to use the GPU for a longer period of time and access to more memory, there is a paid version of Colab that you can [use](https://colab.research.google.com/signup/pricing?utm_source=resource_tab&utm_medium=link&utm_campaign=want_more_resources). 

- **Working with data**: You can access your files that are stored in Google Drive from Google Colab. To do this, you need to mount your Google Drive to the notebook. You can do this by running the following code in a code block in your notebook. Use the batch command ```!cd``` the “Files” panel on the left to access your files. 

```Python
from google.colab import drive
drive.mount('/content/drive')
```

You can also upload or download a file (or files) from/to your computer using the following code. 

```Python
from google.colab import files
#Upload a file from your computer
files.upload()
#Download a file to your computer
files.download('path/to/your/file')
```

- **Ensure all files have been completely copied to your Gdrive**: As mentioned above, Colab can and will terminate your session due to inactivity. To ensure that your final model and data is saved to your Gdrive, call ```drive.flush_and_unmount()``` before you terminate your session. Furthermore, you can checkpoint your model during training process and save it to your Gdrive.

```Python
from google.colab import drive
model.fit(...) # Training your model
model.save('path/to/your/model') # Save your model
drive.flush_and_unmount()
print('All changes made in this colab session should now be visible in Drive.')
```
Note that the completion of copying/writing files to /content/drive/MyDrive/ does not mean that all files are safely on GDrive and that you can immediately terminate your Colab instance because the transfer of data between the VM and Google’s infrastructure happens asynchronously, so performing this flushing will help ensure it’s indeed safe to disconnect.



- **Disconnecting from the notebook**: Be a responsible user and disconnect from the notebook when you are finished with your GPU. Google will notice and reward you with a better GPU the next time you request one! So after you’ve copied your saved model files to GDrive or Google Storage, you can disconnect from the notebook by clicking the “Disconnect” button in the “Runtime” menu. Or you can run the following code in a code block in your notebook. 

```Python
from google.colab import runtime
runtime.unassign()
```




## Setting up the environment on the server

At UCL, Department of Geography, we have a dedicated server that we can use to run GPU-accelerated notebooks, installed with the Tesla V100 GPU. In theory, you should be able to connect to the server from anywhere in the world and run your GPU analyses.

Setting up the environment on the server is a bit more complicated than setting up the environment on your local machine. However, it is not that difficult. The main difference is that you need to install the GPU-accelerated libraries from source. This is because the GPU-accelerated libraries are not available on the Anaconda package manager. 

### SSH into the server
you can log into the server using the following bash command. 
```bash
ssh username@server_ip port 
```
After this command, you will be prompted to enter your password.

### Installing the libraries you will need
You can initialize a new environment using the following bash command. 

```bash
conda create -n gpu -c rapidsai -c nvidia -c conda-forge rapids=22.06 python=3.9 cudatoolkit=11.5 jupyterlab 
```
This will create a new conda environment called “gpu” and install all the libraries you need and the CUDA tookit.  

### Activate the environment
You can activate the environment using the following bash command. 
```bash
conda activate gpu
```

### Start the JupyterLab server
You can start the JupyterLab server using the following bash command. 
```bash
jupyter lab --no-browser 
```
This will start the JupyterLab server and you will be able to access it from your browser. 


## Setting up the environment on your local machine

Setting up the environment on your local machine varies depending on your operating system. I will not go into details about how to install the CUDA toolkit on your local machine. Instead, I will refer you to the official documentation for installing CUDA on [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/).
### Hardware and Software requirements

Most of these GPU-accelerated libraries build on top of the CUDA framework. Therefore, you need to have a GPU that supports CUDA. You can check the [CUDA GPU support matrix](https://developer.nvidia.com/cuda-gpus) to see if your GPU is supported. But of course, you should have a decent CPU, RAM and Storage to be able to do some leverage the power of the GPU. 

For the minimum hardware requirements, I recommend the following: 
- **CPU**: Minimum 4 cores and at 2.6GHz, at least 16GB of RAM.
- **GPU**: Nvidia GPU with at least 6GB of VRAM.
