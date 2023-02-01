---
title: "GPU"
weight: 1
# bookFlatSection: false
# bookToc: true
# bookHidden: false
# bookCollapseSection: false
# bookComments: false
# bookSearchExclude: false
---

## Introduction
Over the past few decades, social science research has faced challenges due to a lack of available data, the cost and time needed to conduct surveys, and limitations on computational power for analysis. These issues have been exacerbated by a decline in survey quality as well as an increase in biases in the characteristics of respondents. There was also no guarantee that long-term, time-consuming surveys would continue to be available due to fiscal austerity ([Singleton *et al.* , 2017](https://uk.sagepub.com/en-gb/eur/urban-analytics/book249267)). However, in recent years, the availability of big data has greatly expanded the resources available to social scientists, allowing them to access a wide range of information about individuals and societies from sources such as social media, mobile devices, and online transactions. The rapid expansion of data is transforming the practice of social science, providing increasingly granular spatial and behavioural profiles about the society and individuals. The difference is not just a matter of scale.

The increasing data availability extends beyond the realm of identifying consumer preferences, allowing us to measure social processes and the dynamics of spatial trends in unprecedented detail. For example, [Trasberg and Cheshire (2021)](https://uk.sagepub.com/en-gb/eur/urban-analytics/book249267) used a large scale mobility data from mobile applications to explore the activity patterns in London during lockdown, identifying the socio-spatial fragmentation between urban communities. Similarly, [Van Dijk *et al.* (2020)]((https://doi.org/10.1111/rssa.12713)) used an annually updated Consumer Register (LCR) to estimate residential moves and the socio-spatial characteristics of the sedentary population in the UK, overcoming the limitations of the traditional census data. [Gebru *et al.* (2020)](https://doi.org/10.1073/pnas.1700035114), on the other hand, used a large scale dataset of images from Google Street View to estimate the demographic makeup of a neighbourhood. Their results suggested a possibility of using automated systems for monitoring demographics may effectively complement labor-intensive approaches, with the potential to measure demographics with fine spatial resolution, in close to real time.

With the growing availability of data, processing and analysing large datasets need more computational power than is currently available, with more complex algorithms that need more compute power to run. To overcome this challenge, researchers have turned to a GPU (Graphics Processing Unit) to accelerate massive data parallelism and computation. Therefore, in this chapter, we will introduce the concept of GPU and explain why GPU can be a useful tool for social scientists. In addition, we will provide a brief introduction to the CUDA and RAPIDS suite, which is a GPU-accelerated framework that can be used to accelerate the data analysis process.

## CPU vs GPU
A CPU is a general-purpose processor that is capable of executing a wide range of tasks, including running operating systems, executing programs, and performing calculations. CPUs are typically designed with a focus on sequential processing, meaning they are optimised for executing instructions one at a time and in a specific order. They typically have a relatively small number of processing cores (often ranging from 2 to 32) and are capable of performing a wide range of functions. A GPU, on the other hand, is a specialised processor that is designed specifically for handling graphics and visualisations. GPUs have numerous processing cores (usually hundreds or thousands) and are optimised for parallel processing, meaning they can perform many calculations simultaneously. This makes GPUs particularly well-suited for tasks that require a lot of processing power, such as rendering 3D graphics, running machine learning algorithms, or performing scientific simulations [(NVIDIA, 2020)](https://developer.nvidia.com/blog/cuda-refresher-reviewing-the-origins-of-gpu-computing).

The main difference between a CPU and a GPU is the its architecture (see **Figure 1**). GPUs dedicate most of their transistors for ALU units (Arithmetic Logic Unit) which are responsible for performing arithmetic and logical operations. CPUs, on the other hand reserve most of their transistors for caches and control units, which aim to reduce latency within each thread. Most CPUs are multi-core processors, meaning they have multiple processing cores that can execute instructions with multiple data streams. This architecture is called Multiple Instruction, Multiple Data (MIMD). This architecture is designed to minimise the time it takes to access data (ref: white bars in **Figure 2**). In a single time slice, a CPU thread tries to get as much work done as possible (ref: green bar in **Figure 2**). To achieve this, CPUs require low latency, which is achieved through large caches and complex control logic. However, caches work best with only a few threads per core, as switching between threads is expensive [(NVIDIA, 2020)](https://developer.nvidia.com/blog/cuda-refresher-reviewing-the-origins-of-gpu-computing).

GPUs hide instruction and memory latency with computation. GPUs use a Single Instruction, Multiple Data (SIMD) architecture, where each thread is assigned a small amount of memory (blue bar), resulting a much higher latency per thread. However, GPUs have many threads per core, and it can switch from one thread to another at no cost, resulting higher throughput and bandwidth for large data. What this means, in the end, is that we can store more data in the GPU memory and caches, which can be reused for matrix multiplication and operations that are more computationally intensive [(NVIDIA, 2020)](https://developer.nvidia.com/blog/cuda-refresher-reviewing-the-origins-of-gpu-computing).

As shown in **Figure 2**, when thread *T1* is waiting for data, another thread *T2* begins processing, and so on with *T3* and *T4*. In the meantime, *T1* eventually gets the data it needs to process. In this way, latency is hidden by switching to other available work. As a result, GPUs can utilise overlapping concurrent threads to hide latency and are able to run thousands of threads at once.  The best CPUs have about 50GB/s while the best GPUs have 750GB/s memory bandwidth. So the larger your computational operations are in terms of memory, the larger the advantage of GPUs over CPUs.

We can make the minions' analogy to explain the difference between CPU and GPU. A CPU is like Gru who is considerably intelligent and capable of building fantastic machines, but he can only do one thing at a time. A GPU is like a swarm of minions who are not as intelligent as Gru, but they can do build one thing collectively.

<figure title = "CPU vs GPU 1">
    <center>
     <p><img src="https://github.com/jasoncpit/GPU-Analytics/blob/master/Pictures/GPU_CPU.png?raw=true">
    <figcaption>
    <b>Figure 1: CPU versus GPU architecture. A CPU devotes more transistors to control the data flow, while GPUs devote more transistors to data processing [SOURCE].
    </b>
    </figcaption>
    <center>
</figure>

<figure title = "CPU vs GPU 2">
    <center>
     <p><img src="https://github.com/jasoncpit/GPU-Analytics/blob/master/Pictures/GPU_CPU_process.png?raw=true">
    <figcaption>
    <b>Figure 2: CPU versus GPU processor architecture <a href="https://developer.nvidia.com/blog/cuda-refresher-reviewing-the-origins-of-gpu-computing">(NVIDIA, 2020)</a>.
    </b>
    </figcaption>
    <center>
</figure>

## Are GPUs better than CPUs?
It is not accurate to say that one is always better than the other: both CPUs and GPUs have their own strengths and are optimised for different types of tasks - not all tasks can be accelerated by a GPU. The first thing to consider whether to use a CPU or a GPU for your analysis is the scale of the data. A good GPU can read/write its memory much faster than a CPU can read/write its memory. GPUs can perform calculations much faster and more efficient to process massive datasets than the host CPU. An example of matrix multiplication executed on a CPU and a GPU is shown in **Figure 3**: the CPU can be faster when the matrix size is small, but the GPU is much faster when the matrix size is large. This also applies to other data science tasks such as training a neural network. You may get away without a GPU if your neural network is small in scale. However, it might be worthwhile to consider investing in a GPU if the neural network involves hundreds of thousands of parameters and requires extensive training times.

<figure title = "CPU vs GPU speed comparion">
    <center>
     <p><img src="https://github.com/jasoncpit/GPU-Analytics/blob/master/Pictures/CPU_GPU_speed.png?raw=true">
    <figcaption>
    <b>Figure 3: Computational speed between CPU and GPU <a href="https://www.mathworks.com/help/parallel-computing/measuring-gpu-performance.html">(MathWorks, 2022)</a>.
    </b>
    </figcaption>
    <center>
</figure>

A second thing to consider when deciding between architecture relates to the type of project. If you are working on a project that requires a lot of parallel processing, such as rendering large spatial objects, running machine learning algorithms, or performing scientific simulations, a GPU is a good choice. However, if you are working on a project that requires a high level of sequential processing, such as running multiple functions in a loop, a CPU is a good choice that offers more flexibility. The third thing to keep in mind is general accessibility. Access to GPUs is not as easy as CPUs. GPUs are usually more expensive than CPUs, and they are not as widely available.

More generally, if you are working with a [Cloud GPU providers](https://thechief.io/c/editorial/comparison-cloud-gpu-providers/) such as Google Colab, Microsoft Azure, or Amazon AWS, the cost of GPU per hour would be somewhere between **$0.5** and **$3.5**. If you are working with a local GPU, the average cost is around **$450**. Furthermore, you need to make sure that your GPU is compatible with the CUDA toolkit. M1 chips, for example, are not compatible with CUDA toolkit, prohibiting some GPU computing libraries from being used.

## Why have GPUs become more popular?
Over the past few years, GPUs have become more and more popular in data science. This is mainly due to the development of the [The NVIDIA® CUDA® Toolkit](https://developer.nvidia.com/blog/cuda-refresher-the-gpu-computing-ecosystem/), which simplifies GPU-based programming. CUDA (**C**ompute **U**nified **D**evice **A**rchitecture) is a general purpose parallel computing platform and application programming interface (API) developed by NVIDIA. CUDA uses a hybrid data processing model where serial sections of code run on the CPU and parallel sections run on the GPU (see **Figure 4**).

CUDA gives us direct access to the GPU’s virtual instruction set and parallel computational elements, making it easier for compiling compute kernels into code that will execute efficiently on the GPU. In CUDA, the computation is divided into small units of work called threads. Threads are grouped into thread blocks, and multiple thread blocks can be executed simultaneously on the GPU. Thread blocks are further grouped into grids, which represent a collection of thread blocks that can be executed in parallel on the GPU.

In CUDA, the function (kernel) is executed with the aid of threads. Each thread performs the same operation on different elements of the input data, allowing the GPU to perform the computation in parallel. A block of threads is controlled by the streaming multiprocessing unit that can be scheduled and executed concurrently. Several blocks of threads can be grouped into a grid, which constitutes the whole GPU unit.

The relationship between grid, thread, and block in CUDA can be visualised as a three-dimensional structure, with each thread block representing a plane and each grid representing a collection of planes. The number of threads per block determines the width of the plane, and the number of blocks per grid determines the number of planes in the collection. The GPU schedules and executes the threads in the grid, and each thread performs its computation on a different element of the input data.

<figure title = "GPU acceleration">
     <center>
     <p><img src="https://github.com/jasoncpit/GPU-Analytics/blob/master/Pictures/GPU_software.png?raw=true">
    <figcaption>
    <b>Figure 4: How GPU acceleration works [SOURCE].
    </b>
    </figcaption>
    </center>
</figure>

The CUDA toolkit and many third-party libraries offer a collection of well-optimised and pre-compiled functions that enable drop-in acceleration across multiple domains such as linear algebra, image and video processing, deep learning, and graph analytics. For developing custom algorithms, you can use available integrations with commonly used languages and numerical packages, as well as well-published development API operations. Some widely used libraries are:

- Mathematical libraries: cuBLAS, cuRAND, cuFFT, cuSPARSE, cuTENSOR, cuSOLVER
- Parallel algorithm libraries: nvGRAPH, Thrust
- Image and video libraries: nvJPEG, NPP, Optical Flow SDK
- Communication libraries: NVSHMEM, NCCL
- Deep learning libraries: cuDNN, TensorRT, Riva, DALI
- Partner libraries: OpenCV, FFmpeg, ArrayFire, MAGMA  

As a social scientist, you may not need to use all of these libraries. However, it is important to know that these libraries exist and are available for you to use. Libraries such as [RAPIDS](https://rapids.ai/), [Numba](https://numba.readthedocs.io/en/stable/user/5minguide.html) or [PyTorch](https://pytorch.org/) offer a high level programming interface that allows you to use GPU computing without having to learn C/C++ or CUDA. Furthermore, library such as RAPIDS *cuDf* and *CuPy* provide a drop-in replacement for *Pandas* and *Numpy* allowing you to use GPU computing for tasks such as data cleaning, feature engineering, and machine learning without having to change much of your code. Some [libraries that fall under the RAPIDS banner](https://docs.rapids.ai/api) with their CPU-based counterparts are:

- cuDF (Pandas)
- cuML (Sklearn)
- cuGraph (NetworkX)
- cuSpatial (Geopandas/Shapely)
- CuPy (Numpy)

It is important to note that CUDA is not the only parallel computing platform. With the rise of GPU computing, companies such as AMD and Intel have developed their own computing framework. [OpenCL](https://www.khronos.org/opencl/), for instance, is another parallel computing platform and programming model that is supported by AMD and Intel. However, at the moment CUDA is the most popular parallel computing platform and it is the only one that is supported by NVIDIA.
