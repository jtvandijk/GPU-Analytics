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
## What is a GPU?

[GPU](https://www.zhihu.com/question/319355296)

- GPU (Graphics Processing Unit) is a specialized electronic circuit designed to rapidly manipulate and alter memory to accelerate the creation of images in a frame buffer intended for output to a display device. 
- GPUs are used in embedded systems, mobile phones, personal computers, workstations, and game consoles. 
- Modern GPUs are very efficient at manipulating computer graphics and image processing, and their highly parallel structure makes them more effective than general-purpose CPUs for algorithms where processing of large blocks of data is done in parallel. 

GPU vs CPU 
- CPU is like a brain, which has a few strong cores, while GPU is like a soul, which has many weak cores.
- GPU has a large number of cores, which are small processors that can run in parallel.
- GPU has a large amount of memory, which is used to store data and instructions.
- GPU has a high bandwidth, which is used to transfer data between the GPU and the CPU.
- GPU has a high clock speed, which is used to execute instructions.
- GPU has a high floating-point performance, which is used to perform mathematical calculations.

CUDA 
- First released in 2007, the parallel computing platform lets coders take advantage of the computing power of GPUs for general purpose processing by inserting a few simple commands into their code.

## Is GPU better than CPU?
Yes and no 

**Function** 
- CPU is better at sequential tasks, optimized for single-threaded applications.

- We need to understand the bottleneck of GPU and CPU. 
- GPU is better than CPU when we have a lot of data and we need to do a lot of calculations.
- CPU is better when we are doing a lot of sequential tasks on a small amount of data.

## How can we use GPU for social and geographical analysis? 
- The advent of consumer data has led to the development of new methods for social and geographical analysis.
- 5V (Volume, Velocity, Variety, Veracity, and Value) 
- It requires modern computing infrastructure to process and analyze the data.
- GPU is a good choice for social and geographical analysis.

Examples of social and geographical analysis
- Social network analysis
- Geospatial analysis
- Text mining
- Image processing
- Machine learning

## What are the tools available for using GPU?

- RAPIDS
- TensorFlow
- PyTorch
- Caffe

## Choosing the right GPU for your analysis

GPU can be expensive. So, it is important to choose the right GPU for your analysis. 
**Local GPU**
- Not all GPU has the same performance. 
- Not all GPU can be intergrated with the CUDA toolkit.
- For example M1 chips are not compatible with CUDA toolkit.

**Cloud GPU**
- Cloud GPU is a good choice for GPU computing.
- Cloud GPU is cheaper than local GPU.
- Providers: Google Cloud, Amazon Web Services, Microsoft Azure, IBM Cloud, Oracle Cloud, Alibaba Cloud, Tencent Cloud, etc.


