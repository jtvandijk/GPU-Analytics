---
title: "Preface"
weight: 1
# bookFlatSection: false
# bookToc: true
# bookHidden: false
# bookCollapseSection: false
# bookComments: false
# bookSearchExclude: false
---

# GPU-based analysis for social and geographic applications

## Introduction 
This document provides the tools, the code and utility of the graphics processing unit (GPU) for social and geographic research and applications.This project is funded by The British Academy's [Talent Development Award](https://www.thebritishacademy.ac.uk/funding/talent-development-awards/). 

In most cases data science tasks are executed on local servers or personal devices where calculations are handled by one or more Central Processing Units (CPUs). CPUs can only handle one task at the time, meaning that the computational time for millions of sequential operations can only be sped up by adding more CPUs and parallelising these calculations. A Graphics Processing Units (GPU), on the other hand, is designed to execute several tasks simultaneously. GPU-accelerated analytics harness the parallel-capabilities of the GPU to accelerate processing-intensive operations. This can be particularly useful for social and geographic research where the data sets are often large and complex. 

In this document, we will explore a collection of novel and innovative computational methods in social and geographic data science. This is essential for at least two reasons. First, with the proliferation of large-scale data sets as well as the availability of increasingly powerful personal computing devices over the past decade or so, social science and geography have witnessed a second quantitative revolution - [Kitchen (2014, p.3)](https://doi.org/10.1177/2053951714528481) speaks of a fourth paradigm of science with a focus on data-intensive analyses, statistical exploration and data mining. However, there is limited availability of well-documented resources for both social and geographical data science researchers and students in handling large volumes of data in an efficient manner.

Second, whereas ‘Big Data’ can be very rich sources of information, they tend to be accidental (e.g. a by-product of online transactions) and highly diverse in quality and resolution. As a result, many ‘Big Data’ sources are not representative of the population or phenomenon of study and contain a variety of biases and uncertainties. An illustrative example of the problem is described in Van [Dijk et al. (2021)](https://doi.org/10.1111/rssa.12713) where, in the absence of frequently updated data on the nature of residential moves in the United Kingdom, the authors use population registers and administrative data to develop robust annual estimates of residential mobility between all UK neighbourhoods by ascribing individuals that seemingly vacate a property to their most probably residential destination. With circa 7.8 billion possible origin-destination pairings, this was a very time-consuming and computationally intensive model.

This raises the question of how technological innovations can be harnessed for the benefit of social and geographic data science research: both to enable future highly computationally intensive research as well as how to effectively communicate these new research pipelines to researchers and students within the domains of computational social sciences and quantitative geography. 

## Who is this document for? 

In this document, we will explicitly explore the potential of GPU with **three case studies**, and best practices for GPU in social and geographic research. This document is aimed at researchers and students in the domains of computational social sciences and quantitative geography who are interested in learning and using GPU as part of their research. It is also aimed at data science and machine learning practitioners who are interested in working with geographical data and problems in a more efficient manner. While some content is aimed at a more technical audience, the document is written in a way that is accessible to a wide audience. 

We also want to highlight that this document is not a comprehensive guide to GPU and GPU programming. Rather, it is a collection of resources and projects that can be used to explore the potential of GPU-based analysis in social and geographic research. 

## Document details

In this project we will first explain the basics of GPU and how to configure GPU both locally and on the cloud. In this project, we will be using a NVDIA Tesla V100 GPU for our demonstrations, but the code and instructions can be easily adapted to other GPUs. 

After configuration, the usability of GPU-accelerated analytics in the context of three different geographic applications will be benchmarked through three case studies. The first of these case studies will explore the matching and joining of large textual datasets comprising millions of addresses. The second case study will explore the performance of deep learning algorithms for GeoAI applications. The third case study will look specifically at the opportunities for improving the performance of disease risk prediction using raster data. The final deliverable of the project will be a workshop alongside a freely available, shareable digital resource with detailed instructions on how to implement GPU-based analysis in social and geographic research and teaching.

## Project members
- [Dr Justin van Dijk](www.mappingdutchman.com)
- [Dr Stephen Law](https://www.turing.ac.uk/people/researchers/stephen-law)
- [Dr Anwar Musah](https://www.geog.ucl.ac.uk/people/academic-staff/anwar-musah)
- [Mr Jason Chi Sing Tang](https://www.ucl.ac.uk/geospatial-analytics/people/jason-chi-sing-tang)

## License
This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

## Project reference: **TDA21\210069**.  