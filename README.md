# GPU-based analysis for social and geographic applications

## About
This GitHub repository generates the GPU-Analytics website, as part of the *Exploring GPU-based analysis for social and geographic applications* project. This project is funded by The British Academy's [Talent Development Award](https://www.thebritishacademy.ac.uk/funding/talent-development-awards/). Project reference: **TDA21\210069**.

## Project context
This project will explore novel and innovative computational methods in social and geographic data science. This is essential for at least two reasons. First, with the proliferation of large-scale data sets as well as the availability of increasingly powerful personal computing devices over the past decade or so, social science and geography have witnessed a second quantitative revolution - [Kitchen (2014, p.3)](https://doi.org/10.1177/2053951714528481) speaks of a fourth paradigm of science with a focus on data-intensive analyses, statistical exploration and data mining. However, there is limited availability of well-documented resources for both social and geographical data science researchers and students in handling large volumes of data in an efficient manner.

Second, whereas ‘Big Data’ can be very rich sources of information, they tend to be accidental (e.g. a by-product of online transactions) and highly diverse in quality and resolution. As a result, many ‘Big Data’ sources are not representative of the population or phenomenon of study and contain a variety of biases and uncertainties. An illustrative example of the problem is described in Van [Dijk et al. (2021)](https://doi.org/10.1111/rssa.12713) where, in the absence of frequently updated data on the nature of residential moves in the United Kingdom, the authors use population registers and administrative data to develop robust annual estimates of residential mobility between all UK neighbourhoods by ascribing individuals that seemingly vacate a property to their most probably residential destination. With circa 7.8 billion possible origin-destination pairings, this was a very time-consuming and computationally intensive model.

This raises the question how technological innovations can be harnessed for the benefit of social and geographic data science research: both to enable future highly computationally intensive research as well as how to effectively communicate these new research pipelines to researchers and students within the domains of computational social sciences and quantitative geography.

## Project details
The project is guided by the aim to explore the feasibility to extend computationally intensive social science and quantitative geography tasks with GPU-accelerated analyses. In most cases data science tasks are executed on local servers or personal devices where calculations are handled by one or more Central Processing Units (CPUs). CPUs can only handle one task at the time, meaning that the computational time for millions of sequential operations can only be sped up by adding more CPUs and parallelising these calculations. A Graphics Processing Units (GPU), on the other hand, is designed to execute several tasks simultaneously. GPU-accelerated analytics harness the parallel-capabilities of the GPU to accelerate processing-intensive operations.

In this project we will first configure a NVIDIA Tesla V100 GPU within a local computer cluster. After configuration, the usability of GPU-accelerated analytics in the context of three different geographic applications will be investigated through three case studies. The first of these case studies will explore the matching and joining of large textual datasets comprising millions of addresses. The second case study will explore the performance of deep learning algorithms for GeoAI applications. The third case study will look specifically at the opportunities for improving the performance of disease risk prediction using raster data. The final deliverable of the project will be a workshop alongside a freely available, shareable digital resource with detailed instructions on how to implement GPU-based analysis in social and geographic research and teaching.

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
