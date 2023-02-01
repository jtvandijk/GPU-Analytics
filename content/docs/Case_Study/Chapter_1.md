---
title: "Chapter 1 - Address geocoding"
weight: 1
# bookFlatSection: false
# bookToc: true
# bookHidden: false
# bookCollapseSection: false
# bookComments: false
# bookSearchExclude: false
---

## Address geocoding  
Address geocoding, or address matching, is an example of a broad class of data science problems known as data linkage. Data linkage problems involve connecting two datasets together by establishing an association between two records based on one or more common properties. In the case of address geocoding, one would try to assign XY coordinates to a list of address strings derived from, for instance, consumer sources that are not explicitly georeferenced (see, for instance, [Lansley *et al.* 2019](https://doi.org/10.1111/rssa.12476)). The process involves linking the address list to a authoritative reference database that contains address strings with their associated coordinates. However, due to the lack of a universal standard on how addresses are structured and stored, matching addresses becomes a significantly complicated task that requires a great deal of data preparation and processing (see **Table 1**).

|Input address string| Reference data to match against|
| --- | ----------- |
|Unstructured text |Structured, tokenised|
|Messy, containing typos and abbreviations|Clean, standardised, (mostly) correct |
|Incomplete| Snapshot of addresses at a given time|
|Range from historic to very recent addresses, including businesses| Organisation / business names are not always part of the address|

**Table 1: Summary of address matching challenges [(Office for National Statistics, 2022](https://www.ons.gov.uk/methodology/methodologicalpublications/generalmethodology/onsworkingpaperseries/onsworkingpaperseriesno17usingdatasciencefortheaddressmatchingservice#:~:text=Address%20matching%20is%20an%20example,common%20property%20of%20the%20data.)).**

## Objective
Not only is address matching problem a complex problem, its complexity increases exponentially with an increase in data sizes. The objective of this first **Case Study**, therefore, is to showcase how we can utilise a GPU's processing capabilities for address matching - and data linkage more general.

## Address matching pipeline
The address matching process can be split into three high-level steps: data pre-processing, candidate address retrieval, and candidate scoring and ranking.

### Address string-preprocessing
At the most fundamental level, we need to prepare the data for the matching process. There are potentially different approach to do this, but the most common approach is to concatenate (join) the address into its constituent parts if this is not the case already. Alternatively, the input address can be split into corresponding parts, such as the street name, house number, postcode, and so on. The former approach is more common, but it ignores the information in the data about the address, and makes it impossible to rely on the natural structure of the address to help match the desired address with the input string. The latter approach is more complex, but flexible, it allows for more accurate comparison because comparing tokens precludes the possibility of the same word representing one element of the address being compared against an unrelated element of the address.

### Candidate address: retrieval
In the second step, we need a method to compare the input address with the reference data. The simplest approach is to compare each token of the input address / each address string with each token / each address string of the reference data. This approach is simple, but it cannot deal with typos and abbreviations. A common solution to deal with this is by using a probabilistic data matching algorithm, such as fuzzy string matching with similarity measures to accommodate typos and misspellings: to make sure that, for instance, "Birmingam" would match "Birmingham".  However, in practice, when each record is compared against every other record using some type of similarity measure (e.g. the [Levenshtein distance](https://doi.org/10.1016/j.cosrev.2020.100300)), the number of comparisons can be very large, thus leading to a very expensive computation that cannot effectively be deployed on a GPU. Instead, we can consider using a Natural Language Processing (NLP) approach to compare the addresses by assuming that the address is a sequence of unstructured text. One of the most common approaches is to convert the address into a vector and then compare the vector similarity (e.g., [TF-IDF](http://www.tfidf.com/)) to select potentially matching candidates.

### Candidate address: scoring and ranking
The last step in the process is to evaluate the quality of the match between the input address string and the retrieved candidate addresses. The most common approach is to use a similarity score to evaluate the similarity among all potential candidate addresses. Depending on the application, the similarity score can be a simple percentage or a more complex score. For instance, we can define a threshold for the similarity score, and only return the candidate addresses that have a similarity score above the threshold. We can also evaluate the model performance by validating the results with a ground truth dataset.

## GPU Considerations
The address matching problem is a computationally intensive problem. A core challenge is to understand which part of the process is the most computationally intensive and which part of the process can be efficiently parallelised on a GPU. In the first part, we can concatenate the address from its constituent parts, whereby treating each address as a sequence of unstructured text. In the second part, we can use a [character-based n-gram TF-IDF](https://medium.com/in-pursuit-of-artificial-intelligence/brief-introduction-to-n-gram-and-tf-idf-tokenization-e58d22555bab) to convert the address into a vector. While the terms in TF-IDF are usually words, this is not a requirement. We can use n-grams, sequences of N continuous characters, to convert the address into a vector representation based on the character level. For example:

```python
#Input
text = "Birmingham"

#Create a list of n-grams
n_grams = [text[i:i+3] for i in range(len(text)-2+1)]
print(n_grams)
#['Bir', 'irm', 'rmi', 'min', 'ing', 'ngh', 'gha', 'ham']

```

In the third part, we can use a similarity score to evaluate the similarity among all potential candidate addresses; for instance by ranking the candidate addresses based on the similarity score.

Using TF-IDF with n-grams as terms to find similar strings transforms the problem into a matrix multiplication problem, which is computationally much cheaper. This approach can significantly reduce the **memory** it takes to compare strings in comparison to a fuzzy string matching algorithm with TF-IDF and a nearest neighbours algorithm. More importantly, using a GPU for the matrix multiplication can further speed up the string comparison.

## Case Study
### Example data
In this tutorial, we will be using two US hospital data sets. The [first](https://raw.githubusercontent.com/chris1610/pbpython/master/data/hospital_account_info.csv) is a data set that contains hospital banking details (see **Table 2**). The [second](https://raw.githubusercontent.com/chris1610/pbpython/master/data/hospital_reimbursement.csv) data set contains information on payments hospitals have received from the insurance provider (see **Table 3**). We would like to linkt these two datasets for further analysis, however, unfortunately, they currently are not linked.

|    |   Account_Num | Facility Name                     | Address                   | City         | State   |   ZIP Code | County Name   | Phone Number   | Hospital Type             | Hospital Ownership             |
|---:|--------------:|:----------------------------------|:--------------------------|:-------------|:--------|-----------:|:--------------|:---------------|:--------------------------|:-------------------------------|
|  0 |         10605 | SAGE MEMORIAL HOSPITAL            | STATE ROUTE 264 SOUTH 191 | GANADO       | AZ      |      86505 | APACHE        | (928) 755-4541 | Critical Access Hospitals | Voluntary non-profit - Private |
|  1 |         24250 | WOODRIDGE BEHAVIORAL CENTER       | 600 NORTH 7TH STREET      | WEST MEMPHIS | AR      |      72301 | CRITTENDEN    | (870) 394-4113 | Psychiatric               | Proprietary                    |

**Table 2: Sample of the account hospital data**

|    |   Provider_Num | Provider Name                    | Provider Street Address    | Provider City   | Provider State   |   Provider Zip Code |   Total Discharges |   Average Covered Charges |   Average Total Payments |   Average Medicare Payments |
|---:|---------------:|:---------------------------------|:---------------------------|:----------------|:-----------------|--------------------:|-------------------:|--------------------------:|-------------------------:|----------------------------:|
|  0 |         839987 | SOUTHEAST ALABAMA MEDICAL CENTER | 1108 ROSS CLARK CIRCLE     | DOTHAN          | AL               |               36301 |                118 |                   20855.6 |                  5026.19 |                     4115.52 |
|  1 |         519118 | MARSHALL MEDICAL CENTER SOUTH    | 2505 U S HIGHWAY 431 NORTH | BOAZ            | AL               |               35957 |                 43 |                   13289.1 |                  5413.63 |                     4490.93 |

**Table 3: Sample of the reimbursement data**

Below we compare the speed of computing TF-IDF using *sklearn* (CPU) and RAPIDS' *cuML* (GPU) module. To make the difference more distinguishable, we will test how the increase of vocabulary size would affect the performance in multiple runs.

### Setting up
We start by importing the libraries that we will need, loading the individual data sets, and concatenating the individual address components in each of the data sets.

```python
#Import libraries
import cudf
import pandas as pd
from cuml.feature_extraction.text import TfidfVectorizer as GPU_TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as CPU_TfidfVectorizer
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Load data
data1_url = "https://raw.githubusercontent.com/chris1610/pbpython/master/data/hospital_account_info.csv"
data2_url = "https://raw.githubusercontent.com/chris1610/pbpython/master/data/hospital_reimbursement.csv"

account = pd.read_csv(data1_url) #Hospital account information
reimbursement = pd.read_csv(data2_url) #Hospital reimbursement information

#Converting facility name, address, city and state into one string
account_full_address = account.apply(lambda x: " ".join([x['Facility Name'],x['Address'],x['City'],x['State']]), axis=1).tolist()

#Converting facility name, address, city and state into one string
reimbursement_full_address = reimbursement.apply(lambda x: " ".join([x['Provider Name'],x['Provider Street Address'],x['Provider City'],x['Provider State']]), axis=1).tolist()
```

### TF-IDF vectorisation
Now we are set-up, we can assess the impact of data size on computational time for both our CPU and GPU approach. To allow for a fair comparison between the CPU and GPU, we will be using the same data set and increase the size by the data set by *x* times on each run.

```python
#Inititate sklearn vectoriser and cuml vectosiser

#CPU vectorizer from sklearn
cpu_vectorizer = CPU_TfidfVectorizer(analyzer='char',ngram_range=(1,2))
#GPU vectorizer from cuml
gpu_vectorizer = GPU_TfidfVectorizer(analyzer='char',ngram_range=(1,2))

#analyzer='char' means we are using character as the input
#ngram_range = (1,2) means we are looking at both unigram and bigram for the model input

#Manually inflating number of rows with 10 run times
total_datasize = []
cpu_time = []
gpu_time = []
for run in range(1,10):
  for i in range(1,50):
    #Manually inflating the number of records
    input = reimbursement_full_address*i
    total_datasize.append(len(input))
    #Cpu runtime --------------------------------  
    start = time.time()
    cpu_output = cpu_vectorizer.fit_transform(input)
    done = time.time()
    elapsed = done - start
    cpu_time.append(elapsed)

    #gpu runtime --------------------------------
    start = time.time()
    #Convert input to cudf series
    gpu_output = gpu_vectorizer.fit_transform(cudf.Series(input))
    done = time.time()
    elapsed = done - start
    gpu_time.append(elapsed)

#Create a dataframe to store the results
gpu_elapsed = pd.DataFrame({"time":gpu_time,"data_size":total_datasize,'label':"gpu"})
cpu_elapsed = pd.DataFrame({"time":cpu_time,"data_size":total_datasize,'label':"cpu"})
result = pd.concat([gpu_elapsed,cpu_elapsed]).reset_index()

#Plot results
fig, ax = plt.subplots(figsize=(10,10))
sns.lineplot(x= 'data_size',y='time',hue = 'label',data = result,ax = ax )
plt.xlabel('Data Size')
plt.ylabel("Time Elapsed ")
plt.title("Comparing the speed of TF-IDF vectorisation on CPU and GPU")
plt.show()
print(cpu_output.shape) #(25000, 874) -> meaning we have 24273 rows of address and 874 characters in the vocabulary as input
```

<figure title = "Speed comparison">
     <center>
     <p><img src="https://github.com/jasoncpit/GPU-Analytics/blob/master/Pictures/tf_idf_vec.png?raw=true">
    <figcaption>
    <b>Figure 1: Comparing the speed of TF-IDF vectorisation on CPU and GPU.
    </b>
    </figcaption>
    </center>
</figure>

There are few things we can observe in the code above. First, the code between the CPU and GPU is *almost* identical. The only difference is that we are using the functions from different libraries. Second, we can see that the GPU is able to process the data much faster than the CPU. On average, it takes less than **0.1** seconds to process around 25 thousand addresses on the GPU. In comparison, it takes **1** second on the CPU for the same input dataset. Furthermore, we can also observe that the run time of *CuML’s* TfidfVectorizer is almost constant as the input data size increases, whereas the run time of scikit-learn’s TfidfVectorizer grows exponentially. This is because the GPU is able to process the data in parallel, which makes it much more efficient than the CPU.

### Cosine similarity
Now that we have our TF-IDF matrix, we want to use the hospital reimbursement data to find the most relevant address from the hospital data. To do this, we can find the address with the smallest distance (or highest similarity) to our query address. For the second step, we will compare the speed of computing the cosine similarity using the CPU and GPU. We do this by comparing the computation time of calculating the pair-wise cosine similarity between two matrices using *NumPy*, *CuPy* and *Numba* from scratch. We will use the TF-IDF matrix we created from the reimbursement data in the previous step as the input, and hospital data as the target.

```python
#Vectorize the target address
cpu_target =cpu_vectorizer.transform(account_full_address)
gpu_target = gpu_vectorizer.transform(cudf.Series(account_full_address))
```

```python
#Import libraries
import cupy as cp
import numpy as np
import scipy
import gc
import torch
from cupyx.scipy.sparse.linalg import norm as cp_norm
from scipy.sparse.linalg import norm as sc_norm

#Numpy
def np_cosine_similarity(query, target):
    # Assert that the input matrices have the same number of columns
    assert(query.shape[1] == target.shape[1])

    #Calculate the dot product
    dot_product = np.dot(query,target.T)

    #Calculate l2 norm for query and target
    query_norm = sc_norm(query,axis=1)
    target_norm = sc_norm(target,axis=1)

    return dot_product/(query_norm[:,np.newaxis]*target_norm[:,np.newaxis].T)

#Cupy
def cp_cosine_similarity(query, target):
    # Assert that the input matrices have the same number of columns
    assert(query.shape[1] == target.shape[1])
    #Initiate GPU instance
    with cp.cuda.Device(0):
        #Check whether the sparse matrix is compatible with Cupy, if not then convert
        if isinstance(query,scipy.sparse._csr.csr_matrix) and isinstance(target,scipy.sparse._csr.csr_matrix):
        #Convert the input matrices to sparse format and copy to the GPU
            query = cp.sparse.csr_matrix(query, copy=True)
            target = cp.sparse.csr_matrix(target, copy=True)

        #Dot product using cupy.dot()
        dot_product = query.dot(target.T)

        #Calculate l2 norm for query and target
        query_norm = cp_norm(query,axis=1)
        target_norm = cp_norm(target,axis=1)

        #Compute the cosine similarity
        output = dot_product / (cp.expand_dims(query_norm,axis=1) * cp.expand_dims(target_norm,axis=0))

        #Converting back to numpy array
        result = output.get()
    return result
```

From the above code, we can see that again the differences between *NumPy* and *CuPy* are minimal. Most mathematics functions in *CuPy* such as calculating the dot products and l2 norm have the same API as *NumPy* and *SciPy*. However, there are few things worth noticing. Firstly, when using *CuPy*, we have to manually initiate the GPU instance by calling function ``` with cp.cuda.Device(0): ```  and then execute your calculation. This allows to temporarily switching the currently active GPU device. Secondly, we need to manually specify the data type and transfer the data on the GPU instance by using ```cp.sparse.csr_matrix(...,copy = True) ```. This is to ensure we can properly prepare the data types for CUDA operation (for more information on the supporting data types, [please refer to the *CuPy* documentation](https://docs.cupy.dev/en/stable/overview.html)). And lastly, to get the data from the GPU, we need to use ```.get()``` to return a copy of the array on host memory.

#### Numba [Optional]
We can actually go a step further, we can actually explicitly write computing kernels with *Numba*. *Numba* is a Just-In-Time (JIT) compiler for Python that can take functions with numerical values or arrays and compile them to assembly, so they run at high speed. One advantage of writing with *Numba* is that we can write the low-level code in Python and then integrate it with CUDA assembly. In contrast to *CuPy*, *Numba* offers some flexibility in terms of the data types and kernels that can be used. However, *Numba* does not yet implement the full CUDA API, so some features are not available. It is worth noting here the goal of this post is to demonstrate the use of *Numba* and not to provide a comprehensive guide on how to use *Numba*. If you are interested in learning more about *Numba*, [see the *Numba* documentation](https://numba.pydata.org/). The following code shows an example of the implementation of the cosine similarity function using *Numba*.

```python
#Numba ----------------------------------------------------------------
from numba import cuda

#Define the kernel for calculation
@cuda.jit #compiler decorator
def pairwise_cosine_similarity(A, B, C):
    i, j = cuda.grid(2) #use 1 for x
    if i < C.shape[0] and j < C.shape[1]:
        dot = 0.0
        normA = 0.0
        normB = 0.0
        for k in range(A.shape[1]):
            a = A[i, k]
            b = B[j, k]
            dot += a * b
            normA += a ** 2
            normB += b ** 2
        C[i, j] = dot / (normA * normB)**0.5

def Numba_cuda_cosine_similarity(query,target):
    #Assert that the input matrices have the same number of columns
    assert(query.shape[1] == target.shape[1])
    #Allocate memory on the device for the result
    output = cuda.device_array((query.shape[0],target.shape[0]))


    #Check whether the sparse matrix is compatible with numba, if not then convert
    if isinstance(query,scipy.sparse._csr.csr_matrix) and isinstance(target,scipy.sparse._csr.csr_matrix):
    #Convert the input matrices to numpy array and copy to the GPU
        query = cuda.to_device(query.toarray())
        target = cuda.to_device(target.toarray())
    #Set the number of threads in a block -----
    threadsperblock = (32,32)

    #Calculate the number of thread blocks in the grid
    blockspergrid_x = (output.shape[0] + (threadsperblock[0] - 1)) // threadsperblock[0]
    blockspergrid_y = (output.shape[1] + (threadsperblock[1] - 1)) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    #Starting the kernel
    pairwise_cosine_similarity[blockspergrid, threadsperblock](query, target, output)

    #Copy the result back to the host
    return output.copy_to_host()
```

Let's break down the code. ```@cuda.jit``` is a decorator, and it defines functions which will compile on the GPU (kernels). The ```pairwise_cosine_similarity``` function defines the steps of calculation inside the kernel. The ```cuda.grid``` function returns the 2D grid indices for the current thread executing in a CUDA kernel. The CUDA programming model uses a grid of threads to perform parallel computation, where each thread operates on a different portion of the input data. In the example code, the ```cuda.grid``` function is used to determine the indices of the current thread in the grid, represented as ```i``` and ```j``` in the code. These indices are used to access the corresponding row of the input matrices ```A``` and ```B```, and the corresponding location in the output matrix ```C```. In essence, the ```cuda.grid``` function is used to ensure that each thread operates on a different portion of the input data and outputs its result to the corresponding location in the output matrix, thus enabling the parallel computation of the cosine similarity. Furthermore, we need to manually define the shape of matrix ```C```, and allocate memory on the device for the result. This is because the kernels cannot return numerical values, so we can get around that by passing inputs and outputs.  

In the ```Numba_cuda_cosine_similarity``` function, we follow the same logic as with the *CuPy* function, however, we have to manually define the number of threads in a block and calculate the number of thread blocks in the grid. The ```threadsperblock``` defines a group of threads that can execute concurrently. A single thread represents the smallest unit of execution in a CUDA kernel. Each thread operates on a different portion of the input data and computes its result. The results computed by each thread are then combined to produce the final result. In general, the number of threads per block is a trade-off between performance and memory usage. Increasing the number of threads per block can increase performance by allowing more parallel computation, but it also increases the memory usage of the block. The GPU has a limited amount of shared memory that is shared by all threads in a block, and increasing the number of threads per block can cause the shared memory usage to exceed the available memory. The value of ```32, 32``` for ```threadsperblock``` is a common choice because it is a relatively small number of threads that can still provide good performance, while minimising the memory usage of the block. However, the optimal value for ```threadsperblock``` depends on the specific requirements of the computation and the hardware being used, and it may need to be adjusted for different computations or hardware.

Next we define the number of thread blocks in the grid with the variable ```blockspergrid```. The number of thread blocks in the grid is defined by the number of threads in each dimension of the block, and the number of rows and columns in the output matrix. The purpose of adding ```(threadsperblock[0] - 1) to C.shape[0]``` is to ensure that the integer division will round up to the nearest whole number, rather than rounding down. In the last step, we can invoke the kernel by the defined block size and thread size and passing the input matrices and the output matrix to the kernel function. The kernel function will then perform the computation in parallel on the GPU. After the computation is finished, we can copy the result back to the host. The code below compares the *Numba* implementation to the *CuPy* (GPU) and *sklearn* (CPU) implementations.

```python
#Result placeholders
total_datasize = []
cpu_time = []
gpu_time = []
numba_time = []

for size in tqdm(range(1000,cpu_output.shape[0],5000)):
    total_datasize.append(size)
    query = cpu_output[0:size,]

    #CPU calculation
    start = time.time()
    _ = np_cosine_similarity(query,cpu_target)
    done = time.time()
    elapsed = done - start
    cpu_time.append(elapsed)

    #GPU calculation
    start = time.time()
    _ = cp_cosine_similarity(query,cpu_target)
    done = time.time()
    elapsed = done - start
    gpu_time.append(elapsed)

    #Numba CUDA ----
    start = time.time()
    _ = Numba_cuda_cosine_similarity(query,cpu_target)
    done = time.time()
    elapsed = done - start
    numba_time.append(elapsed)

#Plot
fig, ax = plt.subplots(figsize=(10,10))
gpu_elapsed = pd.DataFrame({"time":gpu_time,"data_size":total_datasize,'label':"gpu"})
cpu_elapsed = pd.DataFrame({"time":cpu_time,"data_size":total_datasize,'label':"cpu"})
numba_elasped = pd.DataFrame({"time":numba_time,"data_size":total_datasize,'label':"numba"})
result = pd.concat([gpu_elapsed,cpu_elapsed,numba_elasped]).reset_index()
sns.lineplot(x= 'data_size',y='time',hue = 'label',data = result,ax = ax )
plt.xlabel('Data Size')
plt.ylabel("Time Elapsed ")
plt.title("Comparing the speed of matrix multiplication on CPU and GPU")
plt.show()
```

<figure title = "Speed comparison">
     <center>
     <p><img src="https://github.com/jasoncpit/GPU-Analytics/blob/master/Pictures/chp1_matrix_multiplication.png?raw=true">
    <figcaption>
    <b>Figure 2: Comparing the speed of matrix multiplication on CPU and GPU.
    </b>
    </figcaption>
    </center>
</figure>

Again we can see a major speed difference. Overall, the operations on GPU are much faster than the ones on CPU. Specifically, by leveraging JIT compilation and *Numba* CUDA, we can reduce the time of matrix multiplication from 10.8 seconds to around 1 second. The code above is a demonstration of matrix multiplication, but in practice, we can use functions such as ```cosine_similarity``` from ```sklearn.metrics.pairwise``` or ```sparse_pairwise_distances ```from ```cuml.metrics``` to calculate the cosine similarity between two matrices.

### Similarity scores
Now we know how to run and speed up string matching, we can have a look at the actual results of the task. In this step, we will use the address with the highest cosine similarity result to find the best match for each query. Since we do not have a ground truth, we will be using the zip code as a proxy to check the accuracy of the model.   

```python
#Import libraries
from collections import defaultdict

#Run analysis pipeline on the GPU
reimbursement_tfidf= gpu_vectorizer.fit_transform(cudf.Series(reimbursement_full_address))
account_tfidf = gpu_vectorizer.transform(cudf.Series(account_full_address))
similarity_matrix = Numba_cuda_cosine_similarity(reimbursement_tfidf,account_tfidf)
result = defaultdict(list)

#Getting the reimbursement address and state, account address and state
for index in range(similarity_matrix.shape[0]):
    most_similar_index = similarity_matrix[index].argmax()
    result['reimbursement_address'].append(reimbursement_full_address[index])
    result['account_address'].append(account_full_address[most_similar_index])  
    result['reimbursment_zip'].append(reimbursement.loc[index,'Provider State'])
    result['account_zip'].append(account.loc[most_similar_index,'State'])
    result['similarity_score'].append(similarity_matrix[index][most_similar_index])

result_df = pd.DataFrame(result)
print(result_df.sort_values('similarity_score',ascending=True).head(2)
print(result_df.sort_values('similarity_score',ascending=False).head(2)
```

|      | reimbursement_address                            | account_address                                                                |   reimbursment_zip |   account_zip |   similarity_score |
|-----:|:-------------------------------------------------|:-------------------------------------------------------------------------------|-------------------:|--------------:|-------------------:|
| 1325 | UNITY HOSPITAL 550 OSBORNE ROAD FRIDLEY MN       | UNITY HOSPITAL OF ROCHESTER 1555 LONG POND ROAD ROCHESTER NY                   |              55432 |         14626 |           0.551186 |
| 1634 | TLC HEALTH NETWORK 845 ROUTES 5 AND 20 IRVING NY | STANDING ROCK INDIAN HEALTH SERVICE HOSPITAL 10 NORTH RIVER ROAD FORT YATES ND |              14081 |         58538 |           0.555283 |

**Table 4: Examples of matched records with the lowest cosine similarity score.**

|      | reimbursement_address                                                         | account_address                                                               |   reimbursment_zip |   account_zip |   similarity_score |
|-----:|:------------------------------------------------------------------------------|:------------------------------------------------------------------------------|-------------------:|--------------:|-------------------:|
| 2242 | BAPTIST MEMORIAL HOSPITAL UNION CITY 1201 BISHOP ST, PO BOX 310 UNION CITY TN | BAPTIST MEMORIAL HOSPITAL UNION CITY 1201 BISHOP ST, PO BOX 310 UNION CITY TN |              38261 |         38261 |                  1 |
|   89 | BANNER DESERT MEDICAL CENTER 1400 SOUTH  DOBSON ROAD MESA AZ                  | BANNER DESERT MEDICAL CENTER 1400 SOUTH  DOBSON ROAD MESA AZ                  |              85202 |         85202 |                  1 |

**Table 5: Examples of matched records with the highest cosine similarity score.**

The tables show  that the matched records with the lowest cosine similarity score are not the same, but the matched records with the highest cosine similarity score are  almost identical. This is a good sign that the model is working as expected, we still need the inspect the accuracy scores in a bit more detail.

```python
#Import libraries
from sklearn.metrics import accuracy_score

#Get accuracy scores
print(accuracy_score(result_df.reimbursment_zip,result_df.account_zip))
```

The accuracy score is 0.97, which means that 97% of the time, the model identifies the best match within the same zip code. Furthermore, by inspecting the distribution of the similarity score (see **Figure 3**), we can see that majority of matched records have cosine similarity scores above 0.95. However, there are some records that have a similarity score below 0.5, this may be due to the absence of such address from the hospital account holders.

```python
#Import libraries
import matplotlib.pyplot as plt

#Plot
fig,ax = plt.subplots(figsize=(10,10))
result_df.similarity_score.plot(kind='hist',bins=10,ax=ax)
plt.xlabel('Cosine Similarity Score')
plt.title('Distribution of cosine similarity score')
```

<figure title = "Similarity scores  ">
     <center>
     <p><img src="https://github.com/jasoncpit/GPU-Analytics/blob/master/Pictures/chp1_cosine_distribution.png?raw=true">
    <figcaption>
    <b>Figure 3: Distribution of cosine similarity score among matched records.
    </b>
    </figcaption>
    </center>
</figure>

## Conclusion
In this Case Study we have explored the potential of utilising GPUs to speed up explicit data linkage process that rely on string comparisons. By implementing vectorisation and cosine similarity calculation on a GPU, we were able to significantly enhance the performance of the process. More importantly, we can see that the functions and code can rather easily be ported to a GPU without too much modification.

More general, however, address matching is a complex problem that involves many factors that can impact the accuracy of the model. Some of these challenges include the presence of special characters, abbreviations, changes and absence of unique identifiers in the data. Often the task of data linkage is not a one-time process, but rather an iterative process that requires constant monitoring on the performance of the model and making adjustments accordingly. GPU can be a great tool to help speed up such processes.

One last thing to be aware of, particularly when dealing with large datasets, is the memory limitation of the GPU. One might consider using a GPU in conjunction with a scaling tool such as DASK, which can distribute the computation across multiple machines and servers. This approach can help mitigate the GPU memory limitation that can arise when processing larger data sets. Additionally, we can consider the batch size or the number of records that can be processed at a time to ensure that the GPU's memory is not overloaded.
