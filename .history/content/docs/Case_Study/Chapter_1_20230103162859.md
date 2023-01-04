---
title: "Chapter 1 - Address matching and Geocoding address"
weight: 1
# bookFlatSection: false
# bookToc: true
# bookHidden: false
# bookCollapseSection: false
# bookComments: false
# bookSearchExclude: false
---

## Addresses and Geocoding textual datasets using GPU 

Data linkage is the process of joining together records that pertain to the same entity, such as a person or business. In its most simplistic form, links (or “matches”) can be completed by comparing unique identifier keys specific to each object, for example, a person’s National Insurance number. 

However, considerable time and effort must be invested in preparing the data prior to linking, to increase the chance of detecting matches while ensuring accuracy is preserved. 

In this tutorial, I'm experimenting the possibility of using GPU for data linkage. More specifically, we can stress-test the speed differenece and performance between RAPIDS cuML (GPU-package) and Sklearn (CPU-package) on the same ML pipeline.  For this tutorial, I'm using Tesla T4 for the experimentation. 

# Pipeline for data linkage 

At the most foundemntal level, the goal of data linkage is to compute the similarity between two strings (e.g. Levenshtein, Damerau-Levenshtein, Jaro-Winkler, q-gram, cosine).

One approach we can experiement is to transform each string into some kind of vectors and compute the pair-wise similarity. 

- Load our data 
- Vectorize our data using TF-IDF 
- Compare pair-wise similairty of the data

At each steps, we can compare the compuatation speed differences between PAPIDS cuML and Sklearn.  


## Data 
In this tutorial, I have two data sets. The first is a data set that contains basic hospital account number, name and ownership information. The second data set contains hospital information (called provider) as well as the number of discharges and Medicare payment for a specific Heart Failure procedure. The goal is that we want to match up the hospital reimbursement information with the first data so we have more information to analyze our hospital customers. In this instance we have 5339 hospital accounts and 2697 hospitals with reimbursement information. 
