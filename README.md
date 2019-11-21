# i-dash2019
## NUS - CS 6203 Final Project
This is a final class project on secure multi-party computation
The code is still a work-in-progress. 
The project was/is NOT an official submission to the i-dash challenge

## run : sudo docker run -it --net=host houruomu/cs6203 jupyter notebook --allow-root

## Track IV: Secure Collaborative Training of Machine Learning Model

### Background: 
Training a modern machine learning model often require a large amount of data. Oftentimes, however, data owners could be reluctant to share their data (e.g., genome data from human subjects) due to privacy concerns even though they also desire to build a better trained model. Therefore, it becomes highly important to allows two or more owners to build a joint ML model using a secure computing protocol such as Secure Multiparty Computation (SMC). This task is designed to understand the efficiency of the SMC implementation achievable in building a machine learning model for supporting a secure collaboration.
Experimental setting: We will provide two testing datasets, and each participating team will submit an implementation of a general training algorithm so that each testing dataset can be used to train a model. We will provide an ML model directly trained on the data as a benchmark.(ML model link) The solution does not need to use the same model as the benchmark, but it is supposed to perform similarly.

### Requirement:  
The solution should follow the security standard of SMC. The meta-data, i.e. the number of features/records should be made public, as well as the final model. However, any information that can't be inferred from that should not be leaked. Each computing node shouldn't learn anything about the data (i.e., the feature values for each record).
Evaluation Criteria: Submissions are qualified if they fulfill the security requirements. Qualified solutions are ranked based on their performances, including their prediction accuracy (how close it is comparing to the model built by the non-secure algorithm), total running time, and the communication cost (the rounds and sizes of data exchange among computing nodes in the SMC). The evaluation team will run the training code on the released data for up to 24 hours. The solutions that do not complete within 24 hours will be disqualified.
Dataset:link The testing datasets BC-TCGA and GSE2034 are gene expression data obtained from breast cancer patients. BC-TCGA is part of comprehensive molecular portraits of human breast tumors processed by TCGA, collected by using a microarray platform[1]. The dataset consists of expression levels of 17,814 genes (features) from 48 normal tissue samples (negative samples) and 422 breast cancer tissue samples (positive samples)[2]. GSE2034 is obtained by using the Affymetrix Human U133a GeneChips from frozen tumour samples[3]. It contains the expression levels of 12,634 genes from 225 breast cancer samples, in which 142 are recurrence tumor samples (positive samples) and the rest 83 samples are non-recurrence samples (negative samples) [2].

### Submission:  
Please install your solution and any required software or library in the docker (image can be downloaded at link; see the documentation at link), and submit the resulting docker file. You should make sure your program is executable within the docker configuration. Please set each party's address as 127.0.0.1 and use a different port as the communication channel. For evaluation purpose, we will put each docker container (for each party) on different machine/node. Thus, please do not put all three parties on one machine/node. The testbed of track 4 is running docker CE 19.03.01 docker 17.06.0-ce, on Ubuntu 18.04 Ubuntu 16.04 with Linux kernel version 4.15.0-55-generic x86_64 4.4.0-150-generic x86_64. Please make sure the runtime environments of submissions are the same with the testbed. The memory of the test machine is 8GB with 4G swap and its CPU is i5-3470 @3.2GHz with 4 processors Intel Xeon E3-1280 v5. As for multi-party evaluation, your implementation should allow us to config/specify/assign/modify the IP address of your docker image, so that we could deploy your submission in a right way on our cluster. Please contact Diyue Bu (diybu@indiana.edu) to get the link for the submission (please include your track number in the email).

### References:

    Cancer Genome Atlas Network et al. Comprehensive molecular portraits ofhuman breast tumours.Nature, 490(7418):61, 2012.
    Haozhe Xie, Jie Li, Qiaosheng Zhang, and Yadong Wang. Comparisonamong dimensionality reduction techniques based on random projectionfor cancer classification.Computational biology and chemistry, 65:165–172,2016.
    Yixin Wang, Jan GM Klijn, Yi Zhang, Anieta M Sieuwerts, Maxime PLook, Fei Yang, Dmitri Talantov, Mieke Timmermans, Marion E Meijer-vanGelder, Jack Yu, et al. Gene-expression profiles to predict distant metastasis of lymph-node-negative primary breast cancer.The Lancet, 365(9460):671–679, 2005
