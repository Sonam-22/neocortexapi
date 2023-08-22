# ML22/23-4 Investigate and Implement SDR Classifier - Team-Lightening - Azure Cloud Implementation

## I. Software Engineering Project Description

### Software Engineering Project : [Code](https://github.com/Sonam-22/neocortexapi/tree/team-lightening/source/MySEProject) & [Paper](https://github.com/Sonam-22/neocortexapi/blob/team-lightening/source/MySEProject/Documentation/SDR-Classifier.pdf)

The Hierarchical Temporal Memory (HTM) model has evolved as a result of advancements in Artificial Neural Networks, which have also improved the Cortical Learning Algorithm (CLA) classifier. This development led to the development of the Sparse Distributed Representation (SDR) classifier, which departs from the CLA method by combining maximum likelihood estimation with a feed-forward neural network. The SDR Classifier is implemented in this work utilizing Numenta's tried-and-true technique.

With changes to its weight matrix, the SDR Classifier outperforms its predecessor in continuous learning by rewarding correct predictions and punishing inaccurate ones. The learning approaches are used in this process. Sparse Distributed Representations (SDRs), which are essential to the cortex theory of intelligence, serve as the foundation for the SDR Classifier. Using HTM concepts to build overlapping SDRs that efficiently categorize data hierarchically, this method develops an algorithm for anomaly detection and classification.

The SDR Classifier's higher accuracy and efficiency compared to modern algorithms are demonstrated via evaluation across benchmark datasets. These outcomes highlight the SDR Classifier's effectiveness and demonstrate how widely applicable it is in a variety of fields.




## II. Cloud Project Description

The execution of this project according to the workflow is implemented by the class Experiment.cs. It is situated in NeoCortexUtils directory in the project folder and called by the main Program.cs for implementation. The purpose of experiment class is to establish the folder path locally where files will be downloaded from blob storage and the data present in the downloaded file is executed and uploaded back on azure (Blob & Table). The program will then be executed until signaled to cancel via a cancellation token. The results of the program will then be uploaded in the storage blob and table. This stream of actions is illustrated in several methods discussed below.
