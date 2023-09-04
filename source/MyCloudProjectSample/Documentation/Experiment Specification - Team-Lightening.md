# ML22/23-4 Investigate and Implement SDR Classifier - Team-Lightening - Azure Cloud Implementation

## Description

This experiment demonstrates a Multisequence learning experiment using SDRClassifier and complete HTM pipeline inlcuding Encoders, Spatial Pooler and Temporal Memory. In the final stage, job of classification of SDRs provided by Temporal memory is achieved using a SDRClassifier, configured in a zero step mode.

In this experiment, we first convert the raw sequence inputs into SDRs using the scalar encoder. In the next steps, we train the Spatial pooler and Temporal memory using the inputs of scalar encoder, with a total of 3500 cycles for each sequence. On the other hand, once the training of spatial pooler and Temporal memory is completed, we train the SDR Classifier, using the predictive cells of Temporal memory and Bucket index of scalar encoder, with a maximum of 3500 cycles.

While we train the SDR Classifier, we keep measuring the attained accuracy after each cycle. As a matter of fact, SDR Classifier updates its weight matrix during its supervised training phase, so that it could attain maximum possible accuracy.

In this experiment, we weight upto 30 times, until the accuracy of learning reaches maximum possible accuracy, which is 100 in this case. This part is repeated for each sequence. During each training cycle set, once training of SDR Classifier for a sequence is completed, temporal memory is reset. This enables the first element starts always from the beginning.

After a certain number cycles, which is around 250 to 300 in this experiment, learning gets completed for both the sequence.
Once the model trained with both the sequence, it is available to use for predictions.

In the next steps, we feed a few relevant numbers to the model. The model predicts the next element and the name of the sequence in which there is a highiest probablity of that number to appear. Based on the predictions results, average accuracy of the model is calculated.

# How to run this experiment on cloud.

Steps involved in executing the experiment on Azure.

1. Building the docker image
The code to build the docker container is added as shell script named `build-docker-image.sh`. We just have execute the script from the `source` folder.

```sh
$ sh build-docker-image.sh
```

## What is your experiment about

Describe here what your experiment is doing. Provide a reference to your SE project documentation (PDF)*)

1. What is the **input**?

2. What is the **output**?

3. What your algorithmas does? How ?

## How to run experiment

Describe Your Cloud Experiment based on the Input/Output you gave in the Previous Section.

**_Describe the Queue Json Message you used to trigger the experiment:_**  

~~~json
{
     ExperimentId = "123",
     InputFile = "https://beststudents2.blob.core.windows.net/documents2/daenet.mp4",
     .. // see project sample for more information 
};
~~~

- ExperimentId : Id of the experiment which is run  
- InputFile: The video file used for trainign process  

**_Describe your blob container registry:**  

what are the blob containers you used e.g.:  
- 'training_container' : for saving training dataset  
  - the file provided for training:  
  - zip, images, configs, ...  
- 'result_container' : saving output written file  
  - The file inside are result from the experiment, for example:  
  - **file Example** screenshot, file, code  


**_Describe the Result Table_**

 What is expected ?
 
 How many tables are there ? 
 
 How are they arranged ?
 
 What do the columns of the table mean ?
 
 Include a screenshot of your table from the portal or ASX (Azure Storage Explorer) in case the entity is too long, cut it in half or use another format
 
 - Column1 : explaination
 - Column2 : ...
Some columns are obligatory to the ITableEntities and don't need Explaination e.g. ETag, ...
 
