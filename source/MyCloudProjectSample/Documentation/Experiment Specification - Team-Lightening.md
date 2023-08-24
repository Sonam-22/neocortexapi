# ML22/23-4 Investigate and Implement SDR Classifier - Team-Lightening - Azure Cloud Implementation

This experiment demonstrates a Multisequence experiment using SDRClassifier and complete HTM pipeline inlcuding Encoders, Spatial Pooler and Temporal Memory. In the final stage job of classification of SDRs provided by Temporal memory is achieved using a SDRClassifier, configured in a zero step mode.

~~~csharp
public voiud MyFunction()
{
    Debug.WriteLine("this is a code sample");
}
~~~


## What is your experiment about

Reference of our SE project documentation (PDF)*)
https://github.com/Sonam-22/neocortexapi/blob/team-lightening/source/MySEProject/Documentation/Implementation%20of%20SDR%20Classifier%20presentation.mp4

Readme.md file availiiable about project here :-
https://github.com/Sonam-22/neocortexapi/blob/team-lightening/source/MySEProject/Documentation/README.md

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
 
