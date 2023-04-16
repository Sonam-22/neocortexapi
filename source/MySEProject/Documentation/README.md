## Implementation of a SDR classifier.

A classifier in machine learning is an algorithm that automatically orders or categorizes data into one or more set of “classes”. The SDR classifier takes the form of a single layer classification network that takes SDRs as input and outputs a predicted distribution of classes.The SDR Classifier accepts a binary input pattern from the level below (the "activationPattern" - the vector of Temporal Memory’s active cells) and information from the sensor and encoders (the "classification") describing the true (target) input.

![image](https://user-images.githubusercontent.com/116737927/213930553-b17a2fa3-12fd-451c-8975-28eb94ce7ce8.png)

Fig1. SDR Classifier in relation to the other modules of an HTM Network

**SDR** by books is known as Sparse Distributed Representations. It is the fundamental data structure which is used to represent and model the information in brain and in HTM systems. Generally, computers represents the infromation in form of traditional data structures such as trees, lists, ASCII representaions etc. These data structures are well suited when information is discrete and well defined. However, information is brain or in any HTM system is neither discrete nor well defined. Hence we represent the information in form of SDR, just like the brain does.

A SDR contains thousand of bits and each bit corresponds to a neuron. At any given point in time, for a given set of bits, certain bits are active and rest of them are inactive. This arrangement represent a thing at that point in time. At next point in time, representation changes making different bits active. Now this arrangement represents an another thing. As the time elapses and machine learns from various inputs, we have multiple SDRs representing muliple things and pattern of real world. These SDR can be associated, linked and compared to predict and classify new types of things which are completely unknown to the system.

**HTM** by definition is Hierarchial Temporal Memory. It is the latest algorithm in AI which replicates the underlying working principles and processes of the neocortex of a biological human brain. In a biological brain, neocortex occupies almost 80% of the brain size and it is reponsible for all the conginitive action of a human.

HTM systems follows the hierarchical structure of data processing and it usually consists of lower level systems such input encoders and spatial poolers and highier level systems such as temporal memory and classifiers. HTM systems represents the data in SDR as described above. Input encoders converts the raw data into SDR and rest of the highier level systems use these SDRs to learn, classify and predict new results over a time period, very much like a biological brain.

The SDR classifier maps input patterns to class labels. There are as many output units as the number of class labels or buckets (in the case of scalar encoders). The output is a probabilistic distribution over all class labels. During inference, the output is calculated by first doing a weighted summation of all the inputs, and then perform a softmax nonlinear function to get the predicted distribution of class labels During learning, the connection weights between input units and output units are adjusted to maximize the likelihood of the model.

## Input:

The input could be any type of data such as numerical or categorical data, text, or images. The data is first encoded by the encoder which transforms it into a sparse distributed representation.
The encoded data is then fed into the spatial pooler which is responsible for selecting a subset of the active neurons in the encoded representation. The result is a sparse representation of the input data in which only a small subset of neurons are active.
Finally, the sparse representation is fed into the temporal memory and the SDR Classifier takes as input a set of active cells from the Temporal Memory, which are represented as a vector. Additionally, the input to the SDR Classifier includes information about the record number and the bucket index that were used to encode the input data using the Encoder. Buckets are classes which represent a pattern in form of bits.

## Classification:

The SDR Classifier uses patterns to select buckets by encoding the input data as an SDR and using the SDR as the key to look up the appropriate bucket.
The encoding process involves two steps: first, the input data is passed through an encoder that converts it into a sparse binary representation. This encoder can be configured in various ways, depending on the type of data being encoded. For example, if the input data is a scalar value, the encoder might use a scalar encoder that converts the value into a sparse binary representation with a fixed number of bits.
The second step is to create an SDR from the encoded binary representation. The SDR is a binary vector with a fixed number of bits, where only a small percentage of the bits are set to 1. The bits that are set to 1 are chosen in a way that ensures that similar input values result in similar SDRs.
Once the SDR has been created, it is used as the key to look up the appropriate bucket. The SDR Classifier maintains a set of buckets, where each bucket stores the state of the temporal memory for a particular pattern. When the SDR Classifier receives an input, it first encodes the input data and creates an SDR. It then uses the SDR to look up the appropriate bucket, and retrieves the state of the temporal memory stored in the bucket.
The state of the temporal memory is then used to compute the anomaly score for the input. If the anomaly score exceeds a certain threshold, the SDR Classifier marks the input as anomalous and updates the state of the temporal memory in the bucket to reflect the new pattern. This allows the SDR Classifier to learn and adapt to new patterns over time.

## Methods:

- compute(recordNum, patternNZ, classification, learn, infer)
  : Process one input sample

  - Parameters:

    - recordNum:Record number of this input pattern. Record numbers normally increase sequentially by 1 each time unless there are missing records in the dataset. Knowing this information insures that we don’t get confused by missing records.
    - patternNZ: List of the active indices from the output below. When the input is from TemporalMemory, this list should be the indices of the active cells.
      classification: Dict of the classification information

  - Returns:
    - Dict containing inference results, there is one entry for each step in steps, where the key is the number of steps, and the value is an array containing the relative likelihood for each bucketIdx starting from bucketIdx 0.

- infer(patternNZ, actValueList)
  : Return the inference value from one input sample.

  - Parameters:
    - patternNZ: list of the active indices from the output below
    - classification: dict of the classification information: bucketIdx: index of the encoder bucket actValue: actual value going into the encoder
  - Returns:
    - dict containing inference results, one entry for each step in steps. The key is the number of steps, the value is an array containing the relative likelihood for each bucketIdx starting from bucketIdx 0.

- inferSingleStep(patternNZ, weightMatrix)
  : Perform inference for a single step. Given an SDR input and a weight matrix, return a predicted distribution.

  - Parameters:
    - patternNZ: list of the active indices from the output below
    - weightMatrix: numsharp array of the weight matrix
  - Returns:
    - double[] of the predicted class label distribution

## Tech Stack

Microsoft .NET 7 Core

[NumSharp](https://github.com/SciSharp/NumSharp)

## Project Setup

- Clone the [repository](https://github.com/wubie23/neocortexapi.git) and switch to branch `team_lightning`.
- [Download](https://code.visualstudio.com/download) and Install Microsoft Visual Studio.
- Install the Microsoft .NET 7 Core
- Open the [Project](https://github.com/wubie23/neocortexapi/tree/team-lightening/source/MySEProject) in Visual Studio Code
- Verify the results by executing test cases at [SdRClassifierTest](https://github.com/wubie23/neocortexapi/tree/team-lightening/source/MySEProject/SDRClassifier/SdRClassifierTest)

## Documentation

Detailed documentation about project and the SDR Classifier Algorithm is available at
[SDR-Classifier.pdf](https://github.com/wubie23/neocortexapi/blob/team-lightening/source/MySEProject/Documentation/SDR-Classifier.pdf)

## Refrences

1. M. I. Jordan and T. M. Mitchell, “Machine learning: Trends, Perspectives, and prospects,” Science, vol. 349, no. 6245, pp. 255–260, 2015.
2. nupic.docs.numenta.org
3. Alex Graves. Supervised Sequence Labeling with Recurrent Neural Networks, PhD Thesis, 2008
4. SDR classifier, 10-Sep-2016. [Online]. Available: https://hopding.com/sdr-classifier#title.
5. [Numenta SDR Classifier](https://github.com/numenta/nupic/blob/master/src/nupic/algorithms/sdr_classifier.py)
6. [Numenta SDR Classifier tests](https://github.com/numenta/nupic/blob/master/tests/unit/nupic/algorithms/sdr_classifier_test.py)
7. [Neocortex API](https://github.com/ddobric/neocortexapi)
