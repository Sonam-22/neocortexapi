# **Implementation of SDR Classifier in C#:**
The SDR (Sparse Distributed Representation) classifier is a machine learning algorithm that is used for pattern recognition and classification tasks. It is based on the theory of Hierarchical Temporal Memory (HTM) and uses sparse, distributed representations of the input data to learn and classify patterns.

Example of C# code for an SDR classifier:
https://github.com/wubie23/neocortexapi/blob/Wubishet/source/MySEProject/basicSDRClassifier.cs

This code creates a new SDR classifier with specified input and column dimensions, trains the classifier on a set of training data, and then classifies a new input. The classifier.Train(trainingData) method is used to train the classifier, and the classifier.Classify(input) method is used to classify new inputs.


**Note:**

This is a basic example and there is missing a lot of context, like the data loading, and it doesn't cover all the functionality that a real-world application would need.

Additionally, the package "Numenta.HTM" is open source package developed by Numenta, it's a C# implementation of the HTM theory, this package is not part of the .net core or .net framework, you need to install it via NuGet package manager.

**Encoder Implementation**

An encoder is a component of the Hierarchical Temporal Memory (HTM) model that is used to convert raw input data into a sparse, distributed representation that can be used by the Spatial Pooler and Temporal Memory.

C# code for implementing an encoder:
https://github.com/wubie23/neocortexapi/blob/Wubishet/source/MySEProject/encoderImplementation.cs

This code creates a new ScalarEncoder encoder, this encoder is used for encoding scalar value, it has some parameters like w, minval, maxval and n that are used to configure the encoder.

Then, it encodes a value using the encoder.encode(value) method, which returns the encoded representation of the input value.

You can also use the encoder.decode(encoded) method to decode the encoded value back to the original value.

Note:

In addition to ScalarEncoder, there are other types of encoders like DateEncoder, DeltaEncoder, GeospatialCoordinateEncoder, CategoryEncoder, and more, each one of them is specialized for a specific type of data, you can use the appropriate one for your data.

**Spatial Pooler Implementation**

The Spatial Pooler is an algorithm used in the Hierarchical Temporal Memory (HTM) model for reducing the dimensionality of input data. It learns which features are most informative and creates a sparse, distributed representation of the input.

C# code for implementing the Spatial Pooler:
https://github.com/wubie23/neocortexapi/blob/Wubishet/source/MySEProject/spatialPoolerImplementation.cs

This code creates a new Spatial Pooler with specified input and column dimensions, sets its parameters, initializes the Spatial Pooler, and then computes the Spatial Pooler's output for a given input.

The spatialPooler.setPotentialRadius(16),spatialPooler.setPotentialPct(0.5),spatialPooler.setGlobalInhibition(true),spatialPooler.setNumActiveColumnsPerInhArea(10),spatialPooler.setStimulusThreshold(0) methods are used to set the Spatial Pooler's parameters.

The spatialPooler.init() method is used to initialize the Spatial Pooler, and the spatialPooler.compute(input, true, output) method is used to compute the Spatial Pooler's output for a given input.

**Temporal Memory Implementation** 

The Temporal Memory (TM) is an algorithm used in the Hierarchical Temporal Memory (HTM) model for learning temporal patterns in data by creating sparse, distributed representations of the input.

C# code for implementing the Temporal Memory:
https://github.com/wubie23/neocortexapi/blob/Wubishet/source/MySEProject/temporalMemoryImplementation.cs

This code creates a new Temporal Memory with specified column dimensions, sets its parameters, initializes the Temporal Memory, and then computes the Temporal Memory's output for a given input.

The temporalMemory.setActivationThreshold(13),temporalMemory.setMinThreshold(10),temporalMemory.setMaxNewSynapseCount(20),temporalMemory.setInitialPermanence(0.21),temporalMemory.setConnectedPermanence(0.5),temporalMemory.setPermanenceIncrement(0.10),temporalMemory.setPermanenceDecrement(0.10) methods are used to set the Temporal Memory's parameters.

The temporalMemory.init() method is used to initialize the Temporal Memory, and the temporalMemory.compute(input, output) method is used to compute the Temporal Memory's output for a given input.


