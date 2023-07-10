using static System.Text.Json.JsonSerializer;
using NumSharp;
using System.Diagnostics;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Entities;
using System.Numerics;

namespace SDRClassifier
{

    /// <summary>
    /// The SDR Classifier accepts a binary input pattern from the
    /// level below(the "activationPattern") and information from the sensor and
    /// encoders(the "classification") describing the true (target) input.
    /// The activation pattern is provided as active indices of active bits from a SDR.
    /// The SDR classifier maps input patterns to class labels. There are as many
    /// output units as the number of class labels or buckets(in the case of scalar
    /// encoders). The output is a probabilistic distribution over all class labels.
    /// During inference, the output is calculated by first doing a weighted summation
    /// of all the inputs, and then perform a softmax nonlinear function to get
    /// the predicted distribution of class labels
    /// During learning, the connection weights between input units and output units
    /// are adjusted to maximize the likelihood of the model
    /// </summary>
    public class SDRClassifier<TIN, TOUT> : IClassifier<TIN, TOUT>
    {
      
        public int version = 1;
        public List<int> steps;
        public double alpha;
        public double actValueAlpha;
        public double verbosity;
        private int maxSteps;
        private LinkedList<Tuple<int, List<int>>> patternNZHistory;
        private int maxInputIdx;
        private int maxBucketIdx;
        private Dictionary<int, NDArray> weightMatrix = new();

     
        ///
        ///steps: (list) 
        ///alpha: 
        ///actValueAlpha: (float)
        ///verbosity: (int)
        ///
        private List<object?> actualValues;

        /// <summary>
        /// SDR classifier default constructor
        /// </summary>
        /// <param name="steps">Sequence of the different steps of multi-step predictions to learn</param>
        /// <param name="alpha">The alpha used to adapt the weight matrix during learning. A larger alpha results in faster adaptation to the data.</param>
        /// <param name="actValueAlpha">Used to track the actual value within each bucket. A lower actValueAlpha results in longer term memory</param>
        /// <param name="verbosity">Verbosity level, can be 0, 1, or 2</param>
        /// <param name="version">Current version of the classifier</param>
        public SDRClassifier(List<int> steps, double alpha, double actValueAlpha, double verbosity, int version)
        {
            if (steps.Count == 0)
            {
                throw new InvalidDataException("steps cannot be empty");
            }
            if (alpha < 0)
            {
                throw new InvalidDataException("alpha (learning rate) must be a positive number");
            }
            if (actValueAlpha < 0 || actValueAlpha >= 1)
            {
                throw new InvalidDataException("actValueAlpha be a number between 0 and 1");
            }

            // Save constructor args
            this.steps = steps;
            this.alpha = alpha;
            this.actValueAlpha = actValueAlpha;
            this.verbosity = verbosity;
            // Max # of steps of prediction we need to support
            this.maxSteps = this.steps.Max() + 1;
            // History of the last maxSteps activation patterns. We need to keep
            // these so that we can associate the current iteration's classification
            // with the activationPattern from N steps ago
            this.patternNZHistory = new();
            // This contains the value of the highest input number we've ever seen
            // It is used to pre-allocate fixed size arrays that hold the weights
            this.maxInputIdx = 0;
            // This contains the value of the highest bucket index we've ever seen
            // It is used to pre-allocate fixed size arrays that hold the weights of
            // each bucket index during inference
            this.maxBucketIdx = 0;
            // The connection weight matrix
            this.weightMatrix = new Dictionary<int, NDArray>();

            foreach (var step in this.steps)
            {
                this.weightMatrix.Add(step, np.zeros(shape: (this.maxInputIdx + 1, this.maxBucketIdx + 1)));
            }
            // This keeps track of the actual value to use for each bucket index. We
            // start with 1 bucket, no actual value so that the first infer has something
            // to return
            this.actualValues = new List<object?> { null };
            // Set the version to the latest version.
            // This is used for serialization/deserialization
            this.version = version;
        }

        /// <summary>
        /// Process one input sample.
        /// </summary>
        /// <param name="recordNum">
        /// Record number of this input pattern. Record numbers
        /// normally increase sequentially by 1 each time unless there are missing
        /// records in the dataset. Knowing this information insures that we don't get
        /// confused by missing records.
        /// </param>
        /// <param name="patternNZ">
        /// List of the active indices from the output below. When the
        /// input is from TemporalMemory, this list should be the indices of the
        /// active cells.
        /// </param>
        /// <param name="classification">
        /// Dictionary of the classification information where:
        /// <list type="bullet">
        /// <item>
        /// <description>bucketIdx: list of indices of the encoder bucket</description>
        /// </item>
        /// <item>
        /// <description>actValue: list of actual values going into the encoder</description>
        /// </item>
        /// </list>
        /// Classification could be None for inference mode.
        /// </param>
        /// <param name="learn">if true, learn this sample</param>
        /// <param name="infer">if true, perform inference</param>
        /// <returns>
        /// Dictionary containing inference results, there is one entry for each
        /// step in <c>steps</c>, where the key is the number of steps, and
        /// the value is an array containing the relative likelihood for
        /// each bucketIdx starting from bucketIdx 0.
        /// There is also an entry containing the average actual value to
        /// use for each bucket.The key is 'actualValues'.
        /// </returns>
        /// <exception cref="InvalidDataException">
        ///  Throws invalid data exception when record number increases randomly or classification is null and learn is true.
        /// </exception>
        public SDRClassification<TIN> Compute(
                int recordNum,
                List<int> patternNZ,
                SDRClassification<TIN> classification,
                bool learn,
                bool infer)
        {
            int numCategory = 0;
            var actValueList = Array.Empty<TIN>();
            var bucketIdxList = Array.Empty<double>();

            if (this.verbosity >= 1)
            {
                Debug.WriteLine(String.Format("learn: {0}", learn));
                Debug.WriteLine(String.Format("recordNum {0}:", recordNum));
                Debug.WriteLine(String.Format("patternNZ {0}: {1}", patternNZ.Count, string.Join(",", patternNZ.ToArray())));
                Debug.WriteLine(String.Format("classificationIn: {0}", Serialize(classification)));
            }
            // ensures that recordNum increases monotonically
            if (this.patternNZHistory.Count > 0)
            {
                if (recordNum < this.patternNZHistory.Last?.Value?.Item1)
                {
                    throw new InvalidDataException("the record number has to increase monotonically");
                }
            }
            // Store pattern in our history if this is a new record
            if (this.patternNZHistory.Count == 0 || recordNum > this.patternNZHistory.Last?.Value?.Item1)
            {
                this.patternNZHistory.AddLast(new LinkedListNode<Tuple<int, List<int>>>(Tuple.Create(recordNum, patternNZ)));
            }

            // To allow multi-class classification, we need to be able to run learning
            // without inference being on. So initialize retval outside
            // of the inference block.
            var retval = new SDRClassification<TIN>();
            // Update maxInputIdx and augment weight matrix with zero padding
            if (patternNZ.Max() > this.maxInputIdx)
            {
                var newMaxInputIdx = patternNZ.Max();
                foreach (var step in this.steps)
                {
                    this.weightMatrix[step] = np.concatenate((this.weightMatrix[step], np.zeros(shape: (newMaxInputIdx - this.maxInputIdx, this.maxBucketIdx + 1))), axis: 0);
                }
                this.maxInputIdx = Convert.ToInt32(newMaxInputIdx);
            }
            // Get classification info
            if (classification is not null)
            {
                bucketIdxList = classification.Classifications["bucketIdx"];
                actValueList = classification.ActValues;
                numCategory = classification.Classifications["bucketIdx"].Length;
            }
            else
            {
                if (learn)
                {
                    throw new InvalidDataException("classification cannot be None when learn=True");
                }
            }

            // Inference:
            // For each active bit in the activationPattern, get the classification
            // votes
            if (infer)
            {
                retval = Infer(patternNZ, actValueList);
            }

            if (learn && classification != null && classification.Classifications["bucketIdx"] != null)
            {
                UpdateWeightMatrix(bucketIdxList, actValueList, numCategory, recordNum);
            }

            return retval;
        }

        public TIN GetPredictedInputValue(Cell[] predictiveCells)
        {
            throw new NotImplementedException();
        }

        public void Learn(TIN input, Cell[] output)
        {
            throw new NotImplementedException();
        }

        public void Learn(TIN input, Cell[] output, object optionalInfo)
        {
            var cellIndicies = output
                .Select(v => v.Index)
                .ToList();
            var optionals = (SDRClassifierInput<TIN>) optionalInfo;
            SDRClassification<TIN> classification = new()
            {
                Classifications = new Dictionary<string, double[]>() {{"bucketIdx",  optionals.BucketIndex}},
                ActValues = optionals.ActValues
            };

            Compute(optionals.RecordNumber, cellIndicies, classification, true, false);
        }

        public List<ClassifierResult<TIN>> GetPredictedInputValues(int[] cellIndicies, short howMany = 1)
        {
            var inferredValues = Infer(cellIndicies.ToList(), null);
            return ToClassficationResults(inferredValues, howMany);
        }

        public List<ClassifierResult<TIN>> GetPredictedInputValues(int[] cellIndicies, short howMany, object optionalInfo)
        {
            var optionals = (SDRClassifierInput<TIN>)optionalInfo;
            SDRClassification<TIN> classification = new()
            {
                Classifications = new Dictionary<string, double[]>() { { "bucketIdx", optionals.BucketIndex } },
                ActValues = optionals.ActValues
            };
            var inferredValues = Compute(optionals.RecordNumber, cellIndicies.ToList(), classification, false, true);
            return ToClassficationResults(inferredValues, howMany);
        }

        private List<ClassifierResult<TIN>> ToClassficationResults(SDRClassification<TIN> inferredValues, short howMany = 1) {
            var firstStep = inferredValues.Classifications[steps[0].ToString()];
            var sorted = np.array(firstStep)
                .argsort<int>()["::-1"]
                .ToArray<int>();

            var classificationResults = new List<ClassifierResult<TIN>>();

            for (int i = 0; i < sorted.Length; i++)
            {
                var idx = sorted[i];
                var probablity = firstStep[idx] * 100;
                var actValue = inferredValues.ActValues[idx];
                classificationResults.Add(new ClassifierResult<TIN>()
                {
                    PredictedInput = actValue,
                    Similarity = probablity
                });
            }

            return classificationResults
                .Take(howMany)
                .ToList();
        }

        /// <summary>
        /// Updates the weight matrix.
        /// </summary>
        /// <param name="bucketIdxList">Array of buckets</param>
        /// <param name="actValueList">Array of actual values</param>
        /// <param name="numCategory">Category</param>
        /// <param name="recordNum">Record number for the pattern</param>
        private void UpdateWeightMatrix(double[] bucketIdxList, TIN[] actValueList, int numCategory, int recordNum)
        {
            foreach (int categoryI in Enumerable.Range(0, numCategory))
            {
                var bucketIdx = (int)bucketIdxList[categoryI];
                var actValue = actValueList[categoryI];
                // Update maxBucketIndex and augment weight matrix with zero padding
                if (bucketIdx > this.maxBucketIdx)
                {
                    foreach (int step in this.steps)
                    {
                        var toUpdate = (this.weightMatrix[step], np.zeros(shape: (maxInputIdx + 1, bucketIdx - maxBucketIdx)));
                        this.weightMatrix[step] = np.concatenate(toUpdate, axis: 1);
                    }
                    this.maxBucketIdx = Convert.ToInt32(bucketIdx);
                }
                // Update rolling average of actual values if it's a scalar. If it's
                // not, it must be a category, in which case each bucket only ever
                // sees one category so we don't need a running average.
                while (this.maxBucketIdx > this.actualValues.Count - 1)
                {
                    this.actualValues.Add(null);
                }
                if (this.actualValues[bucketIdx] == null)
                {
                    this.actualValues[bucketIdx] = actValue;
                }
                else
                {
                    if(IsNumeric(actValue)) {
                        var actValueDouble = Convert.ToDouble(actValue);
                        var actualValueDouble = Convert.ToDouble(this.actualValues[bucketIdx]);
                        this.actualValues[bucketIdx] = (TIN)(object)((1.0 - this.actValueAlpha) * (actualValueDouble) + this.actValueAlpha * actValueDouble);
                    }
                    else
                    {
                        this.actualValues[bucketIdx] = actValue;
                    }
                   
                }
            }
            foreach (var tuple in this.patternNZHistory)
            {
                var learnRecordNum = tuple.Item1;
                var learnPatternNZ = tuple.Item2;
                var error = CalculateError(recordNum, bucketIdxList);
                var nSteps = recordNum - learnRecordNum;
                if (this.steps.Contains(nSteps))
                {
                    foreach (var bit in learnPatternNZ)
                    {
                        var updatedAlpha = this.alpha * error[nSteps];
                        this.weightMatrix[nSteps][bit, ":"] += updatedAlpha;
                    }
                }
            }
        }
        /// <summary>
        /// Prints the classification information.
        /// </summary>
        /// <param name="retval">Dictionary of classification information</param>
        private void PrintVerbose(SDRClassification<TIN> retVal)
        {
            Debug.WriteLine("inference: combined bucket likelihoods:");
            Debug.WriteLine(String.Format("actual bucket values {0}", string.Join(",", retVal.ActValues)));
            foreach (var keyValue in retVal.Classifications)
            {
                var nSteps = keyValue.Key;
                var votes = keyValue.Value;
                if (nSteps == "actualValues")
                {
                    continue;
                }
                Debug.WriteLine(String.Format("{0} steps: {1}", nSteps, string.Join(",", votes)));
                var bestBucketIdx = np.array(votes).argmax();
                Debug.WriteLine(String.Format("most likely bucket idx: {0}, value: {1}", bestBucketIdx, retVal.ActValues[bestBucketIdx]));
            }
        }


        /// <summary>
        /// Return the inference value from one input sample. The actual
        /// learning happens in compute().
        /// </summary>
        /// <param name="patternNZ">list of the active indices from the output below</param>
        /// <param name="actValueList">
        /// dict of the classification information: bucketIdx: index of the encoder bucket actValue: actual value going into the encoder
        /// </param>
        /// <returns>
        /// dict containing inference results, one entry for each step in
        /// steps. The key is the number of steps, the value is an
        /// array containing the relative likelihood for each bucketIdx
        /// starting from bucketIdx 0.
        /// </returns>
        public SDRClassification<TIN> Infer(List<int> patternNZ, TIN[]? actValueList)
        {
            TIN? defaultValue;
            /**
             * Return value dict. For buckets which we don't have an actual value
             * for yet, just plug in any valid actual value. It doesn't matter what
             * we use because that bucket won't have non-zero likelihood anyways.
             * NOTE: If doing 0-step prediction, we shouldn't use any knowledge
             * of the classification input during inference.
            */
            if (this.steps[0] == 0 || actValueList == null)
            {
                defaultValue = default;
            }
            else
            {
                defaultValue = actValueList[0];
            }
            var actValues = this.actualValues
              .ConvertAll(item => (TIN)(item ?? defaultValue))
              .ToArray();
            var retval = new SDRClassification<TIN>();
            retval.ActValues = actValues;
            foreach (var nSteps in this.steps)
            {
                retval.Classifications[nSteps.ToString()] = InferSingleStep(patternNZ, this.weightMatrix[nSteps]);
            }
            if(verbosity >= 1) {
                PrintVerbose(retval);
            }
            return retval;
        }

        /// <summary>
        /// Perform inference for a single step. Given an SDR input and a weight matrix, return a predicted distribution.
        /// </summary>
        /// <param name="patternNZ">list of the active indices from the output below</param>
        /// <param name="weightMatrix">Multidimentional array of the weight matrix</param>
        /// <returns>
        ///  double[] of the predicted class label distribution
        /// </returns>
        public double[] InferSingleStep(List<int> patternNZ, NDArray weightMatrix)
        {
            double[] outputActivation = new double[maxBucketIdx + 1];
            // Calculate activation values by summing the weighed inputs
            for (int row = 0; row <= maxBucketIdx; row++)
            {
                foreach (int col in patternNZ)
                {
                    outputActivation[row] += weightMatrix[col, row];
                }
            }

            // softmax normalization
            // Exponentiate each activation value
            double[] expOutputActivation = new double[outputActivation.Length];
            for (int i = 0; i < expOutputActivation.Length; i++)
            {
                expOutputActivation[i] = Math.Exp(outputActivation[i]);
            }
            // Sum of Exponentiated activation value
            double expOutputActivationSum = expOutputActivation.Sum();
            double[] predictDist = new double[outputActivation.Length];
            for (int i = 0; i < predictDist.Length; i++)
            {
                predictDist[i] = expOutputActivation[i] / expOutputActivationSum;
            }

            return predictDist;
        }

        /// <summary>
        /// Calculate error signal
        /// </summary>
        /// <param name="recordNum">Record number of this input pattern from compute method</param>
        /// <param name="bucketIdxList">list of encoder buckets</param>
        /// <returns>
        /// Dictionary containing error. The key is the number of steps
        //  The value is array of error at the output layer
        /// </returns>
        public Dictionary<int, NDArray> CalculateError(int recordNum, double[] bucketIdxList)
        {
            var error = new Dictionary<int, NDArray>();
            var targetDist = np.zeros(this.maxBucketIdx + 1);
            var numCategories = bucketIdxList.Count();
            foreach (int bucketIdx in bucketIdxList)
            {
                targetDist[bucketIdx] = 1.0 / numCategories;
            }
            foreach (var tuple in this.patternNZHistory)
            {
                var learnRecordNum = tuple.Item1;
                var learnPatternNZ = tuple.Item2;
                var nSteps = recordNum - learnRecordNum;
                if (this.steps.Contains(nSteps))
                {
                    var inferred = InferSingleStep(learnPatternNZ, this.weightMatrix[nSteps]);
                    var predictDist = np.array(inferred);
                    error[nSteps] = targetDist - predictDist;
                }
            }
            return error;
        }

        public static bool IsNumeric(object? value)
        {
            return (value is Byte ||
                    value is Int16 ||
                    value is Int32 ||
                    value is Int64 ||
                    value is SByte ||
                    value is UInt16 ||
                    value is UInt32 ||
                    value is UInt64 ||
                    value is BigInteger ||
                    value is Decimal ||
                    value is Double ||
                    value is Single);
        }

        /// <summary>
        /// Clears the elearned state.
        /// </summary>
        public void ClearState()
        {
            patternNZHistory.Clear();
            weightMatrix.Clear();
            actualValues.Clear();
        }

    }

}