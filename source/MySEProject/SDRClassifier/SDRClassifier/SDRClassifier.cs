namespace SDRClassifier
{
    using static System.Text.Json.JsonSerializer;
    using NumSharp;
    using System;
    using System.Collections.Generic;
    using System.Data;
    using System.Linq;
    using Microsoft.VisualBasic;
    using static System.Runtime.InteropServices.JavaScript.JSType;
    using System.Net.Sockets;
    using System.Reflection.Metadata;
    using System.Reflection;

    public class SDRClassifier
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
        private List<double?> actualValues;

        public SDRClassifier(List<int> steps, double alpha, double actValueAlpha, double verbosity, int version)
        {
            if (steps.Count == 0)
            {
                Console.WriteLine("steps cannot be empty");
            }
            if (alpha < 0)
            {
                Console.WriteLine("alpha (learning rate) must be a positive number");
            }
            if (actValueAlpha < 0 || actValueAlpha >= 1)
            {
                Console.WriteLine("actValueAlpha be a number between 0 and 1");
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
            this.actualValues = new List<double?> { null };
            // Set the version to the latest version.
            // This is used for serialization/deserialization
            this.version = version;
        }

        /// <summary>
        ///  Process one input sample.
        ///  recordNum: Record number of this input pattern. Record numbers
        ///             normally increase sequentially by 1 each time unless there are missing
        ///             records in the dataset.Knowing this information insures that we don't get
        ///             confused by missing records.
        /// patternNZ: List of the active indices from the output below.When the
        ///            input is from TemporalMemory, this list should be the indices of the active cells.
        /// classification: Dict of the classification information where:
        /// bucketIdx: list of indices of the encoder bucket
        /// actValue: list of actual values going into the encoder
        ///           Classification could be None for inference mode.
        /// learn: (bool) if true, learn this sample
        /// infer: (bool) if true, perform inference
        /// return: Dict containing inference results, there is one entry for each
        ///         step in this.steps, where the key is the number of steps, and
        ///         the value is an array containing the relative likelihood for
        ///         each bucketIdx starting from bucketIdx 0.
        ///         There is also an entry containing the average actual value to
        ///         use for each bucket. The key is 'actualValues'.
        ///

        public Dictionary<string, double[]> Compute(
                int recordNum,
                List<int> patternNZ,
                Dictionary<string, double[]> classification,
                bool learn,
                bool infer)
        {
            //int nSteps;
            int numCategory = 0;
            double[] actValueList = new double[1];
            double[] bucketIdxList = new double[1];

            if (this.verbosity >= 1)
            {
                Console.WriteLine("learn: {0}", learn);
                Console.WriteLine("recordNum {0}:", recordNum);
                Console.WriteLine("patternNZ {0}: {1}", patternNZ.Count, string.Join(",", patternNZ.ToArray()));
                Console.WriteLine("classificationIn: {0}", Serialize(classification.ToList()));
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
            var retval = new Dictionary<string, double[]>();
            // Update maxInputIdx and augment weight matrix with zero padding
            if (patternNZ.Max() > this.maxInputIdx)
            {
                var newMaxInputIdx = patternNZ.Max();
                foreach (var step in this.steps)
                {
                    this.weightMatrix[step] = np.concatenate(((NDArray, NDArray))(this.weightMatrix[step], np.zeros(shape: (newMaxInputIdx - this.maxInputIdx, this.maxBucketIdx + 1))), axis: 0);
                }
                this.maxInputIdx = Convert.ToInt32(newMaxInputIdx);
            }
            // Get classification info
            if (classification is not null)
            {
                bucketIdxList = classification["bucketIdx"];
                actValueList = classification["actValue"];
                numCategory = classification["bucketIdx"].Length;
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

            if (learn && classification != null && classification["bucketIdx"] != null)
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
                            var toUpdate = ((NDArray, NDArray))(this.weightMatrix[step], np.zeros(shape: (maxInputIdx + 1, bucketIdx - maxBucketIdx)));
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
                        this.actualValues[bucketIdx] = (1.0 - this.actValueAlpha) * (this.actualValues[bucketIdx]) + this.actValueAlpha * actValue;
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

            // Verbose print
            if (infer && this.verbosity >= 1)
            {
                Console.WriteLine("inference: combined bucket likelihoods:");
                Console.WriteLine("actual bucket values: {0}", string.Join(",", retval["actualValues"]));
                foreach (var keyValue in retval)
                {
                    var nSteps = keyValue.Key;
                    var votes = keyValue.Value;
                    if (nSteps == "actualValues")
                    {
                        continue;
                    }
                    Console.WriteLine("{0} steps: {1}", nSteps, string.Join(",", votes));
                    var bestBucketIdx = np.array((double[])votes).argmax();
                    Console.WriteLine("most likely bucket idx: {0}, value: {1}", bestBucketIdx, retval["actualValues"][bestBucketIdx]);
                }
            }
            return retval;
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
        public Dictionary<string, double[]> Infer(List<int> patternNZ, double[]? actValueList)
        {
            double defaultValue = 0.0;
            /**
             * Return value dict. For buckets which we don't have an actual value
             * for yet, just plug in any valid actual value. It doesn't matter what
             * we use because that bucket won't have non-zero likelihood anyways.
             * NOTE: If doing 0-step prediction, we shouldn't use any knowledge
             * of the classification input during inference.
            */
            if (this.steps[0] == 0 || actValueList == null)
            {
                defaultValue = 0.0;
            }
            else
            {
                defaultValue = actValueList[0];
            }
            var actValues = this.actualValues
              .ConvertAll<double>(item => item ?? defaultValue)
              .ToArray();
            var retval = new Dictionary<string, double[]> { { "actualValues", actValues } };
            foreach (var nSteps in this.steps)
            {
                retval[nSteps.ToString()] = InferSingleStep(patternNZ, this.weightMatrix[nSteps]);
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

        //     Calculate error signal
        //     :param bucketIdxList: list of encoder buckets
        //     :return: dict containing error. The key is the number of steps
        //      The value is array of error at the output layer
        //     
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
                    var predictDist = np.array<double>(inferred);
                    error[nSteps] = targetDist - predictDist;
                }
            }
            return error;
        }
    }

}