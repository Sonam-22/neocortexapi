namespace SDRClassifier
{
    using NumSharp;
    using System;
    using System.Collections.Generic;
    using System.Data;
    using System.Linq;

    public class SDRClassifier
    {

        public int version = 1;
        public List<int> steps;
        public double alpha;
        public double actValueAlpha;
        public double verbosity;
        private int maxSteps;
        private LinkedList<Tuple<int, List<int>>> patternNZHistory = new();
        private int maxInputIdx;
        private int maxBucketIdx;
        private Dictionary<int, NDArray> weightMatrix = new();
        private List<object> actualValues = new();

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
            // History of the last _maxSteps activation patterns. We need to keep
            // these so that we can associate the current iteration's classification
            // with the activationPattern from N steps ago
            //this._patternNZHistory = this.deque(maxlen: this._maxSteps);
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
            this.actualValues = new List<object>();
            // Set the version to the latest version.
            // This is used for serialization/deserialization
            this.version = version;
        }

        private List<int> deque(int maxlen)
        {
            throw new NotImplementedException();
        }

        public void Compute(
                int recordNum,
                List<int> patternNZ,
                Dictionary<string, object> classification,
                bool learn,
                bool infer)
        {
            // int nSteps;
            int numCategory;
            List<object> actValueList;
            object bucketIdxList;

            if (this.verbosity >= 1)
            {
                Console.WriteLine("  learn: {0}", learn);
                Console.WriteLine("  recordNum {0}:", recordNum);
                Console.WriteLine("  patternNZ {0}: {1}", patternNZ.Count, string.Join(",", patternNZ.ToArray()));
                Console.WriteLine("  classificationIn: {0}", classification);
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
                this.patternNZHistory.AddLast(new LinkedListNode<Tuple<int, List<int>>>(new Tuple<int, List<int>>(recordNum, patternNZ)));
            }

            // To allow multi-class classification, we need to be able to run learning
            // without inference being on. So initialize retval outside
            // of the inference block.
            var retval = new Dictionary<string, object>();
            // Update maxInputIdx and augment weight matrix with zero padding
            if (patternNZ.Max() > this.maxInputIdx)
            {
                var newMaxInputIdx = patternNZ.Max();
                foreach (var nSteps in this.steps)
                {
                    this.weightMatrix[nSteps] = np.concatenate(((NDArray, NDArray))(this.weightMatrix[nSteps], np.zeros(shape: (newMaxInputIdx - this.maxInputIdx, this.maxBucketIdx + 1))), axis: 0);
                }
                this.maxInputIdx = Convert.ToInt32(newMaxInputIdx);
            }
            // Get classification info
            if (classification is not null)
            {
                if (classification["bucketIdx"].GetType() != typeof(List<>))
                {
                    bucketIdxList = new List<object> { classification["bucketIdx"] };
                    actValueList = new List<object> { classification["actValue"] };
                    numCategory = 1;
                }
                else
                {
                    bucketIdxList = classification["bucketIdx"];
                    actValueList = (List<object>)classification["actValue"];
                    numCategory = ((List<object>)classification["bucketIdx"]).Count();
                }
            }
            else
            {
                if (learn)
                {
                    throw new InvalidDataException("classification cannot be None when learn=True");
                }
                actValueList = null;
                bucketIdxList = null;
            }

            if (infer)
            {
                retval = this.Infer(patternNZ, actValueList);
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
        public Dictionary<string, object> Infer(List<int> patternNZ, List<object>? actValueList)
        {
            object defaultValue;
            /**
             * Return value dict. For buckets which we don't have an actual value
             * for yet, just plug in any valid actual value. It doesn't matter what
             * we use because that bucket won't have non-zero likelihood anyways.
             * NOTE: If doing 0-step prediction, we shouldn't use any knowledge
             * of the classification input during inference.
            */
            if (this.steps[0] == 0 || actValueList == null)
            {
                defaultValue = 0;
            }
            else
            {
                defaultValue = actValueList[0];
            }
            var actValues = (from x in this.actualValues
                             select x != null ? x : defaultValue).ToList();
            var retval = new Dictionary<string, object> { { "actualValues", actValues } };
            foreach (var nSteps in this.steps)
            {
                var predictDist = this.InferSingleStep(patternNZ, this.weightMatrix[nSteps]);
                retval[nSteps.ToString()] = predictDist;
            }
            return retval;
        }

        /// <summary>
        /// Perform inference for a single step. Given an SDR input and a weight matrix, return a predicted distribution.
        /// </summary>
        /// <param name="patternNZ">list of the active indices from the output below</param>
        /// <param name="weightMatrix">Multidimentional array of the weight matrix</param>
        /// <returns>
        /// Multidimentional array of the predicted class label distribution
        /// </returns>
        public double[] InferSingleStep(List<int> patternNZ, NDArray weightMatrix)
        {
            double[] outputActivation = new double[weightMatrix.shape[0]];
            // Calculate activation values by summing the weighed inputs
            for (int row = 0; row < weightMatrix.shape[0]; row++)
            {
                for (int col = 0; col < patternNZ.Count; col++)
                {
                    outputActivation[row] += weightMatrix[row, patternNZ[col]];
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
    }

}