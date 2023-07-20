using System;
using System.Collections.Generic;
using System.Diagnostics;
using NeoCortexApi;
using NeoCortexApi.Encoders;
using NeoCortexApi.Entities;
using NeoCortexApi.Network;
using System.Linq;
using SDRClassifier;

namespace MyExperiment
{
    public class MultiSequenceExperiment
    {

        /// <summary>
        /// Runs the learning of sequences.
        /// </summary>
        /// <param name="sequences">Dictionary of sequences. KEY is the sewuence name, the VALUE is th elist of element of the sequence.</param>
        public Predictor Train(Dictionary<string, List<double>> sequences)
        {
            Console.WriteLine($"Hello NeocortexApi! Experiment {nameof(MultiSequenceExperiment)}");

            int inputBits = 100;
            int numColumns = 1024;

            HtmConfig cfg = new HtmConfig(new int[] { inputBits }, new int[] { numColumns })
            {
                Random = new ThreadSafeRandom(42),

                CellsPerColumn = 25,
                GlobalInhibition = true,
                LocalAreaDensity = -1,
                NumActiveColumnsPerInhArea = 0.02 * numColumns,
                PotentialRadius = (int)(0.15 * inputBits),
                //InhibitionRadius = 15,

                MaxBoost = 10.0,
                DutyCyclePeriod = 25,
                MinPctOverlapDutyCycles = 0.75,
                MaxSynapsesPerSegment = (int)(0.02 * numColumns),

                ActivationThreshold = 15,
                ConnectedPermanence = 0.5,

                // Learning is slower than forgetting in this case.
                PermanenceDecrement = 0.25,
                PermanenceIncrement = 0.15,

                // Used by punishing of segments.
                PredictedSegmentDecrement = 0.1
            };

            double max = 20;

            Dictionary<string, object> settings = new Dictionary<string, object>()
            {
                { "W", 15},
                { "N", inputBits},
                { "Radius", -1.0},
                { "MinVal", 0.0},
                { "Resolution", 1.0},
                { "Periodic", false},
                { "Name", "scalar"},
                { "ClipInput", false},
                { "MaxVal", max}
            };

            EncoderBase encoder = new ScalarEncoder(settings);

            return RunExperiment(inputBits, cfg, encoder, sequences);
        }

        /// <summary>
        ///
        /// </summary>
        private Predictor RunExperiment(int inputBits, HtmConfig cfg, EncoderBase encoder, Dictionary<string, List<double>> sequences)
        {
            Stopwatch sw = new Stopwatch();
            sw.Start();

            int maxMatchCnt = 0;

            int step = 0;

            var mem = new Connections(cfg);

            bool isInStableState = false;

            var numUniqueInputs = GetNumberOfInputs(sequences);

            SDRClassifier<string, ComputeCycle> cls = new SDRClassifier<string, ComputeCycle>(new List<int>() { step }, 0.1, 0.3, 3, 1);

            CortexLayer<object, object> layer1 = new CortexLayer<object, object>("L1");

            TemporalMemory tm = new TemporalMemory();

            // For more information see following paper: https://www.scitepress.org/Papers/2021/103142/103142.pdf
            HomeostaticPlasticityController hpc = new HomeostaticPlasticityController(mem, numUniqueInputs * 150, (isStable, numPatterns, actColAvg, seenInputs) =>
            {
                if (isStable)
                    // Event should be fired when entering the stable state.
                    Debug.WriteLine($"STABLE: Patterns: {numPatterns}, Inputs: {seenInputs}, iteration: {seenInputs / numPatterns}");
                else
                    // Ideal SP should never enter unstable state after stable state.
                    Debug.WriteLine($"INSTABLE: Patterns: {numPatterns}, Inputs: {seenInputs}, iteration: {seenInputs / numPatterns}");

                // We are not learning in instable state.
                isInStableState = isStable;

                // Clear active and predictive cells.
                //tm.Reset(mem);
            }, numOfCyclesToWaitOnChange: 50);


            SpatialPoolerMT sp = new SpatialPoolerMT(hpc);
            sp.Init(mem);
            tm.Init(mem);

            // Please note that we do not add here TM in the layer.
            // This is omitted for practical reasons, because we first eneter the newborn-stage of the algorithm
            // In this stage we want that SP get boosted and see all elements before we start learning with TM.
            // All would also work fine with TM in layer, but it would work much slower.
            // So, to improve the speed of experiment, we first ommit the TM and then after the newborn-stage we add it to the layer.
            layer1.HtmModules.Add("encoder", encoder);
            layer1.HtmModules.Add("sp", sp);

            //double[] inputs = inputValues.ToArray();
            int[] prevActiveCols = new int[0];

            int cycle = 0;
            int matches = 0;

            var lastPredictedValues = new List<string>(new string[] { "0" });

            int maxCycles = 3500;

            //
            // Training SP to get stable. New-born stage.
            //

            for (int i = 0; i < maxCycles && isInStableState == false; i++)
            {
                matches = 0;

                cycle++;

                Debug.WriteLine($"-------------- Newborn Cycle {cycle} ---------------");

                foreach (var inputs in sequences)
                {
                    foreach (var input in inputs.Value)
                    {
                        Debug.WriteLine($" -- {inputs.Key} - {input} --");

                        var lyrOut = layer1.Compute(input, true);

                        if (isInStableState)
                            break;
                    }

                    if (isInStableState)
                        break;
                }
            }



            // We activate here the Temporal Memory algorithm.
            layer1.HtmModules.Add("tm", tm);

            int recordCounter = 0;

            var buckets = new Dictionary<string, int>();

            //
            // Loop over all sequences.
            foreach (var sequenceKeyPair in sequences)
            {
                Debug.WriteLine($"-------------- Sequences {sequenceKeyPair.Key} ---------------");

                int maxPrevInputs = sequenceKeyPair.Value.Count - 1;

                List<string> previousInputs = new List<string> { "-1.0" };


                // Set on true if the system has learned the sequence with a maximum acurracy.
                bool isLearningCompleted = false;

                //
                // Now training with SP+TM. SP is pretrained on the given input pattern set.
                for (int i = 0; i < maxCycles; i++)
                {
                    matches = 0;

                    cycle++;

                    Debug.WriteLine("");

                    Debug.WriteLine($"-------------- Cycle {cycle} ---------------");
                    Debug.WriteLine("");

                    foreach (var input in sequenceKeyPair.Value)
                    {
                        Debug.WriteLine($"-------------- {input} ---------------");

                        var lyrOut = layer1.Compute(input, true) as ComputeCycle;

                        var activeColumns = layer1.GetResult("sp") as int[];


                        previousInputs.Add(input.ToString());
                        if (previousInputs.Count > (maxPrevInputs + 1))
                            previousInputs.RemoveAt(0);

                        if (previousInputs.Count < maxPrevInputs)
                            continue;

                        string key = GetKey(previousInputs, input, sequenceKeyPair.Key);

                        var bucketIndex = (encoder as ScalarEncoder).GetBucketIndex(input)
                            .GetValueOrDefault(0);
                        buckets[key] = bucketIndex;
                        if (buckets.Where(kv => kv.Value == bucketIndex).Count() > 1)
                        {
                            while (buckets.ContainsValue(bucketIndex)) { bucketIndex++; }
                            buckets[key] = bucketIndex;
                        }


                        List<Cell> actCells;

                        if (lyrOut.ActiveCells.Count == lyrOut.WinnerCells.Count)
                        {
                            actCells = lyrOut.ActiveCells;
                        }
                        else
                        {
                            actCells = lyrOut.WinnerCells;
                        }

                        //Creates required sdr classifier inputs
                        var learnInfo = new SDRClassifierInput<string>
                        {
                            BucketIndex = new double[] { bucketIndex },
                            RecordNumber = recordCounter++
                        };

                        cls.Learn(key, actCells.ToArray(), learnInfo);

                        Debug.WriteLine($"Col  SDR: {Helpers.StringifyVector(lyrOut.ActivColumnIndicies)}");
                        Debug.WriteLine($"Cell SDR: {Helpers.StringifyVector(actCells.Select(c => c.Index).ToArray())}");

                        //
                        // If the list of predicted values from the previous step contains the currently presenting value,
                        // we have a match.
                        if (lastPredictedValues.Contains(key))
                        {
                            matches++;
                            Debug.WriteLine($"Match. Actual value: {key} - Predicted value: {lastPredictedValues.FirstOrDefault(key)}.");
                        }
                        else
                        {
                            Debug.WriteLine($"Missmatch! Actual value: {key} - Predicted values: {String.Join(',', lastPredictedValues)}");
                        }
                            
                        if (lyrOut.PredictiveCells.Count > 0)
                        {
                         
                            var indices = ToIndices((List<Cell>)lyrOut.PredictiveCells);
                                
                            //Creates required sdr classifier inputs for prediction
                            var predictInfo = new SDRClassifierInput<string>
                            {
                                ActValues = new string[] { key },
                                BucketIndex = new double[] { bucketIndex },
                                RecordNumber = recordCounter++
                            };
                            var predictedInputValues = cls.GetPredictedInputValues(indices, 3, predictInfo);

                            foreach (var item in predictedInputValues)
                            {
                                Debug.WriteLine($"Current Input: {input} \t| Predicted Input: {item.PredictedInput} - {item.Similarity}%");
                            }

                            lastPredictedValues = predictedInputValues.Select(v => v.PredictedInput).ToList();
                        }
                        else
                        {
                            Debug.WriteLine($"NO CELLS PREDICTED for next cycle.");
                            lastPredictedValues = new List<string>();
                        }
                    }

                    // The first element (a single element) in the sequence cannot be predicted
                    double maxPossibleAccuraccy = (double)((double)sequenceKeyPair.Value.Count - 1) / (double)sequenceKeyPair.Value.Count * 100.0;

                    double accuracy = (double)matches / (double)sequenceKeyPair.Value.Count * 100.0;

                    Debug.WriteLine($"Cycle: {cycle}\tMatches={matches} of {sequenceKeyPair.Value.Count}\t {accuracy}%");

                    if (accuracy >= maxPossibleAccuraccy)
                    {
                        maxMatchCnt++;
                        Debug.WriteLine($"100% accuracy reched {maxMatchCnt} times.");

                        //
                        // Experiment is completed if we are 30 cycles long at the 100% accuracy.
                        if (maxMatchCnt >= 30)
                        {
                            sw.Stop();
                            Debug.WriteLine($"Sequence learned. The algorithm is in the stable state after 30 repeats with with accuracy {accuracy} of maximum possible {maxMatchCnt}. Elapsed sequence {sequenceKeyPair.Key} learning time: {sw.Elapsed}.");
                            isLearningCompleted = true;
                            break;
                        }
                    }
                    else if (maxMatchCnt > 0)
                    {
                        Debug.WriteLine($"At 100% accuracy after {maxMatchCnt} repeats we get a drop of accuracy with accuracy {accuracy}. This indicates instable state. Learning will be continued.");
                        maxMatchCnt = 0;
                    }

                    // This resets the learned state, so the first element starts allways from the beginning.
                    tm.Reset(mem);
                }

                if (isLearningCompleted == false)
                    throw new Exception($"The system didn't learn with expected acurracy!");
            }

            Debug.WriteLine("------------ END ------------");

            return new Predictor(layer1, mem, cls);
        }

        private int[] ToIndices(List<Cell> cells)
        {
            return cells
                 .Select(c => c.Index)
                 .ToArray();
        }


        /// <summary>
        /// Gets the number of all unique inputs.
        /// </summary>
        /// <param name="sequences">Alle sequences.</param>
        /// <returns></returns>
        private int GetNumberOfInputs(Dictionary<string, List<double>> sequences)
        {
            int num = 0;

            foreach (var inputs in sequences)
            {
                //num += inputs.Value.Distinct().Count();
                num += inputs.Value.Count;
            }

            return num;
        }


        /// <summary>
        /// Constracts the unique key of the element of an sequece. This key is used as input for HtmClassifier.
        /// It makes sure that alle elements that belong to the same sequence are prefixed with the sequence.
        /// The prediction code can then extract the sequence prefix to the predicted element.
        /// </summary>
        /// <param name="prevInputs"></param>
        /// <param name="input"></param>
        /// <param name="sequence"></param>
        /// <returns></returns>
        private static string GetKey(List<string> prevInputs, double input, string sequence)
        {
            string key = String.Empty;

            for (int i = 0; i < prevInputs.Count; i++)
            {
                if (i > 0)
                    key += "-";

                key += prevInputs[i];
            }

            return $"{sequence}_{key}";
        }
    }
}

