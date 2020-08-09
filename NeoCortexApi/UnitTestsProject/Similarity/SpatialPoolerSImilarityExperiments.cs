﻿// Copyright (c) Damir Dobric. All rights reserved.
// Licensed under the Apache License, Version 2.0. See LICENSE in the project root for license information.
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeoCortexApi;
using NeoCortexApi.Entities;
using NeoCortexApi.Utility;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using ImageBinarizer;
using System.Drawing;
using NeoCortex;
using NeoCortexApi.Network;
using LearningFoundation;
using System.Globalization;
using MLPerceptron;
using System.Linq;
using NeuralNet.MLPerceptron;
using Microsoft.ML;
using Microsoft.ML.Data;
using NeoCortexApi.DistributedCompute;

namespace UnitTestsProject
{
    [TestClass]
    [TestCategory("Experiment")]
    public class SpatialPoolerSimilarityExperiments
    {
        private const int OutImgSize = 1024;

        /// <summary>
        /// This test do spatial pooling and save hamming distance, active columns 
        /// and speed of processing in text files in Output directory.
        /// </summary>
        /// <param name="digit"></param>
        [TestMethod]
        [TestCategory("LongRunning")]
        //[DataRow("digit7")]
        //[DataRow("digit5")]
        [DataRow("Vertical")]
        //[DataRow("Horizontal")]
        public void SimilarityExperiment(string digit)
        {
            string trainingFolder = "Similarity\\TestFiles";
            int imgSize = 28;
            var colDims = new int[] { 64, 64 };
            int numOfActCols = colDims[0] * colDims[1];

            string TestOutputFolder = $"Output-{nameof(SimilarityExperiment)}";

            var trainingImages = Directory.GetFiles(trainingFolder, $"{digit}*.png");

            Directory.CreateDirectory($"{nameof(SimilarityExperiment)}");

            int counter = 0;
            var parameters = GetDefaultParams();
            //parameters.Set(KEY.DUTY_CYCLE_PERIOD, 20);
            //parameters.Set(KEY.MAX_BOOST, 1);
            //parameters.setInputDimensions(new int[] { imageSize[imSizeIndx], imageSize[imSizeIndx] });
            //parameters.setColumnDimensions(new int[] { topologies[topologyIndx], topologies[topologyIndx] });
            //parameters.setNumActiveColumnsPerInhArea(0.02 * numOfActCols);
            parameters.Set(KEY.NUM_ACTIVE_COLUMNS_PER_INH_AREA, 0.06 * 4096); // TODO. Experiment with different sizes
            parameters.Set(KEY.POTENTIAL_RADIUS, imgSize * imgSize);
            parameters.Set(KEY.POTENTIAL_PCT, 1.0);
            parameters.Set(KEY.GLOBAL_INHIBITION, true); // TODO: Experiment with local inhibition too. Note also the execution time of the experiment.

            // Num of active synapces in order to activate the column.
            parameters.Set(KEY.STIMULUS_THRESHOLD, 50.0);
            parameters.Set(KEY.SYN_PERM_INACTIVE_DEC, 0.008);
            parameters.Set(KEY.SYN_PERM_ACTIVE_INC, 0.05);

            parameters.Set(KEY.INHIBITION_RADIUS, (int)0.02 * imgSize * imgSize); // TODO. check if this has influence in a case of the global inhibition. ALso check how this parameter influences the similarity of SDR.

            parameters.Set(KEY.SYN_PERM_CONNECTED, 0.2);
            parameters.Set(KEY.MIN_PCT_OVERLAP_DUTY_CYCLES, 0.001);
            parameters.Set(KEY.MIN_PCT_ACTIVE_DUTY_CYCLES, 0.001);
            parameters.Set(KEY.DUTY_CYCLE_PERIOD, 1000);
            parameters.Set(KEY.MAX_BOOST, 100);
            parameters.Set(KEY.WRAP_AROUND, true);
            parameters.Set(KEY.SEED, 1969);
            parameters.setInputDimensions(new int[] { imgSize, imgSize });
            parameters.setColumnDimensions(colDims);

            bool isInStableState = false;

            var mem = new Connections();

            parameters.apply(mem);

            HomeostaticPlasticityActivator hpa = new HomeostaticPlasticityActivator(mem, trainingImages.Length * 50, (isStable, numPatterns, actColAvg, seenInputs) =>
            {
                // Event should only be fired when entering the stable state.
                // Ideal SP should never enter unstable state after stable state.
                Assert.IsTrue(isStable);
                Assert.IsTrue(numPatterns == trainingImages.Length);
                isInStableState = true;
                Debug.WriteLine($"Entered STABLE state: Patterns: {numPatterns}, Inputs: {seenInputs}, iteration: {seenInputs / numPatterns}");
            });

            SpatialPooler sp = new SpatialPoolerMT(hpa);

            sp.init(mem, UnitTestHelpers.GetMemory());

            int[] activeArray = new int[numOfActCols];

            string outFolder = $"{TestOutputFolder}\\{digit}";

            Directory.CreateDirectory(outFolder);

            string outputHamDistFile = $"{outFolder}\\digit{digit}_hamming.txt";

            string outputActColFile = $"{outFolder}\\digit{digit}_activeCol.txt";

            using (StreamWriter swHam = new StreamWriter(outputHamDistFile))
            {
                using (StreamWriter swActCol = new StreamWriter(outputActColFile))
                {
                    int cycle = 0;

                    Dictionary<string, int[]> sdrs = new Dictionary<string, int[]>();

                    while (!isInStableState)
                    {
                        foreach (var mnistImage in trainingImages)
                        {
                            FileInfo fI = new FileInfo(mnistImage);

                            string outputImage = $"{outFolder}\\digit_{digit}_cycle_{counter}_{fI.Name}";

                            string testName = $"{outFolder}\\digit_{digit}_{fI.Name}";

                            string inputBinaryImageFile = Helpers.BinarizeImage($"{mnistImage}", imgSize, testName);

                            // Read input csv file into array
                            int[] inputVector = NeoCortexUtils.ReadCsvIntegers(inputBinaryImageFile).ToArray();

                            int[] oldArray = new int[activeArray.Length];
                            List<double[,]> overlapArrays = new List<double[,]>();
                            List<double[,]> bostArrays = new List<double[,]>();

                            sp.compute(inputVector, activeArray, true);

                            if (isInStableState)
                            {
                                var activeCols = ArrayUtils.IndexWhere(activeArray, (el) => el == 1);

                                var distance = MathHelpers.GetHammingDistance(oldArray, activeArray, true);
                                //var similarity = MathHelpers.CalcArraySimilarity(oldArray, activeArray, true);
                                sdrs.Add(mnistImage, activeCols);
                                swHam.WriteLine($"{counter++}|{distance} ");

                                oldArray = new int[numOfActCols];
                                activeArray.CopyTo(oldArray, 0);

                                overlapArrays.Add(ArrayUtils.Make2DArray<double>(ArrayUtils.toDoubleArray(mem.Overlaps), colDims[0], colDims[1]));
                                bostArrays.Add(ArrayUtils.Make2DArray<double>(mem.BoostedOverlaps, colDims[0], colDims[1]));

                                var activeStr = Helpers.StringifyVector(activeArray);
                                swActCol.WriteLine("Active Array: " + activeStr);

                                int[,] twoDimenArray = ArrayUtils.Make2DArray<int>(activeArray, colDims[0], colDims[1]);
                                twoDimenArray = ArrayUtils.Transpose(twoDimenArray);
                                List<int[,]> arrays = new List<int[,]>();
                                arrays.Add(twoDimenArray);
                                arrays.Add(ArrayUtils.Transpose(ArrayUtils.Make2DArray<int>(inputVector, (int)Math.Sqrt(inputVector.Length), (int)Math.Sqrt(inputVector.Length))));

                                NeoCortexUtils.DrawBitmaps(arrays, outputImage, Color.Yellow, Color.Gray, OutImgSize, OutImgSize);
                                NeoCortexUtils.DrawHeatmaps(overlapArrays, $"{outputImage}_overlap.png", 1024, 1024, 150, 50, 5);
                                NeoCortexUtils.DrawHeatmaps(bostArrays, $"{outputImage}_boost.png", 1024, 1024, 150, 50, 5);
                            }

                            Debug.WriteLine($"Cycle {cycle++}");
                        }
                    }

                    CalculateResult(sdrs);
                }
            }
        }

        /// <summary>
        /// Calculate all required results.
        /// 1. Correlation matrix.
        ///    It cross compares all SDRs in the dictionary.
        /// 2. Writes out bitmaps by by cross compare that marks in the extra color non-overlapping bits between two comparing SDRs.
        /// </summary>
        /// <param name="sdrs"></param>
        private void CalculateResult(Dictionary<string, int[]> sdrs)
        {
            return;
        }



        #region Private Helpers

        internal static Parameters GetDefaultParams()
        {

            ThreadSafeRandom rnd = new ThreadSafeRandom(42);

            var parameters = Parameters.getAllDefaultParameters();
            parameters.Set(KEY.POTENTIAL_RADIUS, 10);
            parameters.Set(KEY.POTENTIAL_PCT, 0.75);
            parameters.Set(KEY.GLOBAL_INHIBITION, false);
            parameters.Set(KEY.LOCAL_AREA_DENSITY, -1);
            parameters.Set(KEY.NUM_ACTIVE_COLUMNS_PER_INH_AREA, 50.0);
            parameters.Set(KEY.STIMULUS_THRESHOLD, 0);
            parameters.Set(KEY.SYN_PERM_INACTIVE_DEC, 0.01);
            parameters.Set(KEY.SYN_PERM_ACTIVE_INC, 0.1);
            parameters.Set(KEY.SYN_PERM_CONNECTED, 0.1);
            parameters.Set(KEY.MIN_PCT_OVERLAP_DUTY_CYCLES, 0.001);
            parameters.Set(KEY.MIN_PCT_ACTIVE_DUTY_CYCLES, 0.001);
            //parameters.Set(KEY.WRAP_AROUND, false);
            parameters.Set(KEY.DUTY_CYCLE_PERIOD, 100);
            parameters.Set(KEY.MAX_BOOST, 10.0);
            parameters.Set(KEY.RANDOM, rnd);
            //int r = parameters.Get<int>(KEY.NUM_ACTIVE_COLUMNS_PER_INH_AREA);

            /*
            Random rnd = new Random(42);

            var parameters = Parameters.getAllDefaultParameters();
            parameters.Set(KEY.POTENTIAL_RADIUS, 16);
            parameters.Set(KEY.POTENTIAL_PCT, 0.85);
            parameters.Set(KEY.GLOBAL_INHIBITION, false);
            parameters.Set(KEY.LOCAL_AREA_DENSITY, -1.0);
            //parameters.Set(KEY.NUM_ACTIVE_COLUMNS_PER_INH_AREA, 3.0);
            parameters.Set(KEY.STIMULUS_THRESHOLD, 0.0);
            parameters.Set(KEY.SYN_PERM_INACTIVE_DEC, 0.01);
            parameters.Set(KEY.SYN_PERM_ACTIVE_INC, 0.1);
            parameters.Set(KEY.SYN_PERM_CONNECTED, 0.1);
            parameters.Set(KEY.MIN_PCT_OVERLAP_DUTY_CYCLES, 0.1);
            parameters.Set(KEY.MIN_PCT_ACTIVE_DUTY_CYCLES, 0.1);
            parameters.Set(KEY.DUTY_CYCLE_PERIOD, 10);
            parameters.Set(KEY.MAX_BOOST, 10.0);
            parameters.Set(KEY.RANDOM, rnd);
            //int r = parameters.Get<int>(KEY.NUM_ACTIVE_COLUMNS_PER_INH_AREA);
            */
            return parameters;
        }

        #endregion

    }

    //class InputData
    //{
    //    [ColumnName("PixelValues")]
    //    [VectorType(64)]
    //    public Boolean[] PixelValues;
    //}
}
