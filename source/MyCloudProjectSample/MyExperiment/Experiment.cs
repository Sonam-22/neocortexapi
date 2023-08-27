﻿using Azure.Storage.Queues;
using Azure.Storage.Queues.Models;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using MyCloudProject.Common;
using NeoCortexApi;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using System.Linq;
using System.IO;
using NeoCortexApi.Classifiers;
using System.Collections;

namespace MyExperiment
{
    /// <summary>
    /// This class implements the ML experiment that will run in the cloud. This is refactored code from my SE project.
    /// </summary>
    public class Experiment : IExperiment
    {
        private IStorageProvider storageProvider;

        private ILogger logger;

        private MyConfig config;

        private string projectName;

        public Experiment(IConfigurationSection configSection, IStorageProvider storageProvider, ILogger log, string projectName)
        {
            this.storageProvider = storageProvider;
            this.logger = log;

            config = new MyConfig();
            configSection.Bind(config);
            this.projectName = projectName;
        }

        public Task<IExperimentResult> Run(string inputFile)
        {
            // TODO read file

            var outputFile = "output.txt";
            var text = File.ReadAllText(inputFile, Encoding.UTF8);
            var trainingData = JsonSerializer.Deserialize<TrainingData>(text);
            var sequeneceAsString = trainingData.Sequences
                .Select(kv => $"{kv.Key} : {string.Join(",", kv.Value)}")
                .ToArray();
            sequeneceAsString.Prepend("Training Sequences");
            File.AppendAllLines(outputFile, sequeneceAsString);
            // YOU START HERE WITH YOUR SE EXPERIMENT!!!!

            MultiSequenceExperiment experiment = new(logger);

            ExperimentResult res = new ExperimentResult(this.config.GroupId, "1");

            res.StartTimeUtc = DateTime.UtcNow;

            // Run your experiment code here.

            // Train the model
            var predictor = experiment.Train(trainingData.Sequences);

            var acc = trainingData.Validation
                .Select(seq => PredictNextElement(predictor, seq, outputFile))
                .Average();

            res.Timestamp = DateTime.UtcNow;
            res.EndTimeUtc = DateTime.UtcNow;
            res.ExperimentId = projectName;
            var elapsedTime = res.EndTimeUtc - res.StartTimeUtc;
            res.DurationSec = (long)elapsedTime.GetValueOrDefault().TotalSeconds;
            res.OutputFilesProxy = new string[] { outputFile };
            res.InputFileUrl = inputFile;
            res.Description = "MultiSequence learning with sdr classifier";
            res.Name = "MultiSequence learning";
            res.Accuracy = Convert.ToSingle(acc);
            return Task.FromResult<IExperimentResult>(res);
        }

        

        /// <inheritdoc/>
        public async Task RunQueueListener(CancellationToken cancelToken)
        {
          
            QueueClient queueClient = new QueueClient(this.config.StorageConnectionString, this.config.Queue);

            
            while (cancelToken.IsCancellationRequested == false)
            {
                QueueMessage message = await queueClient.ReceiveMessageAsync();

                if (message != null)
                {
                    try
                    {

                        string msgTxt = Encoding.UTF8.GetString(message.Body.ToArray());

                        logger?.LogInformation($"Received the message {msgTxt}");

                        ExerimentRequestMessage request = JsonSerializer.Deserialize<ExerimentRequestMessage>(msgTxt);

                        var inputFile = await storageProvider.DownloadInputFile(request.InputFile);

                        ExperimentResult result = await Run(inputFile) as ExperimentResult;

                        await storageProvider.UploadResultFile("outputfile.txt", File.ReadAllBytes(result.OutputFilesProxy[0]));

                        await storageProvider.UploadExperimentResult(result);

                        await queueClient.DeleteMessageAsync(message.MessageId, message.PopReceipt);

                        File.Delete(result.OutputFilesProxy[0]);
                    }
                    catch (Exception ex)
                    {
                       logger?.LogError(ex, "Something went wrong while running the experiment");
                    }
                }
                else
                {
                    await Task.Delay(500);
                    logger?.LogTrace("Queue empty...");
                }
            }

            this.logger?.LogInformation("Cancel pressed. Exiting the listener loop.");
        }


        #region Private Methods

        private double PredictNextElement(Predictor predictor, double[] list, string outputFileName)
        {
            List<string> resultLines = new();

            AddAndLog(resultLines, "------------------------------");

            predictor.Reset();

            int totalPredictions = 0;
            int totalMatchCount = 0;
             
            foreach (var item in list)
            {
                AddAndLog(resultLines, $"--------------- Input {item} ---------------");

                var res = predictor.Predict(item);


                if (res.Count > 0)
                {
                    foreach (var pred in res)
                    {
                        AddAndLog(resultLines, $"{pred.PredictedInput} - {pred.Similarity}%");
                    }
                    var predictedSequence = res.First().PredictedInput.Split('_').First();
                    var predictedValue = res.First().PredictedInput.Split('-').Last();
                    AddAndLog(resultLines, $"Predicted Sequence: {predictedSequence}, predicted next element {predictedValue}");
                    totalMatchCount += 1;
                }
                else
                {
                    AddAndLog(resultLines, "Nothing predicted :(");
                }

                totalPredictions += 1;
            }

            var predictionAccuracy = (totalMatchCount * 100) / totalPredictions;

            AddAndLog(resultLines, "------------------------------");
            AddAndLog(resultLines, $"Prediction accuracy for {string.Join(",", list)} is {predictionAccuracy}");
            File.AppendAllLines(outputFileName, resultLines);

            return predictionAccuracy;
        }

        private void AddAndLog(List<string> resultLines, string line) {
            logger.LogInformation(line);
            resultLines.Add(line);
        }

        #endregion
    }
}
