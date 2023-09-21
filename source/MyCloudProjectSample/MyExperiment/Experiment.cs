using Azure.Storage.Queues;
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
                
            var outputFile = "output.txt";
            // Reads the input file specified in queue message
            var text = File.ReadAllText(inputFile, Encoding.UTF8);
            var trainingData = JsonSerializer.Deserialize<TrainingData>(text);
            var sequeneceAsString = trainingData.Sequences
                .Select(kv => $"{kv.Key} : {string.Join(",", kv.Value)}")
                .ToArray();
            sequeneceAsString.Prepend("Training Sequences");
            // Prints the training the sequences into output file
            File.AppendAllLines(outputFile, sequeneceAsString);
            // Creates instance of the multisequence experiment and starts the experiment.
            MultiSequenceExperiment experiment = new(logger);
            // Experiment result instance.
            ExperimentResult res = new ExperimentResult(this.config.GroupId, "1");

            res.StartTimeUtc = DateTime.UtcNow;

            // Train the model
            var predictor = experiment.Train(trainingData.Sequences);
            
            // Calculate the average accuracy for a set of predictions 
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
            res.Accuracy = Convert.ToSingle(acc);
            return Task.FromResult<IExperimentResult>(res);
        }

        

        /// <summary>
        /// This method starts the experiment and waits for a new experiment trigger message.
        /// </summary>
        /// <param name="cancelToken">Token required to interrupt the running experiment and release resources</param>
        /// <returns></returns>
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
                        // Updates the experiment data
                        result.Name = request.Name;
                        result.Description = request.Description;
                        result.ExperimentId = request.ExperimentId;
                        // Upload the output result to blob container.
                        await storageProvider.UploadResultFile("outputfile.txt", File.ReadAllBytes(result.OutputFilesProxy[0]));
                        // Write the experiment result entity into azure table. 
                        await storageProvider.UploadExperimentResult(result);
                        // Deletes the queue message when experiment completes.    
                        await queueClient.DeleteMessageAsync(message.MessageId, message.PopReceipt);
                        // Deleted the temporary out file generated.    
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
        /// <summary>
        /// Predicts the next input element and writes the result into the output file.
        /// </summary>
        /// <param name="predictor">Predictor instance from neocortex api</param>
        /// <param name="list">List of inputs for prediction of next input element</param>
        /// <param name="outputFileName">Name of the output file.</param>
        /// <returns></returns>
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

            // Calculate the accuracy based on the matching results.    
            var predictionAccuracy = (totalMatchCount * 100) / totalPredictions;

            AddAndLog(resultLines, "------------------------------");
            AddAndLog(resultLines, $"Prediction accuracy for {string.Join(",", list)} is {predictionAccuracy}");
            File.AppendAllLines(outputFileName, resultLines);

            return predictionAccuracy;
        }
        /// <summary>
        /// Logs the comment as info and append it to the output file.
        /// </summary>
        /// <param name="resultLines">Result line accumulator list</param>
        /// <param name="line">Current line to be added</param>
        private void AddAndLog(List<string> resultLines, string line) {
            logger.LogInformation(line);
            resultLines.Add(line);
        }

        #endregion
    }
}
