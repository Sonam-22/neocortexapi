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

        public Experiment(IConfigurationSection configSection, IStorageProvider storageProvider, ILogger log)
        {
            this.storageProvider = storageProvider;
            this.logger = log;

            config = new MyConfig();
            configSection.Bind(config);
        }

        public Task<IExperimentResult> Run(string inputFile)
        {
            // TODO read file

            var sequences = new Dictionary<string, List<double>>
            {
                { "S1", new List<double>(new double[] { 0.0, 1.0, 2.0, 3.0, 4.0, 2.0, 5.0, }) },
                { "S2", new List<double>(new double[] { 8.0, 1.0, 2.0, 9.0, 10.0, 7.0, 11.00 }) }
            };

            var predictionInputs = new List<double[]>() {
                new double[] { 1.0, 2.0, 3.0, 4.0, 2.0, 5.0 },
                new double[] { 2.0, 3.0, 4.0 },
                new double[] { 8.0, 1.0, 2.0 }
            };


            // YOU START HERE WITH YOUR SE EXPERIMENT!!!!

            MultiSequenceExperiment experiment = new();

            ExperimentResult res = new ExperimentResult(this.config.GroupId, null);

            res.StartTimeUtc = DateTime.UtcNow;

            // Run your experiment code here.

            // Train the model
            var predictor = experiment.Train(sequences);

            predictionInputs.ForEach(seq => PredictNextElement(predictor, seq));

            return Task.FromResult<IExperimentResult>(res); // TODO...
        }

        

        /// <inheritdoc/>
        public async Task RunQueueListener(CancellationToken cancelToken)
        {
            ExperimentResult res = new ExperimentResult("damir", "123")
            {
                //Timestamp = DateTime.SpecifyKind(DateTime.UtcNow, DateTimeKind.Utc),
                
                Accuracy = (float)0.5,
            };

            await storageProvider.UploadExperimentResult(res);


            QueueClient queueClient = new QueueClient(this.config.StorageConnectionString, this.config.Queue);

            
            while (cancelToken.IsCancellationRequested == false)
            {
                QueueMessage message = await queueClient.ReceiveMessageAsync();

                if (message != null)
                {
                    try
                    {

                        string msgTxt = Encoding.UTF8.GetString(message.Body.ToArray());

                        this.logger?.LogInformation($"Received the message {msgTxt}");

                        ExerimentRequestMessage request = JsonSerializer.Deserialize<ExerimentRequestMessage>(msgTxt);

                        var inputFile = await this.storageProvider.DownloadInputFile(request.InputFile);

                        IExperimentResult result = await this.Run(inputFile);

                        //TODO. do serialization of the result.
                        await storageProvider.UploadResultFile("outputfile.txt", null);

                        await storageProvider.UploadExperimentResult(result);

                        await queueClient.DeleteMessageAsync(message.MessageId, message.PopReceipt);
                    }
                    catch (Exception ex)
                    {
                        this.logger?.LogError(ex, "TODO...");
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

        private void PredictNextElement(Predictor predictor, double[] list)
        {
            Debug.WriteLine("------------------------------");

            foreach (var item in list)
            {
                Debug.WriteLine($"---------------{item}---------------");

                var res = predictor.Predict(item);

                if (res.Count > 0)
                {
                    foreach (var pred in res)
                    {
                        Debug.WriteLine($"{pred.PredictedInput} - {pred.Similarity}%");
                    }

                    var predictedSequence = res.First().PredictedInput.Split('_').First();
                    var predictedValue = res.First().PredictedInput.Split('-').Last();
                    Debug.WriteLine($"Predicted Sequence: {predictedSequence}, predicted next element {predictedValue}");
                }
                else
                    Debug.WriteLine("Nothing predicted :(");
            }

            Debug.WriteLine("------------------------------");
        }


        #endregion
    }
}
