using Azure;
using Azure.Data.Tables;
using Azure.Storage.Blobs;
using Microsoft.Extensions.Configuration;
using MyCloudProject.Common;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace MyExperiment
{
    public class AzureStorageProvider : IStorageProvider
    {
        private MyConfig config;

        public AzureStorageProvider(IConfigurationSection configSection)
        {
            config = new MyConfig();
            configSection.Bind(config);
        }

        public async Task<string> DownloadInputFile(string fileName)
        {
            BlobContainerClient container = new BlobContainerClient(config.StorageConnectionString, config.TrainingContainer);
            await container.CreateIfNotExistsAsync();

            // Get a reference to a blob named "sample-file"
            BlobClient blob = container.GetBlobClient(fileName);

            await blob.DownloadToAsync(fileName);

            return fileName;
        }

        public async Task UploadExperimentResult(IExperimentResult result)
        {
            var client = new TableClient(this.config.StorageConnectionString, this.config.ResultTable);

            await client.CreateIfNotExistsAsync();

            await client.UpsertEntityAsync((ExperimentResult)result);

        }

        public async Task<byte[]> UploadResultFile(string fileName, byte[] data)
        {
            BlobContainerClient container = new BlobContainerClient(config.StorageConnectionString, config.ResultContainer);

            await container.UploadBlobAsync(fileName, BinaryData.FromBytes(data));

            return data;
        }

    }


}
