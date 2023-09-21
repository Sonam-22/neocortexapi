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
        /// <summary>
        /// Intatiates the storage provider
        /// </summary>
        /// <param name="configSection"Reference to configuration</param>
        public AzureStorageProvider(IConfigurationSection configSection)
        {
            config = new MyConfig();
            configSection.Bind(config);
        }
        
        /// <summary>
        /// Dowload the file and returns the file path relative the storage container.
        /// </summary>
        /// <param name="fileName">Name of downloaded file.</param>
        /// <returns>File path</returns>
        public async Task<string> DownloadInputFile(string fileName)
        {
            BlobContainerClient container = new BlobContainerClient(config.StorageConnectionString, config.TrainingContainer);
            await container.CreateIfNotExistsAsync();

            // Get a reference to a blob named "sample-file"
            BlobClient blob = container.GetBlobClient(fileName);
            // Downloads the file asynchrounously
            await blob.DownloadToAsync(fileName);

            return fileName;
        }
        /// <summary>
        /// Writes the experiment result entity into azure cloud result table 
        /// </summary>
        /// <param name="result">Result entity</param>
        public async Task UploadExperimentResult(IExperimentResult result)
        {
            // Create the table client.
            var client = new TableClient(this.config.StorageConnectionString, this.config.ResultTable);
            // Create the table if it does not exsists.
            await client.CreateIfNotExistsAsync();
            // Cast the result into ExperimentResult
            var temp = (ExperimentResult)result;
            // Write the entity into the azure table. This is upsert operation based on replace command. This means
            // that it will replace any existing entity with the newest result. This is done to save the resources and costs on cloud.
            await client.UpsertEntityAsync(temp, TableUpdateMode.Replace);
        }
       
       /// <summary>
       /// Uploads the result file to the blob container.
       /// </summary>
       /// <param name="fileName">Name of the file</param>
       /// <param name="data">Data as bytes to upload</param>
       /// <returns>Returns the file as bytes</returns>
        public async Task<byte[]> UploadResultFile(string fileName, byte[] data)
        {
            // Create the blob container client from provided connection string and result container.
            BlobContainerClient container = new BlobContainerClient(config.StorageConnectionString, config.ResultContainer);
            // Create the blob container if it does not exists.
            await container.CreateIfNotExistsAsync();
            // Delete the existing blob with given file name, if any.
            await container.DeleteBlobIfExistsAsync(fileName);
            // Read the output file as bytes and upload it to the cloud.
            await container.UploadBlobAsync(fileName, BinaryData.FromBytes(data));

            return data;
        }

    }

}
