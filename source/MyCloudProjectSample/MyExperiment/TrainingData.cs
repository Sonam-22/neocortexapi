using System;
using System.Collections.Generic;

namespace MyExperiment
{
    /// <summary>
    /// Model for training data.
    /// </summary>
	public class TrainingData
	{
        public Dictionary<string, List<double>> Sequences { get; set; } = new();
        public List<double[]> Validation { get; set; } = new();
    }
}

