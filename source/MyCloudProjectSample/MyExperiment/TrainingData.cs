using System;
using System.Collections.Generic;

namespace MyExperiment
{
	public class TrainingData
	{
        public Dictionary<string, List<double>> Sequences { get; set; } = new();
        public List<double[]> Validation { get; set; } = new();
    }
}

