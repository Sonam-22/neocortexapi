using System;
namespace SDRClassifier
{
	public class SDRClassification<TIN>
	{
		public Dictionary<string, double[]> Classifications { get; set; } = new();
        public TIN[] ActValues { get; set; } = Array.Empty<TIN>();
    }
}

