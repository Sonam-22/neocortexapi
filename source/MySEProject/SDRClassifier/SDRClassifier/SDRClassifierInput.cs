using System;
namespace SDRClassifier
{
	public class SDRClassifierInput<TIN>
    {
        public TIN[] ActValues { get; set; } = Array.Empty<TIN>();
        public double[] BucketIndex { get; set; } = Array.Empty<double>();
        public int RecordNumber { get; set; } = 0;
        public int Step { get; set; } = 0;
    }
}

