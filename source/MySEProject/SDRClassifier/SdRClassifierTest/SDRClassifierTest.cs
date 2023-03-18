namespace SdRClassifierTest
{
    using SDRClassifier;
    using NumSharp;

    [TestClass]
    public class SDRClassifierTest
    {

        [TestMethod]
        public void TestSingleBucketValue()
        {
            var classifier = new SDRClassifier(new List<int>() { 1 }, 0.001, 0.3, 0.0, 1);

            // Enough times to perform Inference and expect high likelihood for prediction.
            Dictionary<string, double[]> retVal = new();
            for (int recordNum = 0; recordNum < 10; recordNum++)
            {
                retVal = compute(classifier, recordNum, new List<int> { 1, 5 }, new double[] { 0 }, new double[] { 10 });
            }
            Assert.AreEqual(retVal["actualValues"][0], 10.0);
            Assert.AreEqual(retVal["1"][0], 1.0);
        }

        [TestMethod]
        public void TestMultipleBucketValues()
        {
            var classifier = new SDRClassifier(new List<int>() { 1 }, 0.001, 0.3, 0.0, 2);

            // Enough times to perform Inference and expect high likelihood for prediction.
            Dictionary<string, double[]> retVal = new();
            for (int recordNum = 0; recordNum < 10; recordNum++)
            {
                retVal = compute(classifier, recordNum, new List<int> { 1, 5 }, new double[] { 0, 1 }, new double[] { 10, 20 });
            }

            Assert.AreEqual(retVal["1"][0], 0.5);
            Assert.AreEqual(retVal["1"][1], 0.5);
            Assert.AreEqual(retVal["actualValues"][0], 10.0);
            Assert.AreEqual(retVal["actualValues"][1], 20.0);
        }

        private Dictionary<string, double[]> compute(SDRClassifier classifier,
                                                 int recordNum,
                                                 List<int> pattern,
                                                 double[] bucket,
                                                 double[] value)
        {

            Dictionary<string, double[]> classification = new();
            classification.Add("bucketIdx", bucket);
            classification.Add("actValue", value);
            return classifier.Compute(recordNum, pattern, classification, true, true);
        }
    }

}