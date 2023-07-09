namespace SdRClassifierTest
{
    using SDRClassifier;
    using NumSharp;
    using System.Diagnostics;

    [TestClass]
    public class SDRClassifierTest
    {

        [TestMethod]
        public void TestSingleBucketValue()
        {
            var classifier = new SDRClassifier<double, double>(new List<int>() { 1 }, 0.001, 0.3, 3, 1);

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
            var classifier = new SDRClassifier<double, double>(new List<int>() { 1 }, 0.001, 0.3, 3, 2);

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

        [TestMethod]
        public void TestComputeSingleIteration()
        {
            var classifier = new SDRClassifier<double, double>(new List<int>() { 1 }, 0.001, 0.3, 3, 3);
            int recordNum = 0;

            Dictionary<string, double[]> result = compute(classifier, recordNum, new List<int> { 1, 5, 9 }, new double[] { 4 }, new double[] { 34.7 });

            Assert.AreEqual(result["actualValues"].Length, 1);
            Assert.AreEqual(result["actualValues"][0], 34.7);
        }

        [TestMethod]
        public void TestComputeDoubleIteration()
        {
            var classifier = new SDRClassifier<double, double>(new List<int>() { 1 }, 0.001, 0.3, 3, 3);
            int recordNum = 0;
            Dictionary<string, double[]> classification = new();
            classification.Add("bucketIdx", new double[] { 4 });
            classification.Add("actValue", new double[] { 34.7 });
            classifier.Compute(recordNum, new List<int> { 1, 5, 9 }, classification, true, true);
            Dictionary<string, double[]> result = classifier.Compute(++recordNum, new List<int> { 1, 5, 9 }, classification, true, true);

            Assert.AreEqual(result["actualValues"][4], 34.7);
        }

        [TestMethod]
        public void TestComputeMultipleEncoderPatterns()
        {
            var classifier = new SDRClassifier<double, double>(new List<int>() { 1 }, 1.0, 0.1, 3, 1);
            int recordNum = 0;

            compute(classifier, recordNum++, new List<int> { 1, 5, 9 }, new double[] { 4 }, new double[] { 34.7 });
            compute(classifier, recordNum++, new List<int> { 0, 6, 9, 11 }, new double[] { 5 }, new double[] { 41.7 });
            compute(classifier, recordNum++, new List<int> { 6, 9 }, new double[] { 5 }, new double[] { 44.9 });
            compute(classifier, recordNum++, new List<int> { 1, 5, 9 }, new double[] { 4 }, new double[] { 42.9 });

            Dictionary<string, double[]> result = compute(classifier, recordNum++, new List<int> { 1, 5, 9 }, new double[] { 4 }, new double[] { 34.7 });

            Assert.IsNotNull(result["1"]);
            Assert.IsNotNull(result["actualValues"]);
            Assert.AreEqual(result.Keys.Count(), 2);
            Assert.AreEqual(result["actualValues"][4], 35.520000457763672, 0.00001);
            Assert.AreEqual(result["actualValues"][5], 42.020000457763672, 0.00001);
            var resultStep1 = result["1"];
            Assert.AreEqual(resultStep1.Length, 6);
            Assert.AreEqual(resultStep1[0], 0.034234, 0.00001);
            Assert.AreEqual(resultStep1[1], 0.034234, 0.00001);
            Assert.AreEqual(resultStep1[2], 0.034234, 0.00001);
            Assert.AreEqual(resultStep1[3], 0.034234, 0.00001);
            Assert.AreEqual(resultStep1[4], 0.093058, 0.00001);
            Assert.AreEqual(resultStep1[5], 0.770004, 0.00001);
        }


        private Dictionary<string, double[]> compute(SDRClassifier<double, double> classifier,
                                                 int recordNum,
                                                 List<int> pattern,
                                                 double[] bucket,
                                                 double[] value)
        {

            Dictionary<string, double[]> classification = new();
            classification.Add("bucketIdx", bucket);
            classification.Add("actValue", value);
            var retVal = classifier.Compute(recordNum, pattern, classification, true, true);
            Debug.WriteLine("**************************************************************************");
            return retVal;
        }
    }

}