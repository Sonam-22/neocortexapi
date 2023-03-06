namespace SdRClassifierTest
{
    using SDRClassifier;
    using NumSharp;

    [TestClass]
    public class SDRClassifierTest
    {
        [TestMethod]
        public void TestInferSingleStep()
        {
            var classiffier = new SDRClassifier(new List<int>() { 1 }, 0.001, 0.3, 0.0, 1);
            var distributionList = classiffier.InferSingleStep(new List<int>() { 0, 1, 2 }, np.arange(16).reshape(4,4));
            var highiestVoted = distributionList.Max();

            Assert.AreEqual(highiestVoted, distributionList[3]);
        }

        [TestMethod]
        public void TestInfer()
        {
            var classiffier = new SDRClassifier(new List<int>() { 1 }, 0.001, 0.3, 0.0, 1);
            //classiffier.Infer(new List<int>() { 0, 1, 2 });
        }
    }

}