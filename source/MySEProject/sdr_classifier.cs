// SDR Classifier python code from NUPIC converted to C#
//https://github.com/numenta/nupic/blob/master/src/nupic/algorithms/sdr_classifier.py

using System;
using System.Collections.Generic;
using Nupic.Serializable;

namespace Nupic
{
    public class SDRClassifier : Serializable
    {
        private Queue<int> _history = new Queue<int>();
        private int _numBuckets = 0;
        private int _bucketIndex = 0;
        private int _actValue = 0;
        private int _learningEnabled = 0;
        private int _inferenceEnabled = 0;
        private int _learnIterations = 0;
        private int _learn = 0;
        private int _infer = 0;

        public SDRClassifier()
        {
        }
    }


class Model
{
    public const int VERSION = 1;
    private int[] steps;
    private double alpha;
    private double actValueAlpha;
    private int verbosity;
    private int _maxSteps;
    private Queue<int> _patternNZHistory;

    public Model(int[] steps = null, double alpha = 0.001, double actValueAlpha = 0.3, int verbosity = 0)
    {
        steps = steps ?? new[] { 1 };

        if (steps.Length == 0)
        {
            throw new TypeError("steps cannot be empty");
        }
        if (Array.Exists(steps, item => item < 0))
        {
            throw new ValueError("steps must be a list of non-negative ints");
        }
        if (alpha < 0)
        {
            throw new ValueError("alpha (learning rate) must be a positive number");
        }
        if (actValueAlpha < 0 || actValueAlpha >= 1)
        {
            throw new ValueError("actValueAlpha be a number between 0 and 1");
        }

        this.steps = steps;
        this.alpha = alpha;
        this.actValueAlpha = actValueAlpha;
        this.verbosity = verbosity;

        this._maxSteps = steps.Max() + 1;
        this._patternNZHistory = new Queue<int>(this._maxSteps);
    }
}
}



