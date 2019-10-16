using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// A class that contains the current train state to use for learning rate scheduling.
    /// </summary>
    internal class TrainState
    {
        internal int CurrentBatchIndex;
        internal int CurrentEpoch;
        internal int BatchSize;
        internal int BatchesPerEpoch;
    }

    /// <summary>
    /// This interface defines a learning rate scheduler.
    /// </summary>
    public interface ILearningRateScheduler
    {
        internal float GetLearningRate(TrainState options);
    };

    /// <summary>
    /// This class implements a bare minimum learning rate scheduler class that uses a constant learning rate throughout the training.
    /// </summary>
    public sealed class BasicLR : ILearningRateScheduler
    {
        public float LearningRate;
        public BasicLR(float learningrate)
        {
            LearningRate = learningrate;
        }
        float ILearningRateScheduler.GetLearningRate(TrainState options)
        {
            return LearningRate;
        }

    }
}
