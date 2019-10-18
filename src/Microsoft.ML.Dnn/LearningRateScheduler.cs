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
    /// This class implements linear scaling rule and LR decay.
    /// </summary>
    public sealed class LsrDecay : ILearningRateScheduler
    {
        private static readonly float[,] _lrSchedule = new float[,] { { 182, 0.001f }, { 136, 0.01f }, { 91, 0.1f }, { 0, 1.0f } };
        private float GetLearningRateScheduleMultiplier(int epoch)
        {
            for(int i = 0; i < _lrSchedule.Length; i++)
            {
                if (epoch >= _lrSchedule[i,0])
                {
                    return _lrSchedule[i, 1];
                }
            }
            return 1.0f;
        }
        float ILearningRateScheduler.GetLearningRate(TrainState trainstate)
        {
            float learningrate;
            float baseLearningRate = 0.1f;
            float initialLearningRate = baseLearningRate * trainstate.BatchSize / 128;
            learningrate = initialLearningRate * GetLearningRateScheduleMultiplier(trainstate.CurrentEpoch);
            return learningrate;
        }

    }
}
