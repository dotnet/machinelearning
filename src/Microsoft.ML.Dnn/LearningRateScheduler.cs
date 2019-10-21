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
    /// Implementation adopted from RESNET-CIFAR benchmark test in Tensorflow slim.
    /// https://github.com/tensorflow/models/blob/b974c3f95a37acedcc3c58566834c78fcae4b214/official/vision/image_classification/resnet_cifar_main.py
    /// </summary>
    public sealed class LsrDecay : ILearningRateScheduler
    {
        public readonly float[,] LrSchedule;

        public LsrDecay()
        {
            LrSchedule = new float[,] { { 182, 0.001f }, { 136, 0.01f }, { 91, 0.1f }, { 0, 1.0f } };
        }
        public LsrDecay(float[,] lrschedule)
        {
            LrSchedule = lrschedule;
        }

        private float GetLearningRateScheduleMultiplier(int epoch)
        {
            for(int i = 0; i < LrSchedule.Length; i++)
            {
                if (epoch >= LrSchedule[i,0])
                {
                    return LrSchedule[i, 1];
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

    /// <summary>
    /// This class implements Exponential Learning rate decay.
    /// Implemented from the tensorflow documentation.
    /// Source: https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/exponential_decay
    /// Default values and implementation of learning rate is from Tensorflow Slim model tests.
    /// Source : https://github.com/tensorflow/models/blob/master/research/slim/train_image_classifier.py
    /// </summary>
    public sealed class ExponentialLRDecay : ILearningRateScheduler
    {
        public float LearningRate;
        public int GlobalStep;
        public int DecaySteps;
        public float DecayRate;

        public bool Staircase { get; }

        public bool StairCase;
        public float NumEpochsPerDecay;

        public ExponentialLRDecay(float learningRate = 0.01f, float numEpochsPerDecay = 2.0f, float decayRate = 0.94f, bool staircase = true)
        {
            LearningRate = learningRate;
            NumEpochsPerDecay = numEpochsPerDecay;
            DecayRate = decayRate;
            Staircase = staircase;
        }

        float ILearningRateScheduler.GetLearningRate(TrainState trainstate)
        {
            int numSamplesPerEpoch = trainstate.BatchSize * trainstate.BatchesPerEpoch;
            DecaySteps = (int) (numSamplesPerEpoch * NumEpochsPerDecay / trainstate.BatchSize);
            GlobalStep = (trainstate.CurrentEpoch) *(trainstate.BatchesPerEpoch) + trainstate.CurrentBatchIndex;
            float decayPower = (float)GlobalStep / DecaySteps;
            decayPower = Staircase ? (float) Math.Floor(decayPower) : decayPower;
            float decayedLearningRate = LearningRate * (float) Math.Pow(DecayRate, decayPower);
            return decayedLearningRate;
        }

    }
}
