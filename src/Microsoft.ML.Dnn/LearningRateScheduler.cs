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
    public abstract class LearningRateScheduler
    {
        internal abstract float GetLearningRate(TrainState options);
    }

    /// <summary>
    /// This class implements linear scaling rule and LR decay.
    /// Implementation adopted from RESNET-CIFAR benchmark test in Tensorflow slim.
    /// https://github.com/tensorflow/models/blob/b974c3f95a37acedcc3c58566834c78fcae4b214/official/vision/image_classification/resnet_cifar_main.py
    /// </summary>
    public sealed class LsrDecay : LearningRateScheduler
    {
        /// <summary>
        /// Learning rate is scaled at epoch boundaries provided in LrSchedule to corresponding multiplier in the LrSchedule.
        /// Format for LrSchedule: {epoch, scaling factor}
        /// </summary>
        public readonly float[,] LrSchedule;

        /// <summary>
        /// Base Learning rate to start off with.
        /// </summary>
        public float BaseLearningRate;

        /// <summary>
        /// Linear Scale rule and LR Decay construtor assigns a default LR scheduler.
        /// </summary>
        public LsrDecay(float baseLearningRate = 0.1f)
        {
            LrSchedule = new float[,] { { 182, 0.001f }, { 136, 0.01f }, { 91, 0.1f }, { 0, 1.0f } };
            BaseLearningRate = baseLearningRate;
        }

        /// <summary>
        /// Linear Scale rule and LR Decay construtor assigns a user defined LR scheduler.
        /// </summary>
        public LsrDecay(float[,] lrschedule, float baseLearningRate = 0.1f)
        {
            LrSchedule = lrschedule;
            BaseLearningRate = baseLearningRate;
        }

        /// <summary>
        /// This function returns the corresponding scaling factor or multiplier for the given epoch from the LrSchedule.
        /// </summary>
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

        /// <summary>
        /// This function returns the Learning rate using linear scale rule and LR decay.
        /// </summary>
        internal override float GetLearningRate(TrainState trainstate)
        {
            float learningrate;
            float initialLearningRate = BaseLearningRate * trainstate.BatchSize / 128;
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
    public sealed class ExponentialLRDecay : LearningRateScheduler
    {
        /// <summary>
        /// Initial learning rate.
        /// </summary>
        public float LearningRate;

        /// <summary>
        /// The number of batches seen by the graph so far.
        /// </summary>
        public int GlobalStep;

        /// <summary>
        /// Number of decay steps
        /// </summary>
        public int DecaySteps;

        /// <summary>
        /// Learning rate decay factor.
        /// </summary>
        public float DecayRate;

        /// <summary>
        /// If Staircase is True the learning rate decays at discrete intervals and the decayed learning rate follows a staircase function.
        /// </summary>
        public bool Staircase;

        /// <summary>
        /// Number of epochs after which learning rate decays.
        /// </summary>
        public float NumEpochsPerDecay;

        /// <summary>
        /// This contructor initializes intial learning rate, number epochs per decay, decay rate and the staircase option.
        /// The defaults are taken from Tensorflow Slim.
        /// </summary>
        public ExponentialLRDecay(float learningRate = 0.01f, float numEpochsPerDecay = 2.0f, float decayRate = 0.94f, bool staircase = true)
        {
            LearningRate = learningRate;
            NumEpochsPerDecay = numEpochsPerDecay;
            DecayRate = decayRate;
            Staircase = staircase;
        }

        /// <summary>
        /// Computes exponentially decayed learning rate
        /// </summary>
        internal override float GetLearningRate(TrainState trainstate)
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
