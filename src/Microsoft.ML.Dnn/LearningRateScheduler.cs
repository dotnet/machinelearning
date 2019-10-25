// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Dnn
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
    /// This abstract class defines a learning rate scheduler.
    /// </summary>
    public abstract class LearningRateScheduler
    {
        internal LearningRateScheduler()
        {
        }

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
        /// This structure represents a learning rate scheduler item type
        /// </summary>
        public readonly struct LearningRateSchedulerItem
        {

            /// <summary>
            /// Start epoch to match with the scaling factor
            /// </summary>
            public readonly int Epoch;

            /// <summary>
            /// Scaling factor or multiplier that changes the learning rate for Linear scale rule
            /// </summary>
            public readonly float ScalingFactor;

            public LearningRateSchedulerItem(int epoch, float scalingfactor) : this()
            {
                Epoch = epoch;
                ScalingFactor = scalingfactor;
            }
        }

        /// <summary>
        /// Learning rate is scaled at epoch boundaries provided in LrSchedule to corresponding multiplier in the LrSchedule.
        /// Format for LrSchedule: {start epoch, scaling factor}, ordered with largest start epoch first
        /// </summary>
        private readonly IReadOnlyList<LearningRateSchedulerItem> _lrSchedule;

        /// <summary>
        /// Base Learning rate to start off with.
        /// </summary>
        public readonly float BaseLearningRate;
        private IReadOnlyList<LearningRateSchedulerItem> GetDefaultLearningDecayItems()
        {
            List<LearningRateSchedulerItem> lrs = new List<LearningRateSchedulerItem>();
            int[] epochs = { 182, 136, 91, 0 };
            float[] scalingFactor = { 0.0001f, 0.01f, 0.1f, 1.0f };
            for(int i = 0; i < 4; i++)
            {
                LearningRateSchedulerItem item = new LearningRateSchedulerItem(epochs[i], scalingFactor[i]);
                lrs.Add(item);
            }
            return lrs.AsReadOnly();
        }

        /// <summary>
        /// Linear Scale rule and LR Decay construtor assigns a default LR scheduler.
        /// </summary>
        public LsrDecay(float baseLearningRate = 0.1f)
        {
            _lrSchedule = GetDefaultLearningDecayItems();
            BaseLearningRate = baseLearningRate;
        }

        /// <summary>
        /// Linear Scale rule and LR Decay construtor assigns a user defined LR scheduler.
        /// </summary>
        public LsrDecay(IReadOnlyList<LearningRateSchedulerItem> lrschedule, float baseLearningRate = 0.1f)
        {
            _lrSchedule = lrschedule;
            BaseLearningRate = baseLearningRate;
        }

        /// <summary>
        /// This function returns the corresponding scaling factor or multiplier for the given epoch from the LrSchedule.
        /// </summary>
        private float GetLearningRateScheduleMultiplier(int epoch)
        {
            for(int i = 0; i < _lrSchedule.Count; i++)
            {
                if (epoch >= _lrSchedule[i].Epoch)
                {
                    return _lrSchedule[i].ScalingFactor;
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
