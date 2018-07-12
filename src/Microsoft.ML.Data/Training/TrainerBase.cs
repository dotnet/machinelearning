// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Runtime.Training
{
    public abstract class TrainerBase<TPredictor> : ITrainer<TPredictor>, ITrainerEx
        where TPredictor : IPredictor
    {
        /// <summary>
        /// A standard string to use in errors or warnings by subclasses, to communicate the idea that no valid
        /// instances were able to be found.
        /// </summary>
        protected const string NoTrainingInstancesMessage = "No valid training instances found, all instances have missing features.";

        protected IHost Host { get; }

        public string Name { get; }
        public abstract PredictionKind PredictionKind { get; }
        public abstract bool NeedNormalization { get; }
        public abstract bool NeedCalibration { get; }
        public abstract bool WantCaching { get; }

        public virtual bool SupportsValidation => false;
        public virtual bool SupportsIncrementalTraining => false;

        protected TrainerBase(IHostEnvironment env, string name)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckNonEmpty(name, nameof(name));

            Name = name;
            Host = env.Register(name);
        }

        IPredictor ITrainer.Train(TrainContext context) => Train(context);

        public abstract TPredictor Train(TrainContext context);
    }
}
