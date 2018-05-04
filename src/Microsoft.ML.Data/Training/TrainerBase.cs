// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Runtime.Training
{
    public abstract class TrainerBase : ITrainer, ITrainerEx
    {
        public const string NoTrainingInstancesMessage = "No valid training instances found, all instances have missing features.";

        protected readonly IHost Host;

        public string Name { get; }
        public abstract PredictionKind PredictionKind { get; }
        public abstract bool NeedNormalization { get; }
        public abstract bool NeedCalibration { get; }
        public abstract bool WantCaching { get; }

        protected TrainerBase(IHostEnvironment env, string name)
        {
            Contracts.CheckValue(env, nameof(env));
            Contracts.CheckNonEmpty(name, nameof(name));

            Name = name;
            Host = env.Register(name);
        }

        IPredictor ITrainer.CreatePredictor()
        {
            return CreatePredictorCore();
        }

        protected abstract IPredictor CreatePredictorCore();
    }

    public abstract class TrainerBase<TPredictor> : TrainerBase
        where TPredictor : IPredictor
    {
        protected TrainerBase(IHostEnvironment env, string name)
            : base(env, name)
        {
        }

        public abstract TPredictor CreatePredictor();

        protected sealed override IPredictor CreatePredictorCore()
        {
            return CreatePredictor();
        }
    }

    public abstract class TrainerBase<TDataSet, TPredictor> : TrainerBase<TPredictor>, ITrainer<TDataSet, TPredictor>
        where TPredictor : IPredictor
    {
        protected TrainerBase(IHostEnvironment env, string name)
            : base(env, name)
        {
        }

        public abstract void Train(TDataSet data);
    }
}
