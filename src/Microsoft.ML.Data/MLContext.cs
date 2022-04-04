// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML
{
    /// <summary>
    /// The common context for all ML.NET operations. Once instantiated by the user, it provides a way to
    /// create components for data preparation, feature engineering, training, prediction, and model evaluation.
    /// It also allows logging, execution control, and the ability to set repeatable random numbers.
    /// </summary>
    public sealed class MLContext : IHostEnvironmentInternal
    {
        // REVIEW: consider making LocalEnvironment and MLContext the same class instead of encapsulation.
        private readonly LocalEnvironment _env;

        /// <summary>
        /// Trainers and tasks specific to binary classification problems.
        /// </summary>
        public BinaryClassificationCatalog BinaryClassification { get; }
        /// <summary>
        /// Trainers and tasks specific to multiclass classification problems.
        /// </summary>
        public MulticlassClassificationCatalog MulticlassClassification { get; }
        /// <summary>
        /// Trainers and tasks specific to regression problems.
        /// </summary>
        public RegressionCatalog Regression { get; }
        /// <summary>
        /// Trainers and tasks specific to clustering problems.
        /// </summary>
        public ClusteringCatalog Clustering { get; }

        /// <summary>
        /// Trainers and tasks specific to ranking problems.
        /// </summary>
        public RankingCatalog Ranking { get; }

        /// <summary>
        /// Trainers and tasks specific to anomaly detection problems.
        /// </summary>
        public AnomalyDetectionCatalog AnomalyDetection { get; }

        /// <summary>
        /// Trainers and tasks specific to forecasting problems.
        /// </summary>
        public ForecastingCatalog Forecasting { get; }

        /// <summary>
        /// Data processing operations.
        /// </summary>
        public TransformsCatalog Transforms { get; }

        /// <summary>
        /// Operations with trained models.
        /// </summary>
        public ModelOperationsCatalog Model { get; }

        /// <summary>
        /// Data loading and saving.
        /// </summary>
        public DataOperationsCatalog Data { get; }

        // REVIEW: I think it's valuable to have the simplest possible interface for logging interception here,
        // and expand if and when necessary. Exposing classes like ChannelMessage, MessageSensitivity and so on
        // looks premature at this point.
        /// <summary>
        /// The handler for the log messages.
        /// </summary>
        public event EventHandler<LoggingEventArgs> Log;

        /// <summary>
        /// This is a catalog of components that will be used for model loading.
        /// </summary>
        public ComponentCatalog ComponentCatalog => _env.ComponentCatalog;

        /// <summary>
        /// Gets or sets the location for the temp files created by ML.NET.
        /// </summary>
        public string TempFilePath
        {
            get { return _env.TempFilePath; }
            set { _env.TempFilePath = value; }
        }

        /// <summary>
        /// Allow falling back to run on CPU if couldn't run on GPU.
        /// </summary>
        public bool FallbackToCpu
        {
            get => _env.FallbackToCpu;
            set { _env.FallbackToCpu = value; }
        }

        /// <summary>
        /// GPU device ID to run execution on, <see langword="null" /> to run on CPU.
        /// </summary>
        public int? GpuDeviceId
        {
            get => _env.GpuDeviceId;
            set { _env.GpuDeviceId = value; }
        }

        /// <summary>
        /// Create the ML context.
        /// </summary>
        /// <param name="seed">Seed for MLContext's random number generator. See the remarks for more details.</param>
        /// <remarks>
        /// Many operations in ML.NET require randomness, such as
        /// random data shuffling, random sampling, random parameter initialization,
        /// random permutation, random feature selection, and many more.
        /// MLContext's random number generator is the global source of randomness for
        /// all of such random operations.
        ///
        /// If a fixed seed is provided by <paramref name="seed"/>, MLContext environment becomes
        /// deterministic, meaning that the results are repeatable and will remain the same across multiple runs.
        /// For instance in many of ML.NET's API reference example code snippets, a seed is provided.
        /// That's because we want the users to get the same output as what's included in example comments,
        /// when they run the example on their own machine.
        ///
        /// Generally though, repeatability is not a requirement and that's the default behavior.
        /// If a seed is not provided by <paramref name="seed"/>, i.e. it's set to <see langword="null"/>,
        /// MLContext environment becomes non-deterministic and outputs change across multiple runs.
        ///
        /// There are many operations in ML.NET that don't use any randomness, such as
        /// min-max normalization, concatenating columns, missing value indication, etc.
        /// The behavior of those operations are deterministic regardless of the seed value.
        ///
        /// Also ML.NET trainers don't use randomness *after* the training is finished.
        /// So, the predictions from a loaded model don't depend on the seed value.
        /// </remarks>
        public MLContext(int? seed = null)
        {
            _env = new LocalEnvironment(seed);
            _env.AddListener(ProcessMessage);

            BinaryClassification = new BinaryClassificationCatalog(_env);
            MulticlassClassification = new MulticlassClassificationCatalog(_env);
            Regression = new RegressionCatalog(_env);
            Clustering = new ClusteringCatalog(_env);
            Ranking = new RankingCatalog(_env);
            AnomalyDetection = new AnomalyDetectionCatalog(_env);
            Forecasting = new ForecastingCatalog(_env);
            Transforms = new TransformsCatalog(_env);
            Model = new ModelOperationsCatalog(_env);
            Data = new DataOperationsCatalog(_env);
        }

        private void ProcessMessage(IMessageSource source, ChannelMessage message)
        {
            var log = Log;

            if (log == null)
                return;

            log(this, new LoggingEventArgs(message.Message, message.Kind, source.FullName));
        }

        string IExceptionContext.ContextDescription => _env.ContextDescription;
        TException IExceptionContext.Process<TException>(TException ex) => _env.Process(ex);
        IHost IHostEnvironment.Register(string name, int? seed, bool? verbose) => _env.Register(name, seed, verbose);
        IChannel IChannelProvider.Start(string name) => _env.Start(name);
        IPipe<TMessage> IChannelProvider.StartPipe<TMessage>(string name) => _env.StartPipe<TMessage>(name);
        IProgressChannel IProgressChannelProvider.StartProgressChannel(string name) => _env.StartProgressChannel(name);
        int? IHostEnvironmentInternal.Seed => _env.Seed;

        [BestFriend]
        internal void CancelExecution() => ((ICancelable)_env).CancelExecution();
    }
}
