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
    /// The <see cref="MLContext"/> is a starting point for all ML.NET operations. It is instantiated by user,
    /// provides mechanisms for logging and entry points for training, prediction, model operations etc.
    /// </summary>
    public sealed class MLContext : IHostEnvironment
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
        /// Create the ML context.
        /// </summary>
        /// <param name="seed">Random seed. Set to <c>null</c> for a non-deterministic environment.</param>
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
            Transforms = new TransformsCatalog(_env);
            Model = new ModelOperationsCatalog(_env);
            Data = new DataOperationsCatalog(_env);
        }

        private void ProcessMessage(IMessageSource source, ChannelMessage message)
        {
            var log = Log;

            if (log == null)
                return;

            var msg = $"[Source={source.FullName}, Kind={message.Kind}] {message.Message}";

            log(this, new LoggingEventArgs(msg));
        }

        string IExceptionContext.ContextDescription => _env.ContextDescription;
        TException IExceptionContext.Process<TException>(TException ex) => _env.Process(ex);
        IHost IHostEnvironment.Register(string name, int? seed, bool? verbose) => _env.Register(name, seed, verbose);
        IChannel IChannelProvider.Start(string name) => _env.Start(name);
        IPipe<TMessage> IChannelProvider.StartPipe<TMessage>(string name) => _env.StartPipe<TMessage>(name);
        IProgressChannel IProgressChannelProvider.StartProgressChannel(string name) => _env.StartProgressChannel(name);

        [BestFriend]
        internal void CancelExecution() => ((ICancelable)_env).CancelExecution();
    }
}
