// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using System;
using System.ComponentModel.Composition;
using System.ComponentModel.Composition.Hosting;

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
        public BinaryClassificationContext BinaryClassification { get; }
        /// <summary>
        /// Trainers and tasks specific to multiclass classification problems.
        /// </summary>
        public MulticlassClassificationContext MulticlassClassification { get; }
        /// <summary>
        /// Trainers and tasks specific to regression problems.
        /// </summary>
        public RegressionContext Regression { get; }
        /// <summary>
        /// Trainers and tasks specific to clustering problems.
        /// </summary>
        public ClusteringContext Clustering { get; }
        /// <summary>
        /// Trainers and tasks specific to ranking problems.
        /// </summary>
        public RankingContext Ranking { get; }

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
        public DataLoadSaveOperations Data { get; }

        // REVIEW: I think it's valuable to have the simplest possible interface for logging interception here,
        // and expand if and when necessary. Exposing classes like ChannelMessage, MessageSensitivity and so on
        // looks premature at this point.
        /// <summary>
        /// The handler for the log messages.
        /// </summary>
        public Action<string> Log { get; set; }

        /// <summary>
        /// This is a MEF composition container catalog to be used for model loading.
        /// </summary>
        public CompositionContainer CompositionContainer { get; set; }

        /// <summary>
        /// Create the ML context.
        /// </summary>
        /// <param name="seed">Random seed. Set to <c>null</c> for a non-deterministic environment.</param>
        /// <param name="conc">Concurrency level. Set to 1 to run single-threaded. Set to 0 to pick automatically.</param>
        public MLContext(int? seed = null, int conc = 0)
        {
            _env = new LocalEnvironment(seed, conc, MakeCompositionContainer);
            _env.AddListener(ProcessMessage);

            BinaryClassification = new BinaryClassificationContext(_env);
            MulticlassClassification = new MulticlassClassificationContext(_env);
            Regression = new RegressionContext(_env);
            Clustering = new ClusteringContext(_env);
            Ranking = new RankingContext(_env);
            Transforms = new TransformsCatalog(_env);
            Model = new ModelOperationsCatalog(_env);
            Data = new DataLoadSaveOperations(_env);
        }

        private CompositionContainer MakeCompositionContainer()
        {
            if (CompositionContainer == null)
                return null;

            var mlContext = CompositionContainer.GetExportedValueOrDefault<MLContext>();
            if (mlContext == null)
                CompositionContainer.ComposeExportedValue<MLContext>(this);

            return CompositionContainer;
        }

        private void ProcessMessage(IMessageSource source, ChannelMessage message)
        {
            if (Log == null)
                return;

            var msg = $"[Source={source.FullName}, Kind={message.Kind}] {message.Message}";
            // Log may have been reset from another thread.
            // We don't care which logger we send the message to, just making sure we don't crash.
            Log?.Invoke(msg);
        }

        int IHostEnvironment.ConcurrencyFactor => _env.ConcurrencyFactor;
        bool IHostEnvironment.IsCancelled => _env.IsCancelled;
        ComponentCatalog IHostEnvironment.ComponentCatalog => _env.ComponentCatalog;
        string IExceptionContext.ContextDescription => _env.ContextDescription;
        IFileHandle IHostEnvironment.CreateOutputFile(string path) => _env.CreateOutputFile(path);
        IFileHandle IHostEnvironment.CreateTempFile(string suffix, string prefix) => _env.CreateTempFile(suffix, prefix);
        IFileHandle IHostEnvironment.OpenInputFile(string path) => _env.OpenInputFile(path);
        TException IExceptionContext.Process<TException>(TException ex) => _env.Process(ex);
        IHost IHostEnvironment.Register(string name, int? seed, bool? verbose, int? conc) => _env.Register(name, seed, verbose, conc);
        IChannel IChannelProvider.Start(string name) => _env.Start(name);
        IPipe<TMessage> IChannelProvider.StartPipe<TMessage>(string name) => _env.StartPipe<TMessage>(name);
        IProgressChannel IProgressChannelProvider.StartProgressChannel(string name) => _env.StartProgressChannel(name);
        CompositionContainer IHostEnvironment.GetCompositionContainer() => _env.GetCompositionContainer();
    }
}
