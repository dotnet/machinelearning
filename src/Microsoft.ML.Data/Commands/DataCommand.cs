// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Runtime.Command;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// This holds useful base classes for commands that ingest a primary dataset and deal with associated model files.
    /// </summary>
    public static class DataCommand
    {
        public abstract class ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "The data loader", ShortName = "loader", SortOrder = 1, NullName = "<Auto>")]
            public SubComponent<IDataLoader, SignatureDataLoader> Loader;

            [Argument(ArgumentType.AtMostOnce, IsInputFileName = true, HelpText = "The data file", ShortName = "data", SortOrder = 0)]
            public string DataFile;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Model file to save", ShortName = "out")]
            public string OutputModelFile;

            [Argument(ArgumentType.AtMostOnce, IsInputFileName = true, HelpText = "Model file to load", ShortName = "in", SortOrder = 90)]
            public string InputModelFile;

            [Argument(ArgumentType.Multiple, HelpText = "Load transforms from model file?", ShortName = "loadTrans", SortOrder = 91)]
            public bool? LoadTransforms;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Random seed", ShortName = "seed", SortOrder = 101)]
            public int? RandomSeed;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Verbose?", ShortName = "v", Hide = true)]
            public bool? Verbose;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The web server to publish the RESTful API", Hide = true)]
            public ServerChannel.IServerFactory Server;

            // This is actually an advisory value. The implementations themselves are responsible for
            // determining what they consider appropriate, and the actual heuristics is a bit more
            // complex than just this.
            [Argument(ArgumentType.LastOccurenceWins,
                HelpText = "Desired degree of parallelism in the data pipeline", ShortName = "n")]
            public int? Parallel;

            [Argument(ArgumentType.Multiple, HelpText = "Transform", ShortName = "xf")]
            public KeyValuePair<string, SubComponent<IDataTransform, SignatureDataTransform>>[] Transform;
        }

        public abstract class ImplBase<TArgs> : ICommand
            where TArgs : ArgumentsBase
        {
            protected readonly IHost Host;
            protected readonly TArgs Args;
            private readonly ServerChannel.IServerFactory _serverFactory;

            protected ServerChannel.IServer InitServer(IChannel ch)
            {
                Host.CheckValue(ch, nameof(ch));
                ch.Check(Host != null, nameof(InitServer) + " called prematurely");
                return _serverFactory?.CreateComponent(Host, ch);
            }

            /// <summary>
            /// The degree of concurrency is passed in the conc parameter. If it is null, the value
            /// of args.parralel is used. If that is null, zero is used (which means "automatic").
            /// </summary>
            protected ImplBase(IHostEnvironment env, TArgs args, string name, int? conc = null)
            {
                Contracts.CheckValue(env, nameof(env));

                // Note that env may be null here, which is OK since the CheckXxx methods are extension
                // methods designed to allow null.
                env.CheckValue(args, nameof(args));
                env.CheckParam(conc == null || conc >= 0, nameof(conc), "Degree of concurrency must be non-negative (or null)");

                conc = conc ?? args.Parallel;
                env.CheckUserArg(!(conc < 0), nameof(args.Parallel), "Degree of parallelism must be non-negative (or null)");

                // Capture the environment options from args.
                env = env.Register(name, args.RandomSeed, args.Verbose, conc);

                env.CheckNonWhiteSpace(name, nameof(name));
                Host = env.Register(name);
                Args = args;
                _serverFactory = args.Server;
                Utils.CheckOptionalUserDirectory(args.OutputModelFile, nameof(args.OutputModelFile));
            }

            protected ImplBase(ImplBase<TArgs> impl, string name)
            {
                Contracts.CheckValue(impl, nameof(impl));
                Contracts.AssertValue(impl.Host);
                impl.Host.AssertValue(impl.Args);
                impl.Host.AssertValue(name);
                Args = impl.Args;
                Host = impl.Host.Register(name);
            }

            public abstract void Run();

            protected virtual void SendTelemetry(IChannelProvider prov)
            {
                using (var pipe = prov.StartPipe<TelemetryMessage>("TelemetryPipe"))
                {
                    SendTelemetryCore(pipe);
                    pipe.Done();
                }
            }

            protected void SendTelemetryComponent(IPipe<TelemetryMessage> pipe, SubComponent sub)
            {
                Host.AssertValue(pipe);
                Host.AssertValueOrNull(sub);

                if (sub.IsGood())
                    pipe.Send(TelemetryMessage.CreateTrainer(sub.Kind, sub.SubComponentSettings));
            }

            protected virtual void SendTelemetryCore(IPipe<TelemetryMessage> pipe)
            {
                Contracts.AssertValue(pipe);

                if (Args.Transform != null)
                {
                    foreach (var transform in Args.Transform)
                        SendTelemetryComponent(pipe, transform.Value);
                }
            }

            protected void SendTelemetryMetricCore(IChannelProvider prov, Dictionary<string, double> averageMetric)
            {
                using (var pipe = prov.StartPipe<TelemetryMessage>("TelemetryPipe"))
                {
                    Contracts.AssertValue(pipe);

                    if (averageMetric != null)
                    {
                        foreach (var pair in averageMetric)
                            pipe.Send(TelemetryMessage.CreateMetric(pair.Key, pair.Value, null));
                    }
                    pipe.Done();
                }
            }

            protected void SendTelemetryMetric(Dictionary<string, IDataView>[] metricValues)
            {
                Dictionary<string, double> averageMetric = new Dictionary<string, double>();
                foreach (Dictionary<string, IDataView> mValue in metricValues)
                {
                    using (var cursor = mValue.First().Value.GetRowCursor(col => true))
                    {
                        while (cursor.MoveNext())
                        {
                            for (int currentIndex = 0; currentIndex < cursor.Schema.ColumnCount; currentIndex++)
                            {
                                var nameOfMetric = "TLC_" + cursor.Schema.GetColumnName(currentIndex);
                                var type = cursor.Schema.GetColumnType(currentIndex);
                                if (type.IsNumber)
                                {
                                    var getter = RowCursorUtils.GetGetterAs<double>(NumberType.R8, cursor, currentIndex);
                                    double metricValue = 0;
                                    getter(ref metricValue);
                                    if (averageMetric.ContainsKey(nameOfMetric))
                                    {
                                        averageMetric[nameOfMetric] += metricValue;
                                    }
                                    else
                                    {
                                        averageMetric.Add(nameOfMetric, metricValue);
                                    }
                                }
                                else
                                {
                                    if (averageMetric.ContainsKey(nameOfMetric))
                                    {
                                        averageMetric[nameOfMetric] = Double.NaN;
                                    }
                                    else
                                    {
                                        averageMetric.Add(nameOfMetric, Double.NaN);
                                    }
                                }
                            }
                        }
                    }
                }
                Dictionary<string, double> newAverageMetric = new Dictionary<string, double>();
                foreach (var pair in averageMetric)
                {
                    newAverageMetric.Add(pair.Key, pair.Value / metricValues.Length);
                }
                SendTelemetryMetricCore(Host, newAverageMetric);
            }

            protected IDataLoader LoadLoader(RepositoryReader rep, string path, bool loadTransforms)
            {
                return ModelFileUtils.LoadLoader(Host, rep, new MultiFileSource(path), loadTransforms);
            }

            protected void SaveLoader(IDataLoader loader, string path)
            {
                using (var file = Host.CreateOutputFile(path))
                    LoaderUtils.SaveLoader(loader, file);
            }

            protected IDataLoader CreateAndSaveLoader(string defaultLoader = "TextLoader")
            {
                var loader = CreateLoader(defaultLoader);
                if (!string.IsNullOrWhiteSpace(Args.OutputModelFile))
                {
                    using (var file = Host.CreateOutputFile(Args.OutputModelFile))
                        LoaderUtils.SaveLoader(loader, file);
                }
                return loader;
            }

            /// <summary>
            /// Loads multiple artifacts of interest from the input model file, given the context
            /// established by the command line arguments.
            /// </summary>
            /// <param name="ch">The channel to which to provide output.</param>
            /// <param name="wantPredictor">Whether we want a predictor from the model file. If
            /// <c>false</c> we will not even attempt to load a predictor. If <c>null</c> we will
            /// load the predictor, if present. If <c>true</c> we will load the predictor, or fail
            /// noisily if we cannot.</param>
            /// <param name="predictor">The predictor in the model, or <c>null</c> if
            /// <paramref name="wantPredictor"/> was false, or <paramref name="wantPredictor"/> was
            /// <c>null</c> and no predictor was present.</param>
            /// <param name="wantTrainSchema">Whether we want the training schema. Unlike
            /// <paramref name="wantPredictor"/>, this has no "hard fail if not present" option. If
            /// this is <c>true</c>, it is still possible for <paramref name="trainSchema"/> to remain
            /// <c>null</c> if there were no role mappings, or pipeline.</param>
            /// <param name="trainSchema">The training schema if <paramref name="wantTrainSchema"/>
            /// is true, and there were role mappings stored in the model.</param>
            /// <param name="pipe">The data pipe constructed from the combination of the
            /// model and command line arguments.</param>
            protected void LoadModelObjects(
                IChannel ch,
                bool? wantPredictor, out IPredictor predictor,
                bool wantTrainSchema, out RoleMappedSchema trainSchema,
                out IDataLoader pipe)
            {
                // First handle the case where there is no input model file.
                // Everything must come from the command line.

                using (var file = Host.OpenInputFile(Args.InputModelFile))
                using (var strm = file.OpenReadStream())
                using (var rep = RepositoryReader.Open(strm, Host))
                {
                    // First consider loading the predictor.
                    if (wantPredictor == false)
                        predictor = null;
                    else
                    {
                        ch.Trace("Loading predictor");
                        predictor = ModelFileUtils.LoadPredictorOrNull(Host, rep);
                        if (wantPredictor == true)
                            Host.Check(predictor != null, "Could not load predictor from model file");
                    }

                    // Next create the loader.
                    var sub = Args.Loader;
                    IDataLoader trainPipe = null;
                    if (sub.IsGood())
                    {
                        // The loader is overridden from the command line.
                        pipe = sub.CreateInstance(Host, new MultiFileSource(Args.DataFile));
                        if (Args.LoadTransforms == true)
                        {
                            Host.CheckUserArg(!string.IsNullOrWhiteSpace(Args.InputModelFile), nameof(Args.InputModelFile));
                            pipe = LoadTransformChain(pipe);
                        }
                    }
                    else
                    {
                        var loadTrans = Args.LoadTransforms ?? true;
                        pipe = LoadLoader(rep, Args.DataFile, loadTrans);
                        if (loadTrans)
                            trainPipe = pipe;
                    }

                    if (Utils.Size(Args.Transform) > 0)
                        pipe = CompositeDataLoader.Create(Host, pipe, Args.Transform);

                    // Next consider loading the training data's role mapped schema.
                    trainSchema = null;
                    if (wantTrainSchema)
                    {
                        // First try to get the role mappings.
                        var trainRoleMappings = ModelFileUtils.LoadRoleMappingsOrNull(Host, rep);
                        if (trainRoleMappings != null)
                        {
                            // Next create the training schema. In the event that the loaded pipeline happens
                            // to be the training pipe, we can just use that. If it differs, then we need to
                            // load the full pipeline from the model, relying upon the fact that all loaders
                            // can be loaded with no data at all, to get their schemas.
                            if (trainPipe == null)
                                trainPipe = ModelFileUtils.LoadLoader(Host, rep, new MultiFileSource(null), loadTransforms: true);
                            trainSchema = RoleMappedSchema.Create(trainPipe.Schema, trainRoleMappings);
                        }
                        // If the role mappings are null, an alternative would be to fail. However the idea
                        // is that the scorer should always still succeed, although perhaps with reduced
                        // functionality, even when the training schema is null, since not all versions of
                        // TLC models will have the role mappings preserved, I believe. And, we do want to
                        // maintain backwards compatibility.
                    }
                }
            }

            protected IDataLoader CreateLoader(string defaultLoader = "TextLoader")
            {
                var loader = CreateRawLoader(defaultLoader);
                loader = CreateTransformChain(loader);
                return loader;
            }

            private IDataLoader CreateTransformChain(IDataLoader loader)
            {
                return CompositeDataLoader.Create(Host, loader, Args.Transform);
            }

            protected IDataLoader CreateRawLoader(string defaultLoader = "TextLoader", string dataFile = null)
            {
                if (string.IsNullOrWhiteSpace(dataFile))
                    dataFile = Args.DataFile;

                IDataLoader loader;
                if (!string.IsNullOrWhiteSpace(Args.InputModelFile) && !Args.Loader.IsGood())
                {
                    // Load the loader from the data model.
                    using (var file = Host.OpenInputFile(Args.InputModelFile))
                    using (var strm = file.OpenReadStream())
                    using (var rep = RepositoryReader.Open(strm, Host))
                        loader = LoadLoader(rep, dataFile, Args.LoadTransforms ?? true);
                }
                else
                {
                    // Either there is no input model file, or there is, but the loader is overridden.
                    var sub = Args.Loader;
                    if (!sub.IsGood())
                    {
                        var ext = Path.GetExtension(dataFile);
                        var isText =
                            string.Equals(ext, ".txt", StringComparison.OrdinalIgnoreCase) ||
                            string.Equals(ext, ".tlc", StringComparison.OrdinalIgnoreCase);
                        var isBinary = string.Equals(ext, ".idv", StringComparison.OrdinalIgnoreCase);
                        var isTranspose = string.Equals(ext, ".tdv", StringComparison.OrdinalIgnoreCase);
                        sub =
                            new SubComponent<IDataLoader, SignatureDataLoader>(
                                isText ? "TextLoader" : isBinary ? "BinaryLoader" : isTranspose ? "TransposeLoader" : defaultLoader);
                    }

                    loader = sub.CreateInstance(Host, new MultiFileSource(dataFile));

                    if (Args.LoadTransforms == true)
                    {
                        Host.CheckUserArg(!string.IsNullOrWhiteSpace(Args.InputModelFile), nameof(Args.InputModelFile));
                        loader = LoadTransformChain(loader);
                    }
                }
                return loader;
            }

            private IDataLoader LoadTransformChain(IDataLoader srcData)
            {
                Host.Assert(!string.IsNullOrWhiteSpace(Args.InputModelFile));

                using (var file = Host.OpenInputFile(Args.InputModelFile))
                using (var strm = file.OpenReadStream())
                using (var rep = RepositoryReader.Open(strm, Host))
                using (var pipeLoaderEntry = rep.OpenEntry(ModelFileUtils.DirDataLoaderModel, ModelLoadContext.ModelStreamName))
                using (var ctx = new ModelLoadContext(rep, pipeLoaderEntry, ModelFileUtils.DirDataLoaderModel))
                    return CompositeDataLoader.Create(Host, ctx, srcData, x => true);
            }
        }
    }

    public static class LoaderUtils
    {
        /// <summary>
        /// Saves <paramref name="loader"/> to the specified <paramref name="file"/>.
        /// </summary>
        public static void SaveLoader(IDataLoader loader, IFileHandle file)
        {
            Contracts.CheckValue(loader, nameof(loader));
            Contracts.CheckValue(file, nameof(file));
            Contracts.CheckParam(file.CanWrite, nameof(file), "Must be writable");

            using (var stream = file.CreateWriteStream())
            {
                SaveLoader(loader, stream);
            }
        }

        /// <summary>
        /// Saves <paramref name="loader"/> to the specified <paramref name="stream"/>.
        /// </summary>
        public static void SaveLoader(IDataLoader loader, Stream stream)
        {
            Contracts.CheckValue(loader, nameof(loader));
            Contracts.CheckValue(stream, nameof(stream));
            Contracts.CheckParam(stream.CanWrite, nameof(stream), "Must be writable");

            using (var rep = RepositoryWriter.CreateNew(stream))
            {
                ModelSaveContext.SaveModel(rep, loader, ModelFileUtils.DirDataLoaderModel);
                rep.Commit();
            }
        }
    }
}
