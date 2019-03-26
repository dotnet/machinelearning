// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Command;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// This holds useful base classes for commands that ingest a primary dataset and deal with associated model files.
    /// </summary>
    [BestFriend]
    internal static class DataCommand
    {
        public abstract class ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly, HelpText = "The data loader", ShortName = "loader", SortOrder = 1, NullName = "<Auto>", SignatureType = typeof(SignatureDataLoader))]
            public IComponentFactory<IMultiStreamSource, ILegacyDataLoader> Loader;

            [Argument(ArgumentType.AtMostOnce, IsInputFileName = true, HelpText = "The data file", ShortName = "data", SortOrder = 0)]
            public string DataFile;

            [Argument(ArgumentType.AtMostOnce, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly, HelpText = "Model file to save", ShortName = "out")]
            public string OutputModelFile;

            [Argument(ArgumentType.AtMostOnce, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly, IsInputFileName = true, HelpText = "Model file to load", ShortName = "in", SortOrder = 90)]
            public string InputModelFile;

            [Argument(ArgumentType.Multiple, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly, HelpText = "Load transforms from model file?", ShortName = "loadTrans", SortOrder = 91)]
            public bool? LoadTransforms;

            [Argument(ArgumentType.AtMostOnce, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly, HelpText = "Random seed", ShortName = "seed", SortOrder = 101)]
            public int? RandomSeed;

            [Argument(ArgumentType.AtMostOnce, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly, HelpText = "Verbose?", ShortName = "v", Hide = true)]
            public bool? Verbose;

            [Argument(ArgumentType.AtMostOnce, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly, HelpText = "The web server to publish the RESTful API", Hide = true)]
            public ServerChannel.IServerFactory Server;

            // This is actually an advisory value. The implementations themselves are responsible for
            // determining what they consider appropriate, and the actual heuristics is a bit more
            // complex than just this.
            [Argument(ArgumentType.LastOccurenceWins, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly,
                HelpText = "Desired degree of parallelism in the data pipeline", ShortName = "n")]
            public int? Parallel;

            [Argument(ArgumentType.Multiple, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly,
                HelpText = "Transform", Name ="Transform", ShortName = "xf", SignatureType = typeof(SignatureDataTransform))]
            public KeyValuePair<string, IComponentFactory<IDataView, IDataTransform>>[] Transforms;
        }

        [BestFriend]
        internal abstract class ImplBase<TOptions> : ICommand
            where TOptions : ArgumentsBase
        {
            protected readonly IHost Host;
            protected readonly TOptions ImplOptions;
            private readonly ServerChannel.IServerFactory _serverFactory;

            protected ServerChannel.IServer InitServer(IChannel ch)
            {
                Host.CheckValue(ch, nameof(ch));
                ch.Check(Host != null, nameof(InitServer) + " called prematurely");
                return _serverFactory?.CreateComponent(Host, ch);
            }

            protected ImplBase(IHostEnvironment env, TOptions options, string name)
            {
                Contracts.CheckValue(env, nameof(env));

                // Note that env may be null here, which is OK since the CheckXxx methods are extension
                // methods designed to allow null.
                env.CheckValue(options, nameof(options));

                env.CheckUserArg(!(options.Parallel < 0), nameof(options.Parallel), "Degree of parallelism must be non-negative (or null)");

                // Capture the environment options from args.
                env = env.Register(name, options.RandomSeed, options.Verbose);

                env.CheckNonWhiteSpace(name, nameof(name));
                Host = env.Register(name);
                ImplOptions = options;
                _serverFactory = options.Server;
                Utils.CheckOptionalUserDirectory(options.OutputModelFile, nameof(options.OutputModelFile));
            }

            protected ImplBase(ImplBase<TOptions> impl, string name)
            {
                Contracts.CheckValue(impl, nameof(impl));
                Contracts.AssertValue(impl.Host);
                impl.Host.AssertValue(impl.ImplOptions);
                impl.Host.AssertValue(name);
                ImplOptions = impl.ImplOptions;
                Host = impl.Host.Register(name);
            }

            public abstract void Run();

            protected virtual void SendTelemetry(IChannelProvider prov)
            {
                using (var pipe = prov.StartPipe<TelemetryMessage>("TelemetryPipe"))
                {
                    SendTelemetryCore(pipe);
                }
            }

            protected void SendTelemetryComponent(IPipe<TelemetryMessage> pipe, IComponentFactory factory)
            {
                Host.AssertValue(pipe);
                Host.AssertValueOrNull(factory);

                if (factory is ICommandLineComponentFactory commandLineFactory)
                    pipe.Send(TelemetryMessage.CreateTrainer(commandLineFactory.Name, commandLineFactory.GetSettingsString()));
                else
                    pipe.Send(TelemetryMessage.CreateTrainer("Unknown", "Non-ICommandLineComponentFactory object"));
            }

            protected virtual void SendTelemetryCore(IPipe<TelemetryMessage> pipe)
            {
                Contracts.AssertValue(pipe);

                if (ImplOptions.Transforms != null)
                {
                    foreach (var transform in ImplOptions.Transforms)
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
                }
            }

            protected void SendTelemetryMetric(Dictionary<string, IDataView>[] metricValues)
            {
                Dictionary<string, double> averageMetric = new Dictionary<string, double>();
                foreach (Dictionary<string, IDataView> mValue in metricValues)
                {
                    var data = mValue.First().Value;
                    using (var cursor = data.GetRowCursorForAllColumns())
                    {
                        while (cursor.MoveNext())
                        {
                            for (int currentIndex = 0; currentIndex < cursor.Schema.Count; currentIndex++)
                            {
                                var nameOfMetric = "TLC_" + cursor.Schema[currentIndex].Name;
                                var type = cursor.Schema[currentIndex].Type;
                                if (type is NumberDataViewType)
                                {
                                    var getter = RowCursorUtils.GetGetterAs<double>(NumberDataViewType.Double, cursor, currentIndex);
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

            protected ILegacyDataLoader LoadLoader(RepositoryReader rep, string path, bool loadTransforms)
            {
                return ModelFileUtils.LoadLoader(Host, rep, new MultiFileSource(path), loadTransforms);
            }

            protected void SaveLoader(ILegacyDataLoader loader, string path)
            {
                using (var file = Host.CreateOutputFile(path))
                    LoaderUtils.SaveLoader(loader, file);
            }

            protected ILegacyDataLoader CreateAndSaveLoader(Func<IHostEnvironment, IMultiStreamSource, ILegacyDataLoader> defaultLoaderFactory = null)
            {
                var loader = CreateLoader(defaultLoaderFactory);
                if (!string.IsNullOrWhiteSpace(ImplOptions.OutputModelFile))
                {
                    using (var file = Host.CreateOutputFile(ImplOptions.OutputModelFile))
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
                out ILegacyDataLoader pipe)
            {
                // First handle the case where there is no input model file.
                // Everything must come from the command line.

                using (var file = Host.OpenInputFile(ImplOptions.InputModelFile))
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
                    var loaderFactory = ImplOptions.Loader;
                    ILegacyDataLoader trainPipe = null;
                    if (loaderFactory != null)
                    {
                        // The loader is overridden from the command line.
                        pipe = loaderFactory.CreateComponent(Host, new MultiFileSource(ImplOptions.DataFile));
                        if (ImplOptions.LoadTransforms == true)
                        {
                            Host.CheckUserArg(!string.IsNullOrWhiteSpace(ImplOptions.InputModelFile), nameof(ImplOptions.InputModelFile));
                            pipe = LoadTransformChain(pipe);
                        }
                    }
                    else
                    {
                        var loadTrans = ImplOptions.LoadTransforms ?? true;
                        pipe = LoadLoader(rep, ImplOptions.DataFile, loadTrans);
                        if (loadTrans)
                            trainPipe = pipe;
                    }

                    if (Utils.Size(ImplOptions.Transforms) > 0)
                        pipe = LegacyCompositeDataLoader.Create(Host, pipe, ImplOptions.Transforms);

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
                            trainSchema = new RoleMappedSchema(trainPipe.Schema, trainRoleMappings);
                        }
                        // If the role mappings are null, an alternative would be to fail. However the idea
                        // is that the scorer should always still succeed, although perhaps with reduced
                        // functionality, even when the training schema is null, since not all versions of
                        // TLC models will have the role mappings preserved, I believe. And, we do want to
                        // maintain backwards compatibility.
                    }
                }
            }

            protected ILegacyDataLoader CreateLoader(Func<IHostEnvironment, IMultiStreamSource, ILegacyDataLoader> defaultLoaderFactory = null)
            {
                var loader = CreateRawLoader(defaultLoaderFactory);
                loader = CreateTransformChain(loader);
                return loader;
            }

            private ILegacyDataLoader CreateTransformChain(ILegacyDataLoader loader)
            {
                return LegacyCompositeDataLoader.Create(Host, loader, ImplOptions.Transforms);
            }

            protected ILegacyDataLoader CreateRawLoader(
                Func<IHostEnvironment, IMultiStreamSource, ILegacyDataLoader> defaultLoaderFactory = null,
                string dataFile = null)
            {
                if (string.IsNullOrWhiteSpace(dataFile))
                    dataFile = ImplOptions.DataFile;

                ILegacyDataLoader loader;
                if (!string.IsNullOrWhiteSpace(ImplOptions.InputModelFile) && ImplOptions.Loader == null)
                {
                    // Load the loader from the data model.
                    using (var file = Host.OpenInputFile(ImplOptions.InputModelFile))
                    using (var strm = file.OpenReadStream())
                    using (var rep = RepositoryReader.Open(strm, Host))
                        loader = LoadLoader(rep, dataFile, ImplOptions.LoadTransforms ?? true);
                }
                else
                {
                    // Either there is no input model file, or there is, but the loader is overridden.
                    IMultiStreamSource fileSource = new MultiFileSource(dataFile);
                    var loaderFactory = ImplOptions.Loader;
                    if (loaderFactory == null)
                    {
                        var ext = Path.GetExtension(dataFile);
                        var isText =
                            string.Equals(ext, ".txt", StringComparison.OrdinalIgnoreCase) ||
                            string.Equals(ext, ".tlc", StringComparison.OrdinalIgnoreCase);
                        var isBinary = string.Equals(ext, ".idv", StringComparison.OrdinalIgnoreCase);
                        var isTranspose = string.Equals(ext, ".tdv", StringComparison.OrdinalIgnoreCase);

                        return isText ? TextLoader.Create(Host, new TextLoader.Options(), fileSource) :
                               isBinary ? new BinaryLoader(Host, new BinaryLoader.Arguments(), fileSource) :
                               isTranspose ? new TransposeLoader(Host, new TransposeLoader.Arguments(), fileSource) :
                               defaultLoaderFactory != null ? defaultLoaderFactory(Host, fileSource) :
                               TextLoader.Create(Host, new TextLoader.Options(), fileSource);
                    }
                    else
                    {
                        loader = loaderFactory.CreateComponent(Host, fileSource);
                    }

                    if (ImplOptions.LoadTransforms == true)
                    {
                        Host.CheckUserArg(!string.IsNullOrWhiteSpace(ImplOptions.InputModelFile), nameof(ImplOptions.InputModelFile));
                        loader = LoadTransformChain(loader);
                    }
                }
                return loader;
            }

            private ILegacyDataLoader LoadTransformChain(ILegacyDataLoader srcData)
            {
                Host.Assert(!string.IsNullOrWhiteSpace(ImplOptions.InputModelFile));

                using (var file = Host.OpenInputFile(ImplOptions.InputModelFile))
                using (var strm = file.OpenReadStream())
                using (var rep = RepositoryReader.Open(strm, Host))
                using (var pipeLoaderEntry = rep.OpenEntry(ModelFileUtils.DirDataLoaderModel, ModelLoadContext.ModelStreamName))
                using (var ctx = new ModelLoadContext(rep, pipeLoaderEntry, ModelFileUtils.DirDataLoaderModel))
                    return LegacyCompositeDataLoader.Create(Host, ctx, srcData, x => true);
            }
        }
    }

    [BestFriend]
    internal static class LoaderUtils
    {
        /// <summary>
        /// Saves <paramref name="loader"/> to the specified <paramref name="file"/>.
        /// </summary>
        public static void SaveLoader(ILegacyDataLoader loader, IFileHandle file)
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
        public static void SaveLoader(ILegacyDataLoader loader, Stream stream)
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
