// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Internal.Internallearn;

[assembly: LoadableClass(typeof(IDataLoader), typeof(CompositeDataLoader), typeof(CompositeDataLoader.Arguments), typeof(SignatureDataLoader),
    "Composite Data Loader", "CompositeDataLoader", "Composite", "PipeData", "Pipe", "PipeDataLoader")]

[assembly: LoadableClass(typeof(IDataLoader), typeof(CompositeDataLoader), null, typeof(SignatureLoadDataLoader),
    "Pipe DataL Loader", CompositeDataLoader.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// A data loader that wraps an underlying loader plus a sequence of transforms.
    /// It is not valid to have nested <see cref="CompositeDataLoader"/>'s: if a <see cref="CompositeDataLoader"/>
    /// is an underlying loader, the resulting loader will 'flatten' the structure.
    /// The family of <c>Create</c> methods only instantiate <see cref="CompositeDataLoader"/>'s
    /// when there are transforms to keep, otherwise they just return underlying loaders.
    /// </summary>
    public sealed class CompositeDataLoader : IDataLoader, ITransposeDataView
    {
        public sealed class Arguments
        {
            [Argument(ArgumentType.Multiple, HelpText = "The data loader", ShortName = "loader")]
            public SubComponent<IDataLoader, SignatureDataLoader> Loader;

            [Argument(ArgumentType.Multiple, HelpText = "Transform", ShortName = "xf")]
            public KeyValuePair<string, SubComponent<IDataTransform, SignatureDataTransform>>[] Transform;
        }

        internal struct TransformEx
        {
            public readonly string Tag;
            public readonly string ArgsString;
            public readonly IDataTransform Transform;

            public TransformEx(string tag, string argsString, IDataTransform transform)
            {
                Contracts.AssertNonEmpty(tag);
                Contracts.AssertValueOrNull(argsString);
                Contracts.AssertValue(transform, "transform");

                Tag = tag;
                ArgsString = argsString;
                Transform = transform;
            }
        }

        public const string LoaderSignature = "PipeDataLoader";
        private const string RegistrationName = "Composite";
        private const int VersionAddedTags = 0x00010002;
        private const string TransformDirTemplate = "Transform_{0:000}";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "PIPELODR",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Added transform tags and args strings
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        // The composition of loader plus transforms in order.
        private readonly IDataLoader _loader;
        private readonly TransformEx[] _transforms;
        private readonly IDataView _view;
        private readonly ITransposeDataView _tview;
        private readonly ITransposeSchema _tschema;
        private readonly IHost _host;

        /// <summary>
        /// Returns the underlying data view of the composite loader.
        /// This can be used to programmatically explore the chain of transforms that's inside the composite loader.
        /// </summary>
        internal IDataView View { get { return _view; } }

        /// <summary>
        /// Creates a loader according to the specified <paramref name="args"/>.
        /// If there are transforms, then the result will be a <see cref="CompositeDataLoader"/>,
        /// otherwise, it'll be whatever <see cref="IDataLoader"/> is specified in <c>args.loader</c>.
        /// </summary>
        public static IDataLoader Create(IHostEnvironment env, Arguments args, IMultiStreamSource files)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);

            h.CheckValue(args, nameof(args));
            h.CheckUserArg(args.Loader.IsGood(), nameof(args.Loader));
            h.CheckValue(files, nameof(files));

            var loader = args.Loader.CreateInstance(h, files);
            return CreateCore(h, loader, args.Transform);
        }

        /// <summary>
        /// Creates a <see cref="CompositeDataLoader"/> that starts with the <paramref name="srcLoader"/>,
        /// and follows with transforms created from the <paramref name="transformArgs"/> array.
        /// If there are no transforms, the <paramref name="srcLoader"/> is returned.
        /// </summary>
        public static IDataLoader Create(IHostEnvironment env, IDataLoader srcLoader,
            params KeyValuePair<string, SubComponent<IDataTransform, SignatureDataTransform>>[] transformArgs)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);

            h.CheckValue(srcLoader, nameof(srcLoader));
            h.CheckValueOrNull(transformArgs);
            return CreateCore(h, srcLoader, transformArgs);
        }

        private static IDataLoader CreateCore(IHost host, IDataLoader srcLoader,
            KeyValuePair<string, SubComponent<IDataTransform, SignatureDataTransform>>[] transformArgs)
        {
            Contracts.AssertValue(host, "host");
            host.AssertValue(srcLoader, "srcLoader");
            host.AssertValueOrNull(transformArgs);

            if (Utils.Size(transformArgs) == 0)
                return srcLoader;

            var tagData = transformArgs
                .Select(x => new KeyValuePair<string, string>(x.Key, x.Value.ToString()))
                .ToArray();

            // Warn if tags coincide with ones already present in the loader.
            var composite = srcLoader as CompositeDataLoader;
            if (composite != null)
            {
                using (var ch = host.Start("TagValidation"))
                {
                    foreach (var pair in tagData)
                    {
                        if (!string.IsNullOrEmpty(pair.Key) && composite._transforms.Any(x => x.Tag == pair.Key))
                            ch.Warning("The transform with tag '{0}' already exists in the chain", pair.Key);
                    }

                    ch.Done();
                }
            }

            return ApplyTransformsCore(host, srcLoader, tagData,
                (prov, index, data) => transformArgs[index].Value.CreateInstance(prov, data));
        }

        /// <summary>
        /// Appends transforms to the <paramref name="srcLoader"/> and returns a loader that contains these new transforms.
        /// If there are no transforms to append, returns <paramref name="srcLoader"/> intact, otherwise creates a
        /// <see cref="CompositeDataLoader"/>. The transforms are created by sequentially invoking the provided lambda,
        /// one time for each element of <paramref name="tagData"/>.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="srcLoader">The source loader.</param>
        /// <param name="tagData">The array of (tag, creationInfo) pairs. Can be an empty array or null, in which case
        /// the function returns <paramref name="srcLoader"/>.</param>
        /// <param name="createTransform">The delegate to invoke at each transform creation.
        /// Delegate parameters are: host environment, transform index (0 to <c>tagData.Length</c>), source data view.
        /// It should return the <see cref="IDataView"/> that should share the same loader as the source data view.</param>
        /// <returns>The resulting data loader.</returns>
        public static IDataLoader ApplyTransforms(IHostEnvironment env, IDataLoader srcLoader,
            KeyValuePair<string, string>[] tagData, Func<IHostEnvironment, int, IDataView, IDataView> createTransform)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);

            h.CheckValue(srcLoader, nameof(srcLoader));
            h.CheckValueOrNull(tagData);
            h.CheckValue(createTransform, nameof(createTransform));
            if (Utils.Size(tagData) == 0)
                return srcLoader;
            return ApplyTransformsCore(h, srcLoader, tagData, createTransform);
        }

        private static IDataLoader ApplyTransformsCore(IHost host, IDataLoader srcLoader,
            KeyValuePair<string, string>[] tagData, Func<IHostEnvironment, int, IDataView, IDataView> createTransform)
        {
            Contracts.AssertValue(host, "host");
            host.AssertValue(srcLoader, "srcLoader");
            host.AssertNonEmpty(tagData);
            host.AssertValue(createTransform, "createTransform");

            // If the loader is a composite, we need to start with its underlying pipeline end.
            var exes = new List<TransformEx>();
            var composite = srcLoader as CompositeDataLoader;
            IDataView srcView;
            IDataLoader pipeStart;
            if (composite != null)
            {
                srcView = composite._view;
                exes.AddRange(composite._transforms);
                pipeStart = composite._loader;
            }
            else
                srcView = pipeStart = srcLoader;

            IDataView view = srcView;
            using (var ch = host.Start("Transforms"))
            {
                int count = Utils.Size(tagData);
                var newlyCreated = new List<TransformEx>();
                for (int i = 0; i < count; i++)
                {
                    // REVIEW: this might cause silent automatic tag conflicts if the pipeline is short-circuited.
                    // Maybe it's better to allow empty tags?
                    var tag = tagData[i].Key;
                    if (string.IsNullOrEmpty(tag))
                        tag = GenerateTag(exes.Count);

                    var newDataView = createTransform(host, i, view);
                    // Append the newly created transforms to the exes list.
                    // If the newTransform is a 'no-op' transform, i.e. equal to the original view,
                    // the exes array will not be modified: there's no reason to record details of a no-op transform,
                    // especially since this would overwrite the useful details of the upstream transform.
                    newlyCreated.Clear();
                    IDataView curDataView = newDataView;
                    while (true)
                    {
                        var cur = curDataView as IDataTransform;
                        if (cur == null)
                        {
                            // We reached all the way back to the pipe start. The exes accumulated so far are irrelevant.
                            ch.Check(curDataView == pipeStart,
                                "The transform has corrupted the chain (chain no longer starts with the same loader).");
                            exes.Clear();
                            break;
                        }

                        int index = exes.FindLastIndex(x => x.Transform == cur);
                        if (index >= 0)
                        {
                            // We found a transform in exes to attach to.
                            if (index < exes.Count - 1)
                            {
                                // The transform short-circuited some of the existing ones, remove them.
                                exes.RemoveRange(index + 1, exes.Count - index - 1);
                            }
                            break;
                        }

                        newlyCreated.Add(new TransformEx(tag, tagData[i].Value, cur));
                        curDataView = cur.Source;
                    }

                    newlyCreated.Reverse();
                    exes.AddRange(newlyCreated);

                    view = newDataView;
                }

                ch.Done();
            }

            return view == srcView ? srcLoader : new CompositeDataLoader(host, exes.ToArray());
        }

        /// <summary>
        /// Apply one transform to the data loader, and returns a (composite) data loader that contains the result.
        /// The transform is created by invoking the lambda for a data source, and it should return an
        /// <see cref="IDataView"/> that shares the same loader as the provided source.
        /// </summary>
        public static IDataLoader ApplyTransform(IHostEnvironment env, IDataLoader srcLoader,
            string tag, string creationArgs, Func<IHostEnvironment, IDataView, IDataView> createTransform)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);

            h.CheckValue(srcLoader, nameof(srcLoader));
            h.CheckValueOrNull(tag);
            h.CheckValueOrNull(creationArgs);
            h.CheckValue(createTransform, nameof(createTransform));
            var tagData = new[] { new KeyValuePair<string, string>(tag, creationArgs) };
            return ApplyTransformsCore(env.Register(RegistrationName), srcLoader, tagData, (e, index, data) => createTransform(e, data));
        }

        /// <summary>
        /// Loads the entire composite data loader (loader + transforms) from the context.
        /// If there are no transforms, the underlying loader is returned.
        /// </summary>
        public static IDataLoader Create(IHostEnvironment env, ModelLoadContext ctx, IMultiStreamSource files)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);

            h.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            h.CheckValue(files, nameof(files));

            using (var ch = h.Start("Components"))
            {
                // First, load the loader.
                IDataLoader loader;
                ctx.LoadModel<IDataLoader, SignatureLoadDataLoader>(h, out loader, "Loader", files);

                // Now the transforms.
                h.Assert(!(loader is CompositeDataLoader));
                var result = LoadTransforms(ctx, loader, h, x => true);
                ch.Done();
                return result;
            }
        }

        /// <summary>
        /// Creates a <see cref="IDataLoader"/> from the specified source loader, followed by
        /// the transforms that are loaded from the <paramref name="ctx"/>, tags filtered by
        /// by the <paramref name="isTransformTagAccepted"/>.
        /// If the <paramref name="ctx"/> contains no accepted transforms, the <paramref name="srcLoader"/> is
        /// returned intact.
        /// </summary>
        public static IDataLoader Create(IHostEnvironment env, ModelLoadContext ctx,
            IDataLoader srcLoader, Func<string, bool> isTransformTagAccepted)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);

            h.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            h.CheckValue(srcLoader, nameof(srcLoader));
            h.CheckValue(isTransformTagAccepted, nameof(isTransformTagAccepted));

            return LoadTransforms(ctx, srcLoader, h, isTransformTagAccepted);
        }

        /// <summary>
        /// Loads all transforms from the <paramref name="ctx"/> that pass the <paramref name="isTransformTagAccepted"/> test,
        /// applies them sequentially to the <paramref name="srcView"/>, and returns the resulting data view.
        /// If there are no transforms in <paramref name="ctx"/> that are accepted, returns the original <paramref name="srcView"/>.
        /// The difference from the <c>Create</c> method above is that:
        /// - it doesn't wrap the results into a loader, just returns the last transform in the chain.
        /// - it accepts <see cref="IDataView"/> as input, not necessarily a loader.
        /// - it throws away the tag information.
        /// - it doesn't throw if the context is not representing a <see cref="CompositeDataLoader"/>: in this case it's assumed that no transforms
        ///   meet the test, and the <paramref name="srcView"/> is returned.
        /// Essentially, this is a helper method for the LoadTransform class.
        /// </summary>
        public static IDataView LoadSelectedTransforms(ModelLoadContext ctx, IDataView srcView, IHostEnvironment env, Func<string, bool> isTransformTagAccepted)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);

            h.CheckValue(ctx, nameof(ctx));
            h.Check(ctx.Reader.BaseStream.Position == ctx.FpMin + ctx.Header.FpModel);
            var ver = GetVersionInfo();
            if (ctx.Header.ModelSignature != ver.ModelSignature)
            {
                using (var ch = h.Start("ModelCheck"))
                {
                    ch.Info("The data model doesn't contain transforms.");
                    ch.Done();
                }
                return srcView;
            }
            ModelHeader.CheckVersionInfo(ref ctx.Header, ver);

            h.CheckValue(srcView, nameof(srcView));
            h.CheckValue(isTransformTagAccepted, nameof(isTransformTagAccepted));

            // *** Binary format ***
            // int: sizeof(Float)
            // int: number of transforms
            // foreach transform: (starting from version VersionAddedTags)
            //     string: tag
            //     string: args string

            int cbFloat = ctx.Reader.ReadInt32();
            h.CheckDecode(cbFloat == sizeof(Float));

            int cxf = ctx.Reader.ReadInt32();
            h.CheckDecode(cxf >= 0);

            bool hasTags = ctx.Header.ModelVerReadable >= VersionAddedTags;
            var curView = srcView;
            for (int i = 0; i < cxf; i++)
            {
                string tag = "";
                if (hasTags)
                {
                    tag = ctx.LoadNonEmptyString();
                    ctx.LoadStringOrNull(); // ignore the args string
                }
                if (!isTransformTagAccepted(tag))
                    continue;

                IDataTransform xf;
                ctx.LoadModel<IDataTransform, SignatureLoadDataTransform>(env, out xf,
                    string.Format(TransformDirTemplate, i), curView);
                curView = xf;
            }

            return curView;
        }

        private CompositeDataLoader(IHost host, TransformEx[] transforms)
        {
            Contracts.AssertValue(host, "host");
            _host = host;
            _host.AssertNonEmpty(transforms);

            _view = transforms[transforms.Length - 1].Transform;
            _tview = _view as ITransposeDataView;
            _tschema = _tview == null ? new TransposerUtils.SimpleTransposeSchema(_view.Schema) : _tview.TransposeSchema;

            var srcLoader = transforms[0].Transform.Source as IDataLoader;

#if DEBUG
            // Assert that the transforms array is consistent: first one starts with loader,
            // they are chained together, the loader is not a composite.
            for (int i = 1; i < transforms.Length; i++)
                _host.Assert(transforms[i].Transform.Source == transforms[i - 1].Transform, "Transforms are not linked");

            _host.AssertValue(srcLoader, "loader", "Transform chain doesn't start with a loader");
            _host.Assert(!(srcLoader is CompositeDataLoader), "Can't have composite source loader");
#endif

            _loader = srcLoader;
            _transforms = transforms;
        }

        /// <summary>
        /// Loads all transforms from the <paramref name="ctx"/> that pass the <paramref name="isTransformTagAccepted"/> test,
        /// applies them sequentially to the <paramref name="srcLoader"/>, and returns the (composite) data loader.
        /// </summary>
        private static IDataLoader LoadTransforms(ModelLoadContext ctx, IDataLoader srcLoader, IHost host, Func<string, bool> isTransformTagAccepted)
        {
            Contracts.AssertValue(host, "host");
            host.AssertValue(srcLoader);
            host.AssertValue(ctx);

            // *** Binary format ***
            // int: sizeof(Float)
            // int: number of transforms
            // foreach transform: (starting from version VersionAddedTags)
            //     string: tag
            //     string: args string

            int cbFloat = ctx.Reader.ReadInt32();
            host.CheckDecode(cbFloat == sizeof(Float));

            int cxf = ctx.Reader.ReadInt32();
            host.CheckDecode(cxf >= 0);

            bool hasTags = ctx.Header.ModelVerReadable >= VersionAddedTags;
            var tagData = new List<KeyValuePair<string, string>>();
            var acceptedIds = new List<int>();

            for (int i = 0; i < cxf; i++)
            {
                string tag = "";
                string argsString = null;
                if (hasTags)
                {
                    tag = ctx.LoadNonEmptyString();
                    argsString = ctx.LoadStringOrNull();
                }
                if (!isTransformTagAccepted(tag))
                    continue;

                acceptedIds.Add(i);
                tagData.Add(new KeyValuePair<string, string>(tag, argsString));
            }

            host.Assert(tagData.Count == acceptedIds.Count);
            if (tagData.Count == 0)
                return srcLoader;

            return ApplyTransformsCore(host, srcLoader, tagData.ToArray(),
                (h, index, data) =>
                {
                    IDataTransform xf;
                    ctx.LoadModel<IDataTransform, SignatureLoadDataTransform>(host, out xf,
                        string.Format(TransformDirTemplate, acceptedIds[index]), data);
                    return xf;
                });
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            SaveCore(ctx, _loader.Save, _transforms);
        }

        /// <summary>
        /// Save the loader and transforms (if any) to the repository.
        /// This is intended to be used by API, where the components are not part of the same
        /// <see cref="CompositeDataLoader"/>.
        /// </summary>
        /// <param name="env">Environment context</param>
        /// <param name="ctx">The context to write to.</param>
        /// <param name="loaderSaveAction">The code to save the loader.</param>
        /// <param name="transforms">The transforms. Empty list and null are both allowed.</param>
        public static void SavePipe(IHostEnvironment env, ModelSaveContext ctx, Action<ModelSaveContext> loaderSaveAction, IList<IDataTransform> transforms)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);

            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(loaderSaveAction, nameof(loaderSaveAction));
            h.CheckValueOrNull(transforms);

            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            var exes = transforms
                .Select((xf, i) => new TransformEx(GenerateTag(i), null, xf))
                .ToArray();
            SaveCore(ctx, loaderSaveAction, exes);
        }

        private static void SaveCore(ModelSaveContext ctx, Action<ModelSaveContext> loaderSaveAction, TransformEx[] transforms)
        {
            Contracts.AssertValue(ctx);
            Contracts.AssertValue(loaderSaveAction);
            Contracts.AssertValueOrNull(transforms);

            // *** Binary format ***
            // int: sizeof(Float)
            // int: number of transforms
            // foreach transform: (starting from version VersionAddedTags)
            //     string: tag
            //     string: args string

            ctx.Writer.Write(sizeof(Float));
            ctx.Writer.Write(transforms.Length);

            using (var loaderCtx = new ModelSaveContext(ctx.Repository, Path.Combine(ctx.Directory ?? "", "Loader"), ModelLoadContext.ModelStreamName))
            {
                loaderSaveAction(loaderCtx);
                loaderCtx.Done();
            }

            for (int i = 0; i < transforms.Length; i++)
            {
                var dirName = string.Format(TransformDirTemplate, i);
                ctx.SaveModel(transforms[i].Transform, dirName);

                Contracts.AssertNonEmpty(transforms[i].Tag);
                ctx.SaveNonEmptyString(transforms[i].Tag);
                ctx.SaveStringOrNull(transforms[i].ArgsString);
            }
        }

        private static string GenerateTag(int index)
        {
            return string.Format("xf{0:00}", index);
        }

        public long? GetRowCount(bool lazy = true)
        {
            return _view.GetRowCount(lazy);
        }

        public bool CanShuffle
        {
            get { return _view.CanShuffle; }
        }

        public ISchema Schema
        {
            get { return _view.Schema; }
        }

        public ITransposeSchema TransposeSchema
        {
            get { return _tschema; }
        }

        public IRowCursor GetRowCursor(Func<int, bool> predicate, IRandom rand = null)
        {
            _host.CheckValue(predicate, nameof(predicate));
            _host.CheckValueOrNull(rand);
            return _view.GetRowCursor(predicate, rand);
        }

        public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator,
            Func<int, bool> predicate, int n, IRandom rand = null)
        {
            _host.CheckValue(predicate, nameof(predicate));
            _host.CheckValueOrNull(rand);
            return _view.GetRowCursorSet(out consolidator, predicate, n, rand);
        }

        public ISlotCursor GetSlotCursor(int col)
        {
            _host.CheckParam(0 <= col && col < Schema.ColumnCount, nameof(col));
            if (_tschema == null || _tschema.GetSlotType(col) == null)
            {
                throw _host.ExceptParam(nameof(col), "Bad call to GetSlotCursor on untransposable column '{0}'",
                    Schema.GetColumnName(col));
            }
            _host.AssertValue(_tview);
            return _tview.GetSlotCursor(col);
        }
    }
}
