// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.ComponentModel.Composition.Hosting;
using System.IO;
using System.Text;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(typeof(ITransformer), typeof(LambdaTransform), null, typeof(SignatureLoadModel), "", LambdaTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.Api
{
    using Conditional = System.Diagnostics.ConditionalAttribute;

    /// <summary>
    /// Utility class for creating transforms easily.
    /// </summary>
    public static class LambdaTransform
    {
        /// <summary>
        /// A delegate type to create a persistent transform, utilized by the creation functions
        /// as a callback to reconstitute a transform from binary data.
        /// </summary>
        /// <param name="reader">The binary stream from which the load is persisted</param>
        /// <param name="env">The host environment</param>
        /// <param name="input">The dataview this transform should be persisted on</param>
        /// <returns>A transform of the input data, as parameterized by the binary input
        /// stream</returns>
        public delegate ITransformTemplate LoadDelegate(BinaryReader reader, IHostEnvironment env, IDataView input);

        internal const string LoaderSignature = "CustomTransformer";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "CUSTOMXF",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(LambdaTransform).Assembly.FullName);
        }

        internal static void SaveCustomTransformer(IExceptionContext ectx, ModelSaveContext ctx, string contractName)
        {
            ectx.CheckValue(ctx, nameof(ctx));
            ectx.CheckValue(contractName, nameof(contractName));

            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            ctx.SaveString(contractName);
        }

        // Factory for SignatureLoadModel.
        private static ITransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            var contractName = ctx.LoadString();

            var composition = env.GetCompositionContainer();
            ITransformer transformer = composition.GetExportedValue<ITransformer>(contractName);
            return transformer;
        }

        /// <summary>
        /// This is a 'stateful non-savable' version of the map transform: the mapping function is guaranteed to be invoked once per
        /// every row of the data set, in sequence; one user-defined state object will be allocated per cursor and passed to the
        /// map function every time. If <typeparamref name="TSrc"/>, <typeparamref name="TDst"/>, or
        /// <typeparamref name="TState"/> implement the <see cref="IDisposable" /> interface, they will be disposed after use.
        /// </summary>
        /// <typeparam name="TSrc">The type that describes what 'source' columns are consumed from the
        /// input <see cref="IDataView"/>.</typeparam>
        /// <typeparam name="TState">The type of the state object to allocate per cursor.</typeparam>
        /// <typeparam name="TDst">The type that describes what new columns are added by this transform.</typeparam>
        /// <param name="env">The host environment to use.</param>
        /// <param name="source">The input data to apply transformation to.</param>
        /// <param name="mapAction">The function that performs the transformation. The function should transform its <typeparamref name="TSrc"/>
        /// argument into its <typeparamref name="TDst"/> argument and can utilize the per-cursor <typeparamref name="TState"/> state.</param>
        /// <param name="initStateAction">The function that is called once per cursor to initialize state. Can be null.</param>
        /// <param name="inputSchemaDefinition">The optional input schema. If <c>null</c>, the schema is
        /// inferred from the <typeparamref name="TSrc"/> type.</param>
        /// <param name="outputSchemaDefinition">The optional output schema. If <c>null</c>, the schema is
        /// inferred from the <typeparamref name="TDst"/> type.</param>
        public static ITransformTemplate CreateMap<TSrc, TDst, TState>(IHostEnvironment env, IDataView source,
            Action<TSrc, TDst, TState> mapAction, Action<TState> initStateAction,
            SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
            where TSrc : class, new()
            where TDst : class, new()
            where TState : class, new()
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(source, nameof(source));
            env.CheckValue(mapAction, nameof(mapAction));
            env.CheckValueOrNull(initStateAction);
            env.CheckValueOrNull(inputSchemaDefinition);
            env.CheckValueOrNull(outputSchemaDefinition);

            return new StatefulFilterTransform<TSrc, TDst, TState>(env, source,
                (src, dst, state) =>
                {
                    mapAction(src, dst, state);
                    return true;
                }, initStateAction, null, null, inputSchemaDefinition, outputSchemaDefinition);
        }

        /// <summary>
        /// This creates a filter transform that can 'accept' or 'decline' any row of the data based on the contents of the row
        /// or state of the cursor.
        /// This is a 'stateful non-savable' version of the filter: the filter function is guaranteed to be invoked once per
        /// every row of the data set, in sequence (non-parallelizable); one user-defined state object will be allocated per cursor and passed to the
        /// filter function every time.
        /// If <typeparamref name="TSrc"/> or <typeparamref name="TState"/> implement the <see cref="IDisposable" /> interface, they will be disposed after use.
        /// </summary>
        /// <typeparam name="TSrc">The type that describes what 'source' columns are consumed from the
        /// input <see cref="IDataView"/>.</typeparam>
        /// <typeparam name="TState">The type of the state object to allocate per cursor.</typeparam>
        /// <param name="env">The host environment to use.</param>
        /// <param name="source">The input data to apply transformation to.</param>
        /// <param name="filterFunc">The user-defined function that determines whether to keep the row or discard it. First parameter
        /// is the current row's contents, the second parameter is the cursor-specific state object.</param>
        /// <param name="initStateAction">The function that is called once per cursor to initialize state. Can be null.</param>
        /// <param name="inputSchemaDefinition">The optional input schema. If <c>null</c>, the schema is
        /// inferred from the <typeparamref name="TSrc"/> type.</param>
        /// <returns></returns>
        public static ITransformTemplate CreateFilter<TSrc, TState>(IHostEnvironment env, IDataView source,
            Func<TSrc, TState, bool> filterFunc, Action<TState> initStateAction, SchemaDefinition inputSchemaDefinition = null)
            where TSrc : class, new()
            where TState : class, new()
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(source, nameof(source));
            env.CheckValue(filterFunc, nameof(filterFunc));
            env.CheckValueOrNull(initStateAction);
            env.CheckValueOrNull(inputSchemaDefinition);

            return new StatefulFilterTransform<TSrc, object, TState>(env, source,
                (src, dst, state) => filterFunc(src, state), initStateAction, null, null, inputSchemaDefinition);
        }

        /// <summary>
        /// This creates a filter transform that can 'accept' or 'decline' any row of the data based on the contents of the row
        /// or state of the cursor.
        /// This is a 'stateful savable' version of the filter: the filter function is guaranteed to be invoked once per
        /// every row of the data set, in sequence (non-parallelizable); one user-defined state object will be allocated per cursor and passed to the
        /// filter function every time; save and load routines must be provided.
        /// If <typeparamref name="TSrc"/> or <typeparamref name="TState"/> implement the <see cref="IDisposable" /> interface, they will be disposed after use.
        /// </summary>
        /// <typeparam name="TSrc">The type that describes what 'source' columns are consumed from the
        /// input <see cref="IDataView"/>.</typeparam>
        /// <typeparam name="TState">The type of the state object to allocate per cursor.</typeparam>
        /// <param name="env">The host environment to use.</param>
        /// <param name="source">The input data to apply transformation to.</param>
        /// <param name="filterFunc">The user-defined function that determines whether to keep the row or discard it. First parameter
        /// is the current row's contents, the second parameter is the cursor-specific state object.</param>
        /// <param name="initStateAction">The function that is called once per cursor to initialize state. Can be null.</param>
        /// <param name="saveAction">An action that allows us to save state to the serialization stream</param>
        /// <param name="loadFunc">A function that given the serialization stream and a data view, returns
        /// an <see cref="ITransformTemplate"/>. The intent is, this returned object should itself be the same
        /// as if we had recreated it using this method, but this is impossible to enforce. This transform
        /// will do its best to save a description of this method through assembly qualified names of the defining
        /// class, method name, and generic type parameters (if any), and then recover this same method on load,
        /// so it should be a static non-lambda method that this assembly can legally call.</param>
        /// <param name="inputSchemaDefinition">The optional input schema. If <c>null</c>, the schema is
        /// inferred from the <typeparamref name="TSrc"/> type.</param>
        /// <returns></returns>
        public static ITransformTemplate CreateFilter<TSrc, TState>(IHostEnvironment env, IDataView source,
            Func<TSrc, TState, bool> filterFunc, Action<TState> initStateAction,
            Action<BinaryWriter> saveAction, LoadDelegate loadFunc,
            SchemaDefinition inputSchemaDefinition = null)
            where TSrc : class, new()
            where TState : class, new()
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(source, nameof(source));
            env.CheckValue(filterFunc, nameof(filterFunc));
            env.CheckValue(initStateAction, nameof(initStateAction));
            env.CheckValue(saveAction, nameof(saveAction));
            env.CheckValue(loadFunc, nameof(loadFunc));
            env.CheckValueOrNull(inputSchemaDefinition);

            return new StatefulFilterTransform<TSrc, object, TState>(env, source,
                (src, dst, state) => filterFunc(src, state), initStateAction, saveAction, loadFunc, inputSchemaDefinition);
        }
    }

    /// <summary>
    /// Defines common ancestor for various flavors of lambda-based user-defined transforms that may or may not be
    /// serializable.
    ///
    /// In order for the transform to be serializable, the user should specify a save and load delegate.
    /// Specifically, for this the user has to provide the following things:
    ///  * a custom save action that serializes the transform 'state' to the binary writer.
    ///  * a custom load action that de-serializes the transform from the binary reader. This must be a public static method of a public class.
    /// </summary>
    internal abstract class LambdaTransformBase
    {
        private readonly Action<BinaryWriter> _saveAction;
        private readonly byte[] _loadFuncBytes;
        protected readonly IHost Host;
        protected LambdaTransformBase(IHostEnvironment env, string name, Action<BinaryWriter> saveAction, LambdaTransform.LoadDelegate loadFunc)
        {
            Contracts.AssertValue(env);
            env.AssertNonWhiteSpace(name);
            Host = env.Register(name);

            Host.Assert((saveAction == null) == (loadFunc == null));

            if (saveAction != null)
            {
                _saveAction = saveAction;
                // First, verify as best we can, that we can recover the function, by attempting to do it once.
                _loadFuncBytes = SerializableLambdaTransform.GetSerializedStaticDelegate(loadFunc);
                Exception error;
                var recoveredLoadFunc = SerializableLambdaTransform.DeserializeStaticDelegateOrNull(Host, _loadFuncBytes, out error);
                if (recoveredLoadFunc == null)
                {
                    Host.AssertValue(error);
                    throw Host.Except(error, "Load function does not appear recoverable");
                }
            }

            AssertConsistentSerializable();
        }

        /// <summary>
        /// The 'reapply' constructor.
        /// </summary>
        protected LambdaTransformBase(IHostEnvironment env, string name, LambdaTransformBase source)
        {
            Contracts.AssertValue(env);
            env.AssertNonWhiteSpace(name);
            Host = env.Register(name);
            _saveAction = source._saveAction;
            _loadFuncBytes = source._loadFuncBytes;

            AssertConsistentSerializable();
        }

        public void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.Check(CanSave(), "Cannot save this transform as it was not specified as being savable");
            ctx.CheckAtModel();
            ctx.SetVersionInfo(SerializableLambdaTransform.GetVersionInfo());

            // *** Binary format ***
            // int: Number of bytes the load method was serialized to
            // byte[n]: The serialized load method info
            // <arbitrary>: Arbitrary bytes saved by the save action

            Host.AssertNonEmpty(_loadFuncBytes);
            ctx.Writer.WriteByteArray(_loadFuncBytes);

            using (var ms = new MemoryStream())
            {
                using (var writer = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
                    _saveAction(writer);
                ctx.Writer.WriteByteArray(ms.ToArray());
            }
        }

        private bool CanSave()
        {
            return _saveAction != null;
        }

        [Conditional("DEBUG")]
        private void AssertConsistentSerializable()
        {
#if DEBUG
            // This class can be either serializable, or not. Some fields should
            // be null iff the transform is not savable.
            bool canSave = CanSave();
            Host.Assert(canSave == (_saveAction != null));
            Host.Assert(canSave == (_loadFuncBytes != null));
#endif
        }
    }
}
