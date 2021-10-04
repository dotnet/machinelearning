// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Reflection;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(typeof(ITransformer), typeof(LambdaTransform), null, typeof(SignatureLoadModel), "", LambdaTransform.LoaderSignature)]

namespace Microsoft.ML.Transforms
{
    using Conditional = System.Diagnostics.ConditionalAttribute;

    /// <summary>
    /// Utility class for creating transforms easily.
    /// </summary>
    [BestFriend]
    internal static class LambdaTransform
    {
        internal const string LoaderSignature = "CustomTransformer";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "CUSTOMXF",
                //verWrittenCur: 0x00010001,  // Initial
                verWrittenCur: 0x00010002,  // Added name of assembly in which the contractName is present
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(LambdaTransform).Assembly.FullName);
        }

        private const uint VerAssemblyNameSaved = 0x00010002;

        internal static void SaveCustomTransformer(IExceptionContext ectx, ModelSaveContext ctx, string contractName, string contractAssembly)
        {
            ectx.CheckValue(ctx, nameof(ctx));
            ectx.CheckValue(contractName, nameof(contractName));

            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            ctx.SaveString(contractName);
            ctx.SaveString(contractAssembly);
        }

        // Factory for SignatureLoadModel.
        private static ITransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            var contractName = ctx.LoadString();
            if (ctx.Header.ModelVerWritten >= VerAssemblyNameSaved)
            {
                var contractAssembly = ctx.LoadString();
                Assembly assembly = Assembly.Load(contractAssembly);
                env.ComponentCatalog.RegisterAssembly(assembly);
            }

            object factoryObject = env.ComponentCatalog.GetExtensionValue(env, typeof(CustomMappingFactoryAttributeAttribute), contractName);
            if (!(factoryObject is ICustomMappingFactory mappingFactory))
            {
                throw env.Except($"The class with contract '{contractName}' must derive from '{typeof(CustomMappingFactory<,>).FullName}' or from '{typeof(StatefulCustomMappingFactory<,,>).FullName}'.");
            }

            return mappingFactory.CreateTransformer(env, contractName);
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
        public static IDataView CreateMap<TSrc, TDst, TState>(IHostEnvironment env, IDataView source,
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
                }, initStateAction, inputSchemaDefinition, outputSchemaDefinition);
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
        public static IDataView CreateFilter<TSrc, TState>(IHostEnvironment env, IDataView source,
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
                (src, dst, state) => filterFunc(src, state), initStateAction, inputSchemaDefinition);
        }
    }
}
