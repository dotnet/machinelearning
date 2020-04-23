// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// Place this attribute onto a type to cause it to be considered a custom mapping factory.
    /// </summary>
    [AttributeUsage(AttributeTargets.Class)]
    public sealed class CustomMappingFactoryAttributeAttribute : ExtensionBaseAttribute
    {
        public CustomMappingFactoryAttributeAttribute(string contractName)
            : base(contractName)
        {
        }
    }

    internal interface ICustomMappingFactory
    {
        ITransformer CreateTransformer(IHostEnvironment env, string contractName);
    }

    /// <summary>
    /// The base type for custom mapping factories.
    /// </summary>
    /// <typeparam name="TSrc">The type that describes what 'source' columns are consumed from the input <see cref="IDataView"/>.</typeparam>
    /// <typeparam name="TDst">The type that describes what new columns are added by this transform.</typeparam>
    public abstract class CustomMappingFactory<TSrc, TDst> : ICustomMappingFactory
        where TSrc : class, new()
        where TDst : class, new()
    {
        /// <summary>
        /// Returns the mapping delegate that maps from <typeparamref name="TSrc"/> inputs to <typeparamref name="TDst"/> outputs.
        /// </summary>
        public abstract Action<TSrc, TDst> GetMapping();

        ITransformer ICustomMappingFactory.CreateTransformer(IHostEnvironment env, string contractName)
        {
            Action<TSrc, TDst> mapAction = GetMapping();
            return new CustomMappingTransformer<TSrc, TDst>(env, mapAction, contractName);
        }
    }

    /// <summary>
    /// The base type for stateful custom mapping factories.
    /// </summary>
    /// <typeparam name="TSrc">The type that describes what 'source' columns are consumed from the input <see cref="IDataView"/>.</typeparam>
    /// <typeparam name="TDst">The type that describes what new columns are added by this transform.</typeparam>
    /// <typeparam name="TState">The type that describes the state object the mapping uses.</typeparam>
    public abstract class StatefulCustomMappingFactory<TSrc, TDst, TState> : ICustomMappingFactory
        where TSrc : class, new()
        where TDst : class, new()
        where TState : class, new()
    {
        /// <summary>
        /// Returns the mapping delegate that maps from a <typeparamref name="TSrc"/> input and a state object of type <typeparamref name="TState"/>,
        /// to a <typeparamref name="TDst"/> output.
        /// </summary>
        public abstract Action<TSrc, TDst, TState> GetMapping();

        /// <summary>
        /// Returns an action that is called once before the row cursor is initialized, to initialize the state object used by the cursor.
        /// </summary>
        public abstract Action<TState> GetStateInitAction();

        ITransformer ICustomMappingFactory.CreateTransformer(IHostEnvironment env, string contractName)
        {
            Action<TSrc, TDst, TState> mapAction = GetMapping();
            Action<TState> stateInitAction = GetStateInitAction();
            return new StatefulCustomMappingTransformer<TSrc, TDst, TState>(env, mapAction, contractName, stateInitAction);
        }
    }
}
