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
}
