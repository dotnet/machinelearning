// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.Data.DataView;

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// Place this attribute onto a type to cause it to be considered a custom mapping transformer factory.
    /// </summary>
    [AttributeUsage(AttributeTargets.Class)]
    public sealed class CustomMappingTransformerFactoryAttribute : ExtensionBaseAttribute
    {
        public CustomMappingTransformerFactoryAttribute(string contractName)
            : base(contractName)
        {
        }
    }

    internal interface ICustomMappingTransformerFactory
    {
        ITransformer CreateTransformerObject(IHostEnvironment env, string contractName);
    }

    /// <summary>
    /// The base type for custom mapping transformer factories.
    /// </summary>
    /// <typeparam name="TSrc">The type that describes what 'source' columns are consumed from the input <see cref="IDataView"/>.</typeparam>
    /// <typeparam name="TDst">The type that describes what new columns are added by this transform.</typeparam>
    public abstract class CustomMappingTransformerFactory<TSrc, TDst> : ICustomMappingTransformerFactory
        where TSrc : class, new()
        where TDst : class, new()
    {
        public abstract Action<TSrc, TDst> GetTransformer();

        ITransformer ICustomMappingTransformerFactory.CreateTransformerObject(IHostEnvironment env, string contractName)
        {
            Action<TSrc, TDst> mapAction = GetTransformer();
            return new CustomMappingTransformer<TSrc, TDst>(env, mapAction, contractName);
        }
    }
}
