// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Reflection;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.ML.Runtime;
using Microsoft.ML.StaticPipe;

namespace Microsoft.ML.CodeAnalyzer.Tests.Helpers
{
    internal static class AdditionalMetadataReferences
    {
        internal static readonly MetadataReference StandardReference = MetadataReference.CreateFromFile(Assembly.Load("netstandard, Version=2.0.0.0").Location);
        internal static readonly MetadataReference RuntimeReference = MetadataReference.CreateFromFile(Assembly.Load("System.Runtime, Version=0.0.0.0").Location);
        internal static readonly MetadataReference CSharpSymbolsReference = RefFromType<CSharpCompilation>();
        internal static readonly MetadataReference MSDataDataViewReference = RefFromType<IDataView>();
        internal static readonly MetadataReference MLNetCoreReference = RefFromType<IHostEnvironment>();
        internal static readonly MetadataReference MLNetDataReference = RefFromType<MLContext>();
        internal static readonly MetadataReference MLNetStaticPipeReference = RefFromType<CategoricalHashStaticExtensions.OneHotHashVectorOutputKind>();

        internal static MetadataReference RefFromType<TType>()
            => MetadataReference.CreateFromFile(typeof(TType).Assembly.Location);
    }
}
