// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#nullable enable

using System.Collections.Immutable;
using System.IO;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Testing;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.CodeAnalyzer.Tests.Helpers
{
    internal static class AdditionalMetadataReferences
    {
#if NETCOREAPP
        internal static readonly ReferenceAssemblies DefaultReferenceAssemblies = ReferenceAssemblies.Default
            .AddPackages(ImmutableArray.Create(new PackageIdentity("System.Memory", "4.5.1")));
#else
        internal static readonly ReferenceAssemblies DefaultReferenceAssemblies = ReferenceAssemblies.NetFramework.Net472.Default
            .AddPackages(ImmutableArray.Create(new PackageIdentity("System.Memory", "4.5.1")));
#endif

        internal static readonly MetadataReference MSDataDataViewReference = RefFromType<IDataView>();
        internal static readonly MetadataReference MLNetCoreReference = RefFromType<IHostEnvironment>();
        internal static readonly MetadataReference MLNetDataReference = RefFromType<MLContext>();

        internal static MetadataReference RefFromType<TType>()
        {
            var location = typeof(TType).Assembly.Location;
            var documentationProvider = XmlDocumentationProvider.CreateFromFile(Path.ChangeExtension(location, ".pdb"));
            return MetadataReference.CreateFromFile(location, documentation: documentationProvider);
        }
    }
}
