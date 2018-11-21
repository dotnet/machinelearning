// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.EntryPoints;

[assembly: LoadableClass(typeof(void), typeof(Cache), null, typeof(SignatureEntryPointModule), "Cache")]
namespace Microsoft.ML.Runtime.EntryPoints
{
    public static class Cache
    {
        public enum CachingType
        {
            Memory,
            Disk
        }

        public sealed class CacheInput : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "Caching strategy", SortOrder = 2)]
            public CachingType Caching;
        }

        public sealed class CacheOutput
        {
            [TlcModule.Output(Desc = "Dataset", SortOrder = 1)]
            public IDataView OutputData;
        }

        [TlcModule.EntryPoint(Name = "Transforms.DataCache", Desc = "Caches using the specified cache option.", UserName = "Cache Data")]
        public static CacheOutput CacheData(IHostEnvironment env, CacheInput input)
        {
            const string registrationName = "CreateCache";
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(registrationName);
            host.CheckValue(input, nameof(input));
            host.CheckValue(input.Data, nameof(input.Data));

            IDataView data;

            switch (input.Caching)
            {
                case CachingType.Memory:
                    data = new CacheDataView(env, input.Data, null);
                    break;
                case CachingType.Disk:
                    var args = new BinarySaver.Arguments();
                    args.Compression = CompressionKind.Default;
                    args.Silent = true;

                    var saver = new BinarySaver(host, args);
                    var schema = input.Data.Schema;

                    var cols = new List<int>();
                    for (int i = 0; i < schema.ColumnCount; i++)
                    {
                        var type = schema.GetColumnType(i);
                        if (saver.IsColumnSavable(type))
                            cols.Add(i);
                    }

                    // We are not disposing the fileHandle because we want it to stay around for the execution of the graph.
                    // It will be disposed when the environment is disposed.
                    var fileHandle = host.CreateTempFile();

                    using (var stream = fileHandle.CreateWriteStream())
                        saver.SaveData(stream, input.Data, cols.ToArray());
                    data = new BinaryLoader(host, new BinaryLoader.Arguments(), fileHandle.OpenReadStream());
                    break;
                default:
                    throw host.ExceptValue(nameof(input.Caching), $"Unrecognized caching option '{input.Caching}'");
            }

            return new CacheOutput() { OutputData = data };
        }
    }
}
