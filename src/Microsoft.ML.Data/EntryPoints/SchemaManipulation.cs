// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(typeof(void), typeof(SchemaManipulation), null, typeof(SignatureEntryPointModule), "SchemaManipulation")]

namespace Microsoft.ML.EntryPoints
{
    internal static class SchemaManipulation
    {
        [TlcModule.EntryPoint(Name = "Transforms.ColumnConcatenator", Desc = ColumnConcatenatingTransformer.Summary, UserName = ColumnConcatenatingTransformer.UserName, ShortName = ColumnConcatenatingTransformer.LoadName)]
        public static CommonOutputs.TransformOutput ConcatColumns(IHostEnvironment env, ColumnConcatenatingTransformer.Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("ConcatColumns");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var xf = ColumnConcatenatingTransformer.Create(env, input, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModelImpl(env, xf, input.Data), OutputData = xf };
        }

        [TlcModule.EntryPoint(Name = "Transforms.ColumnSelector", Desc = "Selects a set of columns, dropping all others", UserName = "Select Columns")]
        public static CommonOutputs.TransformOutput SelectColumns(IHostEnvironment env, ColumnSelectingTransformer.Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("SelectColumns");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var xf = new ColumnSelectingTransformer(env, input.KeepColumns, input.DropColumns, input.KeepHidden, input.IgnoreMissing).Transform(input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModelImpl(env, xf, input.Data), OutputData = xf };
        }

        [TlcModule.EntryPoint(Name = "Transforms.ColumnCopier", Desc = "Duplicates columns from the dataset", UserName = ColumnCopyingTransformer.UserName, ShortName = ColumnCopyingTransformer.ShortName)]
        public static CommonOutputs.TransformOutput CopyColumns(IHostEnvironment env, ColumnCopyingTransformer.Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("CopyColumns");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);
            var xf = ColumnCopyingTransformer.Create(env, input, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModelImpl(env, xf, input.Data), OutputData = xf };
        }
    }
}
