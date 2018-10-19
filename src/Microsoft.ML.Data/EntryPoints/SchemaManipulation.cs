// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(typeof(void), typeof(SchemaManipulation), null, typeof(SignatureEntryPointModule), "SchemaManipulation")]

namespace Microsoft.ML.Runtime.EntryPoints
{
    public static class SchemaManipulation
    {
        [TlcModule.EntryPoint(Name = "Transforms.ColumnConcatenator", Desc = ConcatTransform.Summary, UserName = ConcatTransform.UserName, ShortName = ConcatTransform.LoadName)]
        public static CommonOutputs.TransformOutput ConcatColumns(IHostEnvironment env, ConcatTransform.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("ConcatColumns");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var xf = ConcatTransform.Create(env, input, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModel(env, xf, input.Data), OutputData = xf };
        }

        [TlcModule.EntryPoint(Name = "Transforms.ColumnSelector", Desc = "Selects a set of columns, dropping all others", UserName = "Select Columns")]
        public static CommonOutputs.TransformOutput SelectColumns(IHostEnvironment env, SelectColumnsTransform.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("SelectColumns");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var xf = new SelectColumnsTransform(env, input.KeepColumns, input.DropColumns, input.KeepHidden, input.IgnoreMissing).Transform(input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModel(env, xf, input.Data), OutputData = xf };
        }

        [TlcModule.EntryPoint(Name = "Transforms.ColumnCopier", Desc = "Duplicates columns from the dataset", UserName = CopyColumnsTransform.UserName, ShortName = CopyColumnsTransform.ShortName)]
        public static CommonOutputs.TransformOutput CopyColumns(IHostEnvironment env, CopyColumnsTransform.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("CopyColumns");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);
            var xf = CopyColumnsTransform.Create(env, input, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModel(env, xf, input.Data), OutputData = xf };
        }
    }
}
