// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Transforms;

[assembly: EntryPointModule(typeof(SelectRows))]

namespace Microsoft.ML.Runtime.EntryPoints
{
    public static class SelectRows
    {
        [TlcModule.EntryPoint(Name = "Transforms.RowRangeFilter", Desc = RangeFilter.Summary, UserName = RangeFilter.UserName, ShortName = RangeFilter.LoaderSignature)]
        public static CommonOutputs.TransformOutput FilterByRange(IHostEnvironment env, RangeFilter.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RangeFilter.LoaderSignature);
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var xf = new RangeFilter(host, input, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModel(env, xf, input.Data), OutputData = xf };
        }

        [TlcModule.EntryPoint(Name = "Transforms.RowSkipFilter", Desc = SkipTakeFilter.SkipFilterSummary, UserName = SkipTakeFilter.SkipFilterUserName,
            ShortName = SkipTakeFilter.SkipFilterShortName)]
        public static CommonOutputs.TransformOutput SkipFilter(IHostEnvironment env, SkipTakeFilter.SkipArguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("SkipFilter");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);
            var xf = SkipTakeFilter.Create(host, input, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModel(env, xf, input.Data), OutputData = xf };
        }

        [TlcModule.EntryPoint(Name = "Transforms.RowTakeFilter", Desc = SkipTakeFilter.TakeFilterSummary, UserName = SkipTakeFilter.TakeFilterUserName,
            ShortName = SkipTakeFilter.TakeFilterShortName)]
        public static CommonOutputs.TransformOutput TakeFilter(IHostEnvironment env, SkipTakeFilter.TakeArguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TakeFilter");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);
            var xf = SkipTakeFilter.Create(host, input, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModel(env, xf, input.Data), OutputData = xf };
        }

        [TlcModule.EntryPoint(Name = "Transforms.RowSkipAndTakeFilter", Desc = SkipTakeFilter.SkipTakeFilterSummary,
            UserName = SkipTakeFilter.SkipTakeFilterUserName, ShortName = SkipTakeFilter.SkipTakeFilterShortName)]
        public static CommonOutputs.TransformOutput SkipAndTakeFilter(IHostEnvironment env, SkipTakeFilter.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("SkipTakeFilter");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);
            var xf = SkipTakeFilter.Create(host, input, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModel(env, xf, input.Data), OutputData = xf };
        }
    }
}
