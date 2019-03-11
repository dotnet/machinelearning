// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.EntryPoints;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: EntryPointModule(typeof(NAHandling))]

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// Entry point methods for NA handling.
    /// </summary>
    internal static class NAHandling
    {
        [TlcModule.EntryPoint(Name = "Transforms.MissingValuesDropper",
            Desc = MissingValueDroppingTransformer.Summary,
            UserName = MissingValueDroppingTransformer.FriendlyName,
            ShortName = MissingValueDroppingTransformer.ShortName)]
        public static CommonOutputs.TransformOutput Drop(IHostEnvironment env, MissingValueDroppingTransformer.Options input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, MissingValueDroppingTransformer.ShortName, input);
            var xf = MissingValueDroppingTransformer.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.MissingValuesRowDropper",
            Desc = NAFilter.Summary,
            UserName = NAFilter.FriendlyName,
            ShortName = NAFilter.ShortName)]
        public static CommonOutputs.TransformOutput Filter(IHostEnvironment env, NAFilter.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, NAFilter.ShortName, input);
            var xf = new NAFilter(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.MissingValueHandler",
            Desc = MissingValueHandlingTransformer.Summary,
            UserName = MissingValueHandlingTransformer.FriendlyName,
            ShortName = MissingValueHandlingTransformer.ShortName)]
        public static CommonOutputs.TransformOutput Handle(IHostEnvironment env, MissingValueHandlingTransformer.Options input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "NAHandle", input);
            var xf = MissingValueHandlingTransformer.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.MissingValueIndicator",
            Desc = MissingValueIndicatorTransformer.Summary,
            UserName = MissingValueIndicatorTransformer.FriendlyName,
            ShortName = MissingValueIndicatorTransformer.ShortName)]
        public static CommonOutputs.TransformOutput Indicator(IHostEnvironment env, MissingValueIndicatorTransformer.Options input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "NAIndicator", input);
            var xf = new MissingValueIndicatorTransformer(h, input).Transform(input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.MissingValueSubstitutor",
            Desc = MissingValueReplacingTransformer.Summary,
            UserName = MissingValueReplacingTransformer.FriendlyName,
            ShortName = MissingValueReplacingTransformer.ShortName)]
        public static CommonOutputs.TransformOutput Replace(IHostEnvironment env, MissingValueReplacingTransformer.Options input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "NAReplace", input);
            var xf = MissingValueReplacingTransformer.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }
    }
}
