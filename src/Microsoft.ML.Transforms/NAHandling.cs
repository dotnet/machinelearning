// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Transforms;

[assembly: EntryPointModule(typeof(NAHandling))]

namespace Microsoft.ML.Transforms
{
    public static class NAHandling
    {
        [TlcModule.EntryPoint(Name = "Transforms.MissingValuesDropper",
            Desc = MissingValueDroppingTransformer.Summary,
            UserName = MissingValueDroppingTransformer.FriendlyName,
            ShortName = MissingValueDroppingTransformer.ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/member[@name=""NADrop""]/*' />",
                                 @"<include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/example[@name=""NADrop""]/*' />" })]
        public static CommonOutputs.TransformOutput Drop(IHostEnvironment env, MissingValueDroppingTransformer.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, MissingValueDroppingTransformer.ShortName, input);
            var xf = new MissingValueDroppingTransformer(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.MissingValuesRowDropper",
            Desc = NAFilter.Summary,
            UserName = NAFilter.FriendlyName,
            ShortName = NAFilter.ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Data/Transforms/doc.xml' path='doc/members/member[@name=""NAFilter""]/*' />",
                                 @"<include file='../Microsoft.ML.Data/Transforms/doc.xml' path='doc/members/example[@name=""NAFilter""]/*' />"})]
        public static CommonOutputs.TransformOutput Filter(IHostEnvironment env, NAFilter.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, NAFilter.ShortName, input);
            var xf = new NAFilter(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.MissingValueHandler",
            Desc = MissingValueHandlingTransformer.Summary,
            UserName = MissingValueHandlingTransformer.FriendlyName,
            ShortName = MissingValueHandlingTransformer.ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Data/Transforms/doc.xml' path='doc/members/member[@name=""NAHandle""]/*' />",
                                 @"<include file='../Microsoft.ML.Data/Transforms/doc.xml' path='doc/members/example[@name=""NAHandle""]/*' />" })]
        public static CommonOutputs.TransformOutput Handle(IHostEnvironment env, MissingValueHandlingTransformer.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "NAHandle", input);
            var xf = MissingValueHandlingTransformer.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.MissingValueIndicator",
            Desc = NAIndicatorTransform.Summary,
            UserName = NAIndicatorTransform.FriendlyName,
            ShortName = NAIndicatorTransform.ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/member[@name=""NAIndicator""]/*' />",
                                 @"<include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/example[@name=""NAIndicator""]/*' />"})]
        public static CommonOutputs.TransformOutput Indicator(IHostEnvironment env, NAIndicatorTransform.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "NAIndicator", input);
            var xf = new NAIndicatorTransform(h, input).Transform(input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.MissingValueSubstitutor",
            Desc = MissingValueReplacingTransformer.Summary,
            UserName = MissingValueReplacingTransformer.FriendlyName,
            ShortName = MissingValueReplacingTransformer.ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/member[@name=""NAReplace""]/*' />",
                                 @"<include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/example[@name=""NAReplace""]/*' />"})]
        public static CommonOutputs.TransformOutput Replace(IHostEnvironment env, MissingValueReplacingTransformer.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "NAReplace", input);
            var xf = MissingValueReplacingTransformer.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, xf, input.Data),
                OutputData = xf
            };
        }
    }
}
