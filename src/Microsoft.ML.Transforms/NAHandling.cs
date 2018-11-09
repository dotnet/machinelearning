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
            Desc = NADropTransform.Summary,
            UserName = NADropTransform.FriendlyName,
            ShortName = NADropTransform.ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/member[@name=""NADrop""]/*' />",
                                 @"<include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/example[@name=""NADrop""]/*' />" })]
        public static CommonOutputs.TransformOutput Drop(IHostEnvironment env, NADropTransform.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, NADropTransform.ShortName, input);
            var xf = new NADropTransform(h, input, input.Data);
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
            Desc = NAHandleTransform.Summary,
            UserName = NAHandleTransform.FriendlyName,
            ShortName = NAHandleTransform.ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Data/Transforms/doc.xml' path='doc/members/member[@name=""NAHandle""]/*' />",
                                 @"<include file='../Microsoft.ML.Data/Transforms/doc.xml' path='doc/members/example[@name=""NAHandle""]/*' />" })]
        public static CommonOutputs.TransformOutput Handle(IHostEnvironment env, NAHandleTransform.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "NAHandle", input);
            var xf = NAHandleTransform.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.MissingValueIndicator",
            Desc = MissingValueIndicatorTransformer.Summary,
            UserName = MissingValueIndicatorTransformer.FriendlyName,
            ShortName = MissingValueIndicatorTransformer.ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/member[@name=""NAIndicator""]/*' />",
                                 @"<include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/example[@name=""NAIndicator""]/*' />"})]
        public static CommonOutputs.TransformOutput Indicator(IHostEnvironment env, MissingValueIndicatorTransformer.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "NAIndicator", input);
            var xf = new MissingValueIndicatorTransformer(h, input).Transform(input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.MissingValueSubstitutor",
            Desc = NAReplaceTransform.Summary,
            UserName = NAReplaceTransform.FriendlyName,
            ShortName = NAReplaceTransform.ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/member[@name=""NAReplace""]/*' />",
                                 @"<include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/example[@name=""NAReplace""]/*' />"})]
        public static CommonOutputs.TransformOutput Replace(IHostEnvironment env, NAReplaceTransform.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "NAReplace", input);
            var xf = NAReplaceTransform.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, xf, input.Data),
                OutputData = xf
            };
        }
    }
}
