// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;

[assembly: LoadableClass(typeof(void), typeof(SelectFeatures), null, typeof(SignatureEntryPointModule), "SelectFeatures")]
namespace Microsoft.ML.Runtime.EntryPoints
{
    public static class SelectFeatures
    {
        [TlcModule.EntryPoint(Name = "Transforms.FeatureSelectorByCount", 
            Desc = CountFeatureSelectionTransform.Summary, 
            UserName = CountFeatureSelectionTransform.UserName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/member[@name=""CountFeatureSelection""]'/>" })]
        public static CommonOutputs.TransformOutput CountSelect(IHostEnvironment env, CountFeatureSelectionTransform.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("CountSelect");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var xf = CountFeatureSelectionTransform.Create(host, input, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModel(env, xf, input.Data), OutputData = xf };
        }

        [TlcModule.EntryPoint(Name = "Transforms.FeatureSelectorByMutualInformation", 
            Desc = MutualInformationFeatureSelectionTransform.Summary, 
            UserName = MutualInformationFeatureSelectionTransform.UserName, 
            ShortName = MutualInformationFeatureSelectionTransform.ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/member[@name=""MutualInformationFeatureSelection""]'/>" })]
        public static CommonOutputs.TransformOutput MutualInformationSelect(IHostEnvironment env, MutualInformationFeatureSelectionTransform.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("MutualInformationSelect");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var xf = MutualInformationFeatureSelectionTransform.Create(host, input, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModel(env, xf, input.Data), OutputData = xf };
        }
    }
}
