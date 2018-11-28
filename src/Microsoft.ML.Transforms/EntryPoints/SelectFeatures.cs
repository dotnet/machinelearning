// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.FeatureSelection;

[assembly: LoadableClass(typeof(void), typeof(SelectFeatures), null, typeof(SignatureEntryPointModule), "SelectFeatures")]

namespace Microsoft.ML.Transforms
{
    public static class SelectFeatures
    {
        [TlcModule.EntryPoint(Name = "Transforms.FeatureSelectorByCount",
            Desc = CountFeatureSelectingEstimator.Summary,
            UserName = CountFeatureSelectingEstimator.UserName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/member[@name=""CountFeatureSelection""]/*'/>",
                                 @"<include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/example[@name=""CountFeatureSelection""]/*'/>"})]
        public static CommonOutputs.TransformOutput CountSelect(IHostEnvironment env, CountFeatureSelectingEstimator.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("CountSelect");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var xf = CountFeatureSelectingEstimator.Create(host, input, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModel(env, xf, input.Data), OutputData = xf };
        }

        [TlcModule.EntryPoint(Name = "Transforms.FeatureSelectorByMutualInformation",
            Desc = MutualInformationFeatureSelectingEstimator.Summary,
            UserName = MutualInformationFeatureSelectingEstimator.UserName,
            ShortName = MutualInformationFeatureSelectingEstimator.ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/member[@name=""MutualInformationFeatureSelection""]/*'/>",
                                 @"<include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/example[@name=""MutualInformationFeatureSelection""]/*'/>"})]
        public static CommonOutputs.TransformOutput MutualInformationSelect(IHostEnvironment env, MutualInformationFeatureSelectingEstimator.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("MutualInformationSelect");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var xf = MutualInformationFeatureSelectingEstimator.Create(host, input, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModel(env, xf, input.Data), OutputData = xf };
        }
    }
}
