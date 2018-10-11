// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

// REVIEW: This is a temporary hack code to allow loading old saved loader models. Delete it once it is no longer needed.

// The below signatures are for (de)serialization purposes only.
[assembly: LoadableClass(typeof(IDataTransform), typeof(CompositeTransform), null, typeof(SignatureLoadDataTransform),
    "Composite Transform", CompositeTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    public static class CompositeTransform
    {
        private const string RegistrationName = "CompositeTransform";

        public const string LoaderSignature = "CompositeRowFunction";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "CMPSTE F",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(CompositeTransform).Assembly.FullName);
        }

        public static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            Contracts.CheckValue(input, nameof(input));

            IDataTransform res = null;
            var h = env.Register(RegistrationName);
            using (var ch = h.Start("Loading Model"))
            {
                // *** Binary format ***
                // number of row functions
                // row functions (each in a separate folder)
                var numFunctions = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(numFunctions > 0);

                for (int i = 0; i < numFunctions; i++)
                {
                    var modelName = string.Format("Model_{0:000}", i);
                    ctx.LoadModel<IDataTransform, SignatureLoadDataTransform>(env, out res, modelName, input);
                    input = res;
                }

                ch.Done();
            }

            return res;
        }
    }
}
