// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

namespace Microsoft.ML.Runtime.Ensemble.OutputCombiners
{
    internal abstract class BaseScalarStacking : BaseStacking<Single>
    {
        internal BaseScalarStacking(IHostEnvironment env, string name, ArgumentsBase args)
            : base(env, name, args)
        {
        }

        internal BaseScalarStacking(IHostEnvironment env, string name, ModelLoadContext ctx)
            : base(env, name, ctx)
        {
        }

        protected override void FillFeatureBuffer(Single[] src, ref VBuffer<Single> dst)
        {
            Contracts.AssertNonEmpty(src);
            int len = src.Length;
            var editor = VBufferEditor.Create(ref dst, len);
            src.CopyTo(editor.Values);
            dst = editor.Commit();
        }
    }
}
