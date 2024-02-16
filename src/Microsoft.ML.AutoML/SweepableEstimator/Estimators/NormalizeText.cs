// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.AutoML.CodeGen
{
    internal partial class NormalizeText
    {
        public override IEstimator<ITransformer> BuildFromOption(MLContext context, NormalizeTextOption param)
        {
            return context.Transforms.Text.NormalizeText(param.OutputColumnName, param.InputColumnName, param.CaseMode, param.KeepDiacritics, param.KeepPunctuations, param.KeepNumbers);
        }
    }
}
