// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Core;

public class DynamicLoadingModule<T, T1, TResult> : torch.nn.Module<T1, TResult>, IDynamicLoadModule
    where T : nn.Module<T1, TResult>
    where T1 : Tensor
{
    private readonly T _model;

    public DynamicLoadingModule(T model)
        : base(model.GetName())
    {
        this._model = model;
        this.RegisterComponents();
    }

    public static DynamicLoadingModule<T, T1, TResult> CreateFromModel(T model)
    {
        return new DynamicLoadingModule<T, T1, TResult>(model);
    }

    public Action<nn.Module>? LoadToDeviceFunc { get; set; }
    public Action<nn.Module>? UnloadFromDeviceFunc { get; set; }

#pragma warning disable MSML_GeneralName // This name should be PascalCased
    public override TResult forward(T1 input)
#pragma warning restore MSML_GeneralName // This name should be PascalCased
    {
        if (LoadToDeviceFunc != null)
        {
            LoadToDeviceFunc(this);
        }

        var output = this._model.forward(input);

        if (UnloadFromDeviceFunc != null)
        {
            UnloadFromDeviceFunc(this);
        }

        return output;
    }
}
