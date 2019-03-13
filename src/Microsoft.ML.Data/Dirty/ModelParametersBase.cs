// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Trainers
{
    /// <summary>
    /// A base class for predictors producing <typeparamref name="TOutput"/>.
    /// Note: This provides essentially no value going forward. New predictors should just
    /// derive from the interfaces they need.
    /// </summary>
    public abstract class ModelParametersBase<TOutput> : ICanSaveModel, IPredictorProducing<TOutput>
    {
        private const string NormalizerWarningFormat =
            "Ignoring integrated normalizer while loading a predictor of type {0}.{1}" +
            "   Please refer to https://aka.ms/MLNetIssue for assistance with converting legacy models.";

        [BestFriend]
        private protected readonly IHost Host;

        [BestFriend]
        private protected ModelParametersBase(IHostEnvironment env, string name)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckNonWhiteSpace(name, nameof(name));
            Host = env.Register(name);
        }

        [BestFriend]
        private protected ModelParametersBase(IHostEnvironment env, string name, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckNonWhiteSpace(name, nameof(name));
            Host = env.Register(name);

            // *** Binary format ***
            // int: sizeof(Float)

            // Verify that the Float type matches.
            int cbFloat = ctx.Reader.ReadInt32();
#pragma warning disable MSML_NoMessagesForLoadContext // This one is actually useful.
            Host.CheckDecode(cbFloat == sizeof(float), "This file was saved by an incompatible version");
#pragma warning restore MSML_NoMessagesForLoadContext
        }

        void ICanSaveModel.Save(ModelSaveContext ctx) => Save(ctx);

        [BestFriend]
        private protected virtual void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            SaveCore(ctx);
        }

        [BestFriend]
        private protected virtual void SaveCore(ModelSaveContext ctx)
        {
            Contracts.AssertValue(ctx);

            // *** Binary format ***
            // int: sizeof(Float)
            // <Derived type stuff>
            ctx.Writer.Write(sizeof(float));
        }

        PredictionKind IPredictor.PredictionKind => PredictionKind;

        [BestFriend]
        private protected abstract PredictionKind PredictionKind { get; }

        /// <summary>
        /// This emits a warning if there is Normalizer sub-model.
        /// </summary>
        [BestFriend]
        private protected static bool WarnOnOldNormalizer(ModelLoadContext ctx, Type typePredictor, IChannelProvider provider)
        {
            Contracts.CheckValue(provider, nameof(provider));
            provider.CheckValue(ctx, nameof(ctx));
            provider.CheckValue(typePredictor, nameof(typePredictor));

            if (!ctx.ContainsModel(@"Normalizer"))
                return false;
            using (var ch = provider.Start("WarnNormalizer"))
            {
                ch.Warning(NormalizerWarningFormat, typePredictor, Environment.NewLine);
            }
            return true;
        }
    }
}
