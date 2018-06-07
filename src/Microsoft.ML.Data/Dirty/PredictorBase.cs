// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Model;

namespace Microsoft.ML.Runtime.Internal.Internallearn
{
    /// <summary>
    /// A base class for predictors producing <typeparamref name="TOutput"/>.
    /// Note: This provides essentially no value going forward. New predictors should just
    /// derive from the interfaces they need.
    /// </summary>
    public abstract class PredictorBase<TOutput> : IPredictorProducing<TOutput>
    {
        public const string NormalizerWarningFormat =
            "Ignoring integrated normalizer while loading a predictor of type {0}.{1}" +
            "   Please contact https://aka.ms/MLNetIssue for assistance with converting legacy models.";

        protected readonly IHost Host;

        protected PredictorBase(IHostEnvironment env, string name)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckNonWhiteSpace(name, nameof(name));
            Host = env.Register(name);
        }

        protected PredictorBase(IHostEnvironment env, string name, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckNonWhiteSpace(name, nameof(name));
            Host = env.Register(name);

            // *** Binary format ***
            // int: sizeof(Float)

            // Verify that the Float type matches.
            int cbFloat = ctx.Reader.ReadInt32();
#pragma warning disable TLC_NoMessagesForLoadContext // This one is actually useful.
            Host.CheckDecode(cbFloat == sizeof(Float), "This file was saved by an incompatible version");
#pragma warning restore TLC_NoMessagesForLoadContext
        }

        public virtual void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            SaveCore(ctx);
        }

        protected virtual void SaveCore(ModelSaveContext ctx)
        {
            Contracts.AssertValue(ctx);

            // *** Binary format ***
            // int: sizeof(Float)
            // <Derived type stuff>
            ctx.Writer.Write(sizeof(Float));
        }

        public abstract PredictionKind PredictionKind { get; }

        /// <summary>
        /// This emits a warning if there is Normalizer sub-model.
        /// </summary>
        public static bool WarnOnOldNormalizer(ModelLoadContext ctx, Type typePredictor, IChannelProvider provider)
        {
            Contracts.CheckValue(provider, nameof(provider));
            provider.CheckValue(ctx, nameof(ctx));
            provider.CheckValue(typePredictor, nameof(typePredictor));

            if (!ctx.ContainsModel(@"Normalizer"))
                return false;
            using (var ch = provider.Start("WarnNormalizer"))
            {
                ch.Warning(NormalizerWarningFormat, typePredictor, Environment.NewLine);
                ch.Done();
            }
            return true;
        }
    }
}
