// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;
using Microsoft.ML.AutoML.CodeGen;
using Microsoft.ML.SearchSpace;

namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// Estimator with search space.
    /// </summary>
    [JsonConverter(typeof(SweepableEstimatorConverter))]
    public class SweepableEstimator : Estimator, ISweepable<IEstimator<ITransformer>>
    {
        private readonly Func<MLContext, Parameter, IEstimator<ITransformer>> _factory;

        public SweepableEstimator(Func<MLContext, Parameter, IEstimator<ITransformer>> factory, SearchSpace.SearchSpace ss)
            : this()
        {
            _factory = factory;
            SearchSpace = ss;
            Parameter = ss.SampleFromFeatureSpace(ss.Default);
        }

        protected SweepableEstimator()
            : base()
        {
        }

        /// <summary>
        /// for test purpose only
        /// </summary>
        /// <param name="estimatorType"></param>
        internal SweepableEstimator(EstimatorType estimatorType)
            : base(estimatorType)
        {
        }

        public virtual IEstimator<ITransformer> BuildFromOption(MLContext context, Parameter param)
        {
            return _factory(context, param);
        }

        public SearchSpace.SearchSpace SearchSpace { get; set; }

        internal virtual IEnumerable<string> CSharpUsingStatements { get; }

        internal virtual IEnumerable<string> NugetDependencies { get; }

        internal virtual string FunctionName { get; }
    }

    internal abstract class SweepableEstimator<TOption> : SweepableEstimator
        where TOption : class, new()
    {
        public abstract IEstimator<ITransformer> BuildFromOption(MLContext context, TOption param);

        /// <summary>
        /// Parameter with type.
        /// </summary>
        public TOption TParameter
        {
            get => base.Parameter.AsType<TOption>();
            set => base.Parameter = Parameter.FromObject(value);
        }

        public override IEstimator<ITransformer> BuildFromOption(MLContext context, Parameter param)
        {
            this.Parameter = param;
            return BuildFromOption(context, param.AsType<TOption>());
        }
    }
}
