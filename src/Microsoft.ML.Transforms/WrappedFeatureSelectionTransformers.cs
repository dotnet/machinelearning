// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Data.StaticPipe.Runtime;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Transforms
{
    /// <include file='doc.xml' path='doc/members/member[@name="CountFeatureSelection"]' />
    public sealed class CountFeatureSelector : TrainedWrapperEstimatorBase
    {
        private readonly long _count;
        private readonly (string input, string output)[] _columns;

        /// <include file='doc.xml' path='doc/members/member[@name="CountFeatureSelection"]' />
        /// <param name="env">The environment.</param>
        /// <param name="inputColumn">The input column to apply feature selection on.</param>
        /// <param name="outputColumn">The output column. Null means <paramref name="inputColumn"/> is used.</param>
        /// <param name="count">If the count of non-default values for a slot is greater than or equal to this threshold, the slot is preserved.</param>
        public CountFeatureSelector(IHostEnvironment env, string inputColumn, string outputColumn = null, long count = 1)
            : this(env, new[] { (inputColumn, outputColumn ?? inputColumn) }, count)
        {
        }

        /// <include file='doc.xml' path='doc/members/member[@name="CountFeatureSelection"]' />
        /// <param name="env">The environment.</param>
        /// <param name="count">If the count of non-default values for a slot is greater than or equal to this threshold, the slot is preserved.</param>
        /// <param name="columns">Columns to use for feature selection.</param>
        public CountFeatureSelector(IHostEnvironment env, (string input, string output)[] columns, long count = 1)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(CountFeatureSelector)))
        {
            _count = count;
            _columns = columns;
        }

        public override TransformWrapper Fit(IDataView input)
        {
            var dataview = CopyColumnsTransform.Create(Host, new CopyColumnsTransform.Arguments()
            {
                Column = _columns.Select(x => new CopyColumnsTransform.Column { Source = x.input, Name = x.output }).ToArray(),
            }, input);

            var names = _columns.Select(x => x.output).ToArray();
            return new TransformWrapper(Host, CountFeatureSelectionTransform.Create(Host, dataview, _count, names));
        }
    }

    /// <include file='doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
    public sealed class MutualInformationFeatureSelector : TrainedWrapperEstimatorBase
    {
        private readonly (string input, string output)[] _columns;
        private readonly string _labelColumn;
        private readonly int _slotsInOutput;
        private readonly int _numBins;

        /// <include file='doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
        /// <param name="env">The environment.</param>
        /// <param name="inputColumn">The input column to apply feature selection on.</param>
        /// <param name="outputColumn">The output column. Null means <paramref name="inputColumn"/> is used.</param>
        /// <param name="labelColumn">Column to use for labels.</param>
        /// <param name="slotsInOutput">The maximum number of slots to preserve in output.</param>
        /// <param name="numBins">Max number of bins for R4/R8 columns, power of 2 recommended.</param>
        public MutualInformationFeatureSelector(IHostEnvironment env,
            string inputColumn,
            string outputColumn = null,
            string labelColumn = DefaultColumnNames.Label,
            int slotsInOutput = 1000,
            int numBins = 256)
            : this(env, new[] { (inputColumn, outputColumn ?? inputColumn) }, labelColumn, slotsInOutput, numBins)
        {
        }

        /// <include file='doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
        /// <param name="env">The environment.</param>
        /// <param name="labelColumn">Column to use for labels.</param>
        /// <param name="slotsInOutput">The maximum number of slots to preserve in output.</param>
        /// <param name="numBins">Max number of bins for R4/R8 columns, power of 2 recommended.</param>
        /// <param name="columns">Columns to use for feature selection.</param>
        public MutualInformationFeatureSelector(IHostEnvironment env,
            (string input, string output)[] columns,
            string labelColumn = DefaultColumnNames.Label,
            int slotsInOutput = 1000,
            int numBins = 256)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(CountFeatureSelector)))
        {
            _labelColumn = labelColumn;
            _slotsInOutput = slotsInOutput;
            _numBins = numBins;
            _columns = columns;
        }

        public override TransformWrapper Fit(IDataView input)
        {
            var dataview = CopyColumnsTransform.Create(Host, new CopyColumnsTransform.Arguments()
            {
                Column = _columns.Select(x => new CopyColumnsTransform.Column { Source = x.input, Name = x.output }).ToArray(),
            }, input);

            var names = _columns.Select(x => x.output).ToArray();
            return new TransformWrapper(Host, MutualInformationFeatureSelectionTransform.Create(Host, dataview, _labelColumn, _slotsInOutput, _numBins, names));
        }
    }

    /// <summary>
    /// Extensions for statically typed <see cref="CountFeatureSelector"/>.
    /// </summary>
    public static class CountFeatureSelectorExtensions
    {
        private sealed class OutPipelineColumn<T> : Scalar<T>
        {
            public readonly Scalar<T> Input;

            public OutPipelineColumn(Scalar<T> input, long count)
                : base(new Reconciler<T>(count), input)
            {
                Input = input;
            }
        }

        private sealed class OutPipelineColumnVec<T> : Vector<T>
        {
            public readonly Vector<T> Input;

            public OutPipelineColumnVec(Vector<T> input, long count)
                : base(new ReconcilerVec<T>(count), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler<T> : EstimatorReconciler
        {
            private readonly long _count;

            public Reconciler(long count)
            {
                _count = count;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);

                var pairs = new List<(string input, string output)>();
                foreach (var outCol in toOutput)
                    pairs.Add((inputNames[((OutPipelineColumn<T>)outCol).Input], outputNames[outCol]));

                return new CountFeatureSelector(env, pairs.ToArray(), _count);
            }
        }

        private sealed class ReconcilerVec<T> : EstimatorReconciler
        {
            private readonly long _count;

            public ReconcilerVec(long count)
            {
                _count = count;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);

                var pairs = new List<(string input, string output)>();
                foreach (var outCol in toOutput)
                    pairs.Add((inputNames[((OutPipelineColumnVec<T>)outCol).Input], outputNames[outCol]));

                return new CountFeatureSelector(env, pairs.ToArray(), _count);
            }
        }

        /// <include file='doc.xml' path='doc/members/member[@name="CountFeatureSelection"]' />
        /// <param name="input">The column to apply to.</param>
        /// <param name="count">If the count of non-default values for a slot is greater than or equal to this threshold, the slot is preserved.</param>
        public static Vector<float> SelectFeaturesBasedOnCount(this Vector<float> input, long count = 1) => new OutPipelineColumnVec<float>(input, count);

        /// <include file='doc.xml' path='doc/members/member[@name="CountFeatureSelection"]' />
        /// <param name="input">The column to apply to.</param>
        /// <param name="count">If the count of non-default values for a slot is greater than or equal to this threshold, the slot is preserved.</param>
        public static Vector<double> SelectFeaturesBasedOnCount(this Vector<double> input, long count = 1) => new OutPipelineColumnVec<double>(input, count);

        /// <include file='doc.xml' path='doc/members/member[@name="CountFeatureSelection"]' />
        /// <param name="input">The column to apply to.</param>
        /// <param name="count">If the count of non-default values for a slot is greater than or equal to this threshold, the slot is preserved.</param>
        public static Vector<string> SelectFeaturesBasedOnCount(this Vector<string> input, long count = 1) => new OutPipelineColumnVec<string>(input, count);

        /// <include file='doc.xml' path='doc/members/member[@name="CountFeatureSelection"]' />
        /// <param name="input">The column to apply to.</param>
        /// <param name="count">If the count of non-default values for a slot is greater than or equal to this threshold, the slot is preserved.</param>
        public static Scalar<float> SelectFeaturesBasedOnCount(this Scalar<float> input, long count = 1) => new OutPipelineColumn<float>(input, count);

        /// <include file='doc.xml' path='doc/members/member[@name="CountFeatureSelection"]' />
        /// <param name="input">The column to apply to.</param>
        /// <param name="count">If the count of non-default values for a slot is greater than or equal to this threshold, the slot is preserved.</param>
        public static Scalar<double> SelectFeaturesBasedOnCount(this Scalar<double> input, long count = 1) => new OutPipelineColumn<double>(input, count);

        /// <include file='doc.xml' path='doc/members/member[@name="CountFeatureSelection"]' />
        /// <param name="input">The column to apply to.</param>
        /// <param name="count">If the count of non-default values for a slot is greater than or equal to this threshold, the slot is preserved.</param>
        public static Scalar<string> SelectFeaturesBasedOnCount(this Scalar<string> input, long count = 1) => new OutPipelineColumn<string>(input, count);
    }

    /// <summary>
    /// Extensions for statically typed <see cref="MutualInformationFeatureSelector"/>.
    /// </summary>
    public static class MutualInformationFeatureSelectorExtensions
    {
        private sealed class OutPipelineColumnVec<T> : Vector<T>
        {
            public readonly Vector<T> Input;

            public OutPipelineColumnVec(Vector<T> input, string labelColumn, int slotsInOutput, int numBins)
                : base(new ReconcilerVec<T>(labelColumn, slotsInOutput, numBins), input)
            {
                Input = input;
            }
        }

        private sealed class OutPipelineColumn<T> : Scalar<T>
        {
            public readonly Scalar<T> Input;

            public OutPipelineColumn(Scalar<T> input, string labelColumn, int slotsInOutput, int numBins)
                : base(new Reconciler<T>(labelColumn, slotsInOutput, numBins), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler<T> : EstimatorReconciler
        {
            private readonly string _labelColumn;
            private readonly int _slotsInOutput;
            private readonly int _numBins;

            public Reconciler(string labelColumn, int slotsInOutput, int numBins)
            {
                _labelColumn = labelColumn;
                _slotsInOutput = slotsInOutput;
                _numBins = numBins;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);

                var pairs = new List<(string input, string output)>();
                foreach (var outCol in toOutput)
                    pairs.Add((inputNames[((OutPipelineColumn<T>)outCol).Input], outputNames[outCol]));

                return new MutualInformationFeatureSelector(env, pairs.ToArray(), _labelColumn, _slotsInOutput, _numBins);
            }
        }

        private sealed class ReconcilerVec<T> : EstimatorReconciler
        {
            private readonly string _labelColumn;
            private readonly int _slotsInOutput;
            private readonly int _numBins;

            public ReconcilerVec(string labelColumn, int slotsInOutput, int numBins)
            {
                _labelColumn = labelColumn;
                _slotsInOutput = slotsInOutput;
                _numBins = numBins;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);

                var pairs = new List<(string input, string output)>();
                foreach (var outCol in toOutput)
                    pairs.Add((inputNames[((OutPipelineColumnVec<T>)outCol).Input], outputNames[outCol]));

                return new MutualInformationFeatureSelector(env, pairs.ToArray(), _labelColumn, _slotsInOutput, _numBins);
            }
        }

        /// <include file='doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
        /// <param name="input">The column to apply to.</param>
        /// <param name="labelColumn">Column to use for labels.</param>
        /// <param name="slotsInOutput">The maximum number of slots to preserve in output.</param>
        /// <param name="numBins">Max number of bins for R4/R8 columns, power of 2 recommended.</param>
        public static Vector<float> SelectFeaturesBasedOnMutualInformation(
            this Vector<float> input,
            string labelColumn = DefaultColumnNames.Label,
            int slotsInOutput = 1000,
            int numBins = 256) => new OutPipelineColumnVec<float>(input, labelColumn, slotsInOutput, numBins);

        /// <include file='doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
        /// <param name="input">The column to apply to.</param>
        /// <param name="labelColumn">Column to use for labels.</param>
        /// <param name="slotsInOutput">The maximum number of slots to preserve in output.</param>
        /// <param name="numBins">Max number of bins for R4/R8 columns, power of 2 recommended.</param>
        public static Vector<double> SelectFeaturesBasedOnMutualInformation(
            this Vector<double> input,
            string labelColumn = DefaultColumnNames.Label,
            int slotsInOutput = 1000,
            int numBins = 256) => new OutPipelineColumnVec<double>(input, labelColumn, slotsInOutput, numBins);

        /// <include file='doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
        /// <param name="input">The column to apply to.</param>
        /// <param name="labelColumn">Column to use for labels.</param>
        /// <param name="slotsInOutput">The maximum number of slots to preserve in output.</param>
        /// <param name="numBins">Max number of bins for R4/R8 columns, power of 2 recommended.</param>
        public static Vector<bool> SelectFeaturesBasedOnMutualInformation(
            this Vector<bool> input,
            string labelColumn = DefaultColumnNames.Label,
            int slotsInOutput = 1000,
            int numBins = 256) => new OutPipelineColumnVec<bool>(input, labelColumn, slotsInOutput, numBins);

        /// <include file='doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
        /// <param name="input">The column to apply to.</param>
        /// <param name="labelColumn">Column to use for labels.</param>
        /// <param name="slotsInOutput">The maximum number of slots to preserve in output.</param>
        /// <param name="numBins">Max number of bins for R4/R8 columns, power of 2 recommended.</param>
        public static Scalar<float> SelectFeaturesBasedOnMutualInformation(
            this Scalar<float> input,
             string labelColumn = DefaultColumnNames.Label,
            int slotsInOutput = 1000,
            int numBins = 256) => new OutPipelineColumn<float>(input, labelColumn, slotsInOutput, numBins);

        /// <include file='doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
        /// <param name="input">The column to apply to.</param>
        /// <param name="labelColumn">Column to use for labels.</param>
        /// <param name="slotsInOutput">The maximum number of slots to preserve in output.</param>
        /// <param name="numBins">Max number of bins for R4/R8 columns, power of 2 recommended.</param>
        public static Scalar<double> SelectFeaturesBasedOnMutualInformation(
            this Scalar<double> input,
             string labelColumn = DefaultColumnNames.Label,
            int slotsInOutput = 1000,
            int numBins = 256) => new OutPipelineColumn<double>(input, labelColumn, slotsInOutput, numBins);

        /// <include file='doc.xml' path='doc/members/member[@name="MutualInformationFeatureSelection"]/*' />
        /// <param name="input">The column to apply to.</param>
        /// <param name="labelColumn">Column to use for labels.</param>
        /// <param name="slotsInOutput">The maximum number of slots to preserve in output.</param>
        /// <param name="numBins">Max number of bins for R4/R8 columns, power of 2 recommended.</param>
        public static Scalar<bool> SelectFeaturesBasedOnMutualInformation(
            this Scalar<bool> input,
             string labelColumn = DefaultColumnNames.Label,
            int slotsInOutput = 1000,
            int numBins = 256) => new OutPipelineColumn<bool>(input, labelColumn, slotsInOutput, numBins);
    }
}
