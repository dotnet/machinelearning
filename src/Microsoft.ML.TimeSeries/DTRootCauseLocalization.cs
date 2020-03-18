// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.TimeSeries;
using Microsoft.ML.Transforms.TimeSeries;

[assembly: LoadableClass(DTRootCauseLocalizationTransformer.Summary, typeof(IDataTransform), typeof(DTRootCauseLocalizationTransformer), typeof(DTRootCauseLocalizationTransformer.Options), typeof(SignatureDataTransform),
    DTRootCauseLocalizationTransformer.UserName, "DTRootCauseLocalizationTransform", "DTRootCauseLocalization")]

[assembly: LoadableClass(DTRootCauseLocalizationTransformer.Summary, typeof(IDataTransform), typeof(DTRootCauseLocalizationTransformer), null, typeof(SignatureLoadDataTransform),
    DTRootCauseLocalizationTransformer.UserName, DTRootCauseLocalizationTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(DTRootCauseLocalizationTransformer), null, typeof(SignatureLoadModel),
    DTRootCauseLocalizationTransformer.UserName, DTRootCauseLocalizationTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(DTRootCauseLocalizationTransformer), null, typeof(SignatureLoadRowMapper),
    DTRootCauseLocalizationTransformer.UserName, DTRootCauseLocalizationTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms.TimeSeries
{
    public sealed class RootCauseLocalizationInputTypeAttribute : DataViewTypeAttribute
    {
        /// <summary>
        /// Create a root cause localizagin input type.
        /// </summary>
        public RootCauseLocalizationInputTypeAttribute()
        {
        }

        /// <summary>
        /// Equal function.
        /// </summary>
        public override bool Equals(DataViewTypeAttribute other)
        {
            if (!(other is RootCauseLocalizationInputTypeAttribute otherAttribute))
                return false;
            return true;
        }

        /// <summary>
        /// Produce the same hash code for all RootCauseLocalizationInputTypeAttribute.
        /// </summary>
        public override int GetHashCode()
        {
            return 0;
        }

        public override void Register()
        {
            DataViewTypeManager.Register(new RootCauseLocalizationInputDataViewType(), typeof(RootCauseLocalizationInput), this);
        }
    }

    public sealed class RootCauseTypeAttribute : DataViewTypeAttribute
    {
        /// <summary>
        /// Create an root cause type.
        /// </summary>
        public RootCauseTypeAttribute()
        {
        }

        /// <summary>
        /// RootCauseTypeAttribute with the same type should equal.
        /// </summary>
        public override bool Equals(DataViewTypeAttribute other)
        {
            if (other is RootCauseTypeAttribute otherAttribute)
                return true;
            return false;
        }

        /// <summary>
        /// Produce the same hash code for all RootCauseTypeAttribute.
        /// </summary>
        public override int GetHashCode()
        {
            return 0;
        }

        public override void Register()
        {
            DataViewTypeManager.Register(new RootCauseDataViewType(), typeof(RootCause), this);
        }
    }

    public sealed class RootCause
    {
        public List<RootCauseItem> Items { get; set; }
    }

    public sealed class RootCauseLocalizationInput
    {
        public DateTime AnomalyTimestamp { get; set; }

        public Dictionary<string, string> AnomalyDimensions { get; set; }

        public List<MetricSlice> Slices { get; set; }

        public DTRootCauseLocalizationEstimator.AggregateType  AggType{ get; set; }

        public string AggSymbol { get; set; }

        public RootCauseLocalizationInput(DateTime anomalyTimestamp, Dictionary<string, string> anomalyDimensions, List<MetricSlice> slices, DTRootCauseLocalizationEstimator.AggregateType aggregateType, string aggregateSymbol) {
            AnomalyTimestamp = anomalyTimestamp;
            AnomalyDimensions = anomalyDimensions;
            Slices = slices;
            AggType = aggregateType;
            AggSymbol = aggregateSymbol;
        }
        public void Dispose()
        {
            AnomalyDimensions = null;
            Slices = null;
        }
    }

    public sealed class MetricSlice
    {
        public DateTime TimeStamp { get; set; }
        public List<Point> Points { get; set; }

        public MetricSlice(DateTime timeStamp, List<Point> points) {
            TimeStamp = timeStamp;
            Points = points;
        }
    }

    public sealed class RootCauseDataViewType : StructuredDataViewType
    {
        public RootCauseDataViewType()
           : base(typeof(RootCause))
        {
        }

        public override bool Equals(DataViewType other)
        {
            if (other == this)
                return true;
            if (!(other is RootCauseDataViewType tmp))
                return false;
            return true;
        }

        public override int GetHashCode()
        {
            return 0;
        }

        public override string ToString()
        {
            return typeof(RootCauseDataViewType).Name;
        }
    }

    public sealed class RootCauseLocalizationInputDataViewType : StructuredDataViewType
    {
        public RootCauseLocalizationInputDataViewType()
           : base(typeof(RootCauseLocalizationInput))
        {
        }

        public override bool Equals(DataViewType other)
        {
            if (!(other is RootCauseLocalizationInputDataViewType tmp))
                return false;
            return true;
        }

        public override int GetHashCode()
        {
            return 0;
        }

        public override string ToString()
        {
            return typeof(RootCauseLocalizationInputDataViewType).Name;
        }
    }

    // REVIEW: Rewrite as LambdaTransform to simplify.
    // REVIEW: Should it be separate transform or part of ImageResizerTransform?
    /// <summary>
    /// <see cref="ITransformer"/> resulting from fitting an <see cref="DTRootCauseLocalizationTransformer"/>.
    /// </summary>
    public sealed class DTRootCauseLocalizationTransformer : OneToOneTransformerBase
    {
        internal sealed class Column : OneToOneColumn
        {
            internal static Column Parse(string str)
            {
                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            internal bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                return TryUnparseCore(sb);
            }
        }

        internal class Options : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Weight for getting the score for the root cause item.", ShortName = "Beta", SortOrder = 2)]
            public double Beta = DTRootCauseLocalizationEstimator.Defaults.Beta;

        }

        internal const string Summary = "Localize root cause for anomaly.";

        internal const string UserName = "DT Root Cause Localization Transform";
        internal const string LoaderSignature = "DTRootCauseLTransform";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "DTRCL",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(DTRootCauseLocalizationTransformer).Assembly.FullName);
        }

        private const string RegistrationName = "RootCauseLocalization";

        /// <summary>
        /// The input and output column pairs passed to this <see cref="ITransformer"/>.
        /// </summary>
        internal IReadOnlyCollection<(string outputColumnName, string inputColumnName)> Columns => ColumnPairs.AsReadOnly();

        private readonly double _beta;

        /// <summary>
        /// Localization root cause for multi-dimensional anomaly.
        /// </summary>
        /// <param name="env">The estimator's local <see cref="IHostEnvironment"/>.</param>
        /// <param name="beta">Weight for generating score.</param>
        /// <param name="columns">The name of the columns (first item of the tuple), and the name of the resulting output column (second item of the tuple).</param>

        internal DTRootCauseLocalizationTransformer(IHostEnvironment env,double beta = DTRootCauseLocalizationEstimator.Defaults.Beta, params (string outputColumnName, string inputColumnName)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), columns)
        {
            Host.CheckUserArg(beta >=0 && beta <= 1, nameof(Options.Beta), "Must be in [0,1]");

            _beta = beta;
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));
            env.CheckValue(options.Columns, nameof(options.Columns));

            return new DTRootCauseLocalizationTransformer(env,options.Beta, options.Columns.Select(x => (x.Name, x.Source ?? x.Name)).ToArray())
                .MakeDataTransform(input);
        }

        // Factory method for SignatureLoadModel.
        private static DTRootCauseLocalizationTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);
            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new DTRootCauseLocalizationTransformer(host, ctx);
        }

        private DTRootCauseLocalizationTransformer(IHost host, ModelLoadContext ctx)
            : base(host, ctx)
        {
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));

            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>
            base.SaveColumns(ctx);
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private protected override void CheckInputColumn(DataViewSchema inputSchema, int col, int srcCol)
        {
            if (!(inputSchema[srcCol].Type is RootCauseLocalizationInputDataViewType))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[col].inputColumnName, "RootCauseLocalizationInputDataViewType", inputSchema[srcCol].Type.ToString());
        }

        private sealed class Mapper : OneToOneMapperBase
        {
            private DTRootCauseLocalizationTransformer _parent;

            public Mapper(DTRootCauseLocalizationTransformer parent, DataViewSchema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new DataViewSchema.DetachedColumn[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    InputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].inputColumnName, out int colIndex);
                    Host.Assert(colIndex >= 0);

                    DataViewType type;
                    type = new RootCauseDataViewType();

                    result[i] = new DataViewSchema.DetachedColumn(_parent.ColumnPairs[i].outputColumnName, type, null);
                }
                return result;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);

                var src = default(RootCauseLocalizationInput);
                var getSrc = input.GetGetter<RootCauseLocalizationInput>(input.Schema[ColMapNewToOld[iinfo]]);

                disposer =
                    () =>
                    {
                        if (src != null)
                        {
                            src.Dispose();
                            src = null;
                        }
                    };

                ValueGetter<RootCause> del =
                    (ref RootCause dst) =>
                    {
                        getSrc(ref src);
                        if (src == null)
                            return;

                        CheckInput(src, Host);

                        LocalizeRootCauses(src, ref dst);
                    };

                return del;
            }
        }

        private static void CheckInput(RootCauseLocalizationInput src, IHost host) {
            if (src.Slices.Count < 1)
            {
                throw host.Except($"Length of Slices must be larger than 0");
            }

            bool containsAnomalyTimestamp = false;
            foreach (MetricSlice slice in src.Slices) {
                if (slice.TimeStamp.Equals(src.AnomalyTimestamp)) {
                    containsAnomalyTimestamp = true;
                }
            }
            if (!containsAnomalyTimestamp) {
                throw host.Except($"Has no points in the given anomaly timestamp");
            }
        }

        private static void LocalizeRootCauses(RootCauseLocalizationInput src , ref RootCause dst) {
            dst = new RootCause();
            dst.Items = new List<RootCauseItem> {  };

            //todo - get total points
            GetRootCauseList(src, ref dst);
            //todo - need to update, temp for test
            GetRootCauseScore(new List<Point>(), src.AnomalyDimensions, ref dst);
        }

        private static void GetRootCauseList(RootCauseLocalizationInput src, ref RootCause dst) {
            DimensionInfo dimensionInfo = DTRootCauseLocalizationUtils.SeperateDimension(src.AnomalyDimensions, src.AggSymbol);
            if (dimensionInfo.AggDim.Count == 0) {
                return;
            }

            List<Point> points = new List<Point>();
            foreach (MetricSlice slice in src.Slices)
            {
                if (slice.TimeStamp.Equals(src.AnomalyTimestamp))
                {
                    points  = slice.Points;
                }
            }

            Dictionary<String, String> subDim = DTRootCauseLocalizationUtils.GetsubDim(src.AnomalyDimensions, dimensionInfo.DetailDim);

            List<Point> totalPoints = DTRootCauseLocalizationUtils.SelectPoints(points, subDim);

            PointTree pointTree = DTRootCauseLocalizationUtils.BuildPointTree(totalPoints, dimensionInfo.AggDim, subDim, src.AggSymbol);
            PointTree anomalyTree = DTRootCauseLocalizationUtils.BuildPointTree(totalPoints, dimensionInfo.AggDim,subDim,src.AggSymbol, true);

            // which means there is no all up here, we would return empty list; in ML.net , should we do the same thing？ todo
            if (anomalyTree.ParentNode == null) {
                return;
            }
            List<RootCauseItem> rootCauses = new List<RootCauseItem>();
            // no point under anomaly dimension
            if (totalPoints.Count == 0) {
                if (anomalyTree.Leaves.Count != 0) {
                    throw new Exception("point leaves not match with anomaly leaves");
                }

                rootCauses.AddRange(DTRootCauseLocalizationUtils.LocalizeRootCauseByAnomaly(totalPoints, anomalyTree, src.AnomalyDimensions));
            }
            else
            {
                double totalEntropy = 1;
                if (anomalyTree.Leaves.Count > 0)
                {
                    totalEntropy= DTRootCauseLocalizationUtils.GetEntropy(totalPoints.Count, anomalyTree.Leaves.Count);
                }

                if (totalEntropy > 0.9)
                {
                    if (dimensionInfo.AggDim.Count == 1) {
                        //root cause is itself;
                        rootCauses.Add(new RootCauseItem(src.AnomalyDimensions));
                    }
                    else {
                        rootCauses.AddRange(DTRootCauseLocalizationUtils.LocalizeRootCauseByDimension(totalPoints, anomalyTree,pointTree, totalEntropy,src.AnomalyDimensions));
                    }
                }
                else if (totalEntropy < 0.5)
                {
                    if (DTRootCauseLocalizationUtils.IsAnomalous(totalPoints.Count, anomalyTree.Leaves.Count)) {
                        //root cause is itself;
                        dst.Items.Add(new RootCauseItem(src.AnomalyDimensions));
                    }
                    else
                    {
                        rootCauses.AddRange(DTRootCauseLocalizationUtils.LocalizeRootCauseByAnomaly(totalPoints, anomalyTree,src.AnomalyDimensions));
                    }
                }
                else {
                    if (dimensionInfo.AggDim.Count == 1) {
                        //because we have known the dimension, so by anomaly
                        rootCauses.AddRange(DTRootCauseLocalizationUtils.LocalizeRootCauseByAnomaly(totalPoints, anomalyTree, src.AnomalyDimensions));
                    }
                    else {
                        rootCauses.AddRange(DTRootCauseLocalizationUtils.LocalizeRootCauseByDimension(totalPoints,anomalyTree, pointTree, totalEntropy, src.AnomalyDimensions));
                    }
                }

                dst.Items = rootCauses;
            }
        }

        private static void GetRootCauseScore(List<Point> points, Dictionary<string, string> anomalyRoot, ref RootCause dst)
        {

            if (dst.Items.Count > 1)
            {
                //get surprise value and explanary power value
                Point anomalyPoint = DTRootCauseLocalizationUtils.FindPointByDimension(anomalyRoot, points);

                double sumSurprise = 0;
                double sumEp = 0;
                List<RootCauseScore> scoreList = new List<RootCauseScore>();

                foreach (RootCauseItem item in dst.Items)
                {
                    Point rootCausePoint = DTRootCauseLocalizationUtils.FindPointByDimension(item.RootCause, points);
                    if (rootCausePoint != null)
                    {
                        if (rootCausePoint.ExpectedValue < rootCausePoint.Value)
                        {
                            item.Direction = AnomalyDirection.Up;
                        }
                        else
                        {
                            item.Direction = AnomalyDirection.Down;
                        }
                    }

                    if (anomalyPoint != null && rootCausePoint != null)
                    {
                        double surprise = GetSurpriseScore(rootCausePoint, anomalyPoint);

                        double ep = (rootCausePoint.Value - rootCausePoint.ExpectedValue) / (anomalyPoint.Value  - anomalyPoint.ExpectedValue);

                        scoreList.Add(new RootCauseScore(surprise, ep));
                        sumSurprise += surprise;
                        sumEp += Math.Abs(ep);
                    }
                }

                //normalize and get final score
                for (int i = 0; i < scoreList.Count; i++)
                {
                    dst.Items[i].Score = GetFinalScore(scoreList[i].Surprise / sumSurprise, Math.Abs(scoreList[i].ExplainaryScore) / sumEp);

                }
            }
            else if (dst.Items.Count == 1)
            {
                //surprise and expananory , max is 1
                Point rootCausePoint = DTRootCauseLocalizationUtils.FindPointByDimension(dst.Items[0].RootCause, points);
                if (rootCausePoint != null)
                {
                    if (rootCausePoint.ExpectedValue < rootCausePoint.Value)
                    {
                        dst.Items[0].Direction = AnomalyDirection.Up;
                    }
                    else
                    {
                        dst.Items[0].Direction = AnomalyDirection.Down;
                    }
                }

                Point anomalyPoint = DTRootCauseLocalizationUtils.FindPointByDimension(anomalyRoot, points);
                if (anomalyPoint != null && rootCausePoint != null)
                {
                    double surprise = GetSurpriseScore(rootCausePoint, anomalyPoint);

                    double ep = (rootCausePoint.Value - rootCausePoint.ExpectedValue) / (anomalyPoint.Value - anomalyPoint.ExpectedValue);
                    dst.Items[0].Score = GetFinalScore(surprise, ep);
                }

            }

        }

        private static  double GetSurpriseScore(Point rootCausePoint, Point anomalyPoint)
        {
            double p = rootCausePoint.ExpectedValue / anomalyPoint.ExpectedValue;
            double q = rootCausePoint.Value / anomalyPoint.Value;
            double surprise = 0.5 * (p * DTRootCauseLocalizationUtils.Log2(2 * p / (p + q)) + q * DTRootCauseLocalizationUtils.Log2(2 * q / (p + q)));

            return surprise;
        }

        private static double GetFinalScore(double surprise, double ep)
        {
            //return Math.Max(1, Parent.Beta * surprise + (1 - Parent.Beta) * ep);
            return 0;
        }
    }

    /// <summary>
    /// <see cref="IEstimator{TTransformer}"/> for the <see cref="DTRootCauseLocalizationTransformer"/>.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    /// ###  Estimator Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Does this estimator need to look at the data to train its parameters? | No |
    /// | Input column data type | <xref: Microsoft.ML.Transforms.TimeSeries.RootCauseLocalizationInput> |
    /// | Output column data type | <xref:System.Drawing.RootCause> |
    /// | Exportable to ONNX | No |
    ///
    /// The resulting <xref:Microsoft.ML.Transforms.Image.DTRootCauseLocalizationTransformer> creates a new column, named as specified in the output column name parameters, and
    /// localize the root causes which contribute most to the anomaly.
    /// Check the See Also section for links to usage examples.
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="TimeSeriesCatalog.LocalizeRootCauseByDT" />
    public sealed class DTRootCauseLocalizationEstimator : TrivialEstimator<DTRootCauseLocalizationTransformer>
    {
        internal static class Defaults
        {
            public const double Beta = 0.5;
        }

        public enum AggregateType
        {
            /// <summary>
            /// Make the aggregate type as sum.
            /// </summary>
            Sum = 0,
            /// <summary>
            /// Make the aggregate type as average.
            ///  </summary>
            Avg = 1,
            /// <summary>
            /// Make the aggregate type as min.
            /// </summary>
            Min = 2,
            /// <summary>
            /// Make the aggregate type as max.
            /// </summary>
            Max = 3
        }

        /// <summary>
        /// Localize root cause.
        /// </summary>
        /// <param name="env">The estimator's local <see cref="IHostEnvironment"/>.</param>
        /// <param name="columns">The name of the columns (first item of the tuple), and the name of the resulting output column (second item of the tuple).</param>
        /// <param name="beta">The weight for generating score in output result.</param>
        [BestFriend]
        internal DTRootCauseLocalizationEstimator(IHostEnvironment env, double beta = Defaults.Beta,params(string outputColumnName, string inputColumnName)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(DTRootCauseLocalizationEstimator)), new DTRootCauseLocalizationTransformer(env, beta,columns))
        {
        }

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            foreach (var colInfo in Transformer.Columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.inputColumnName, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.inputColumnName);
                if (!(col.ItemType is RootCauseLocalizationInputDataViewType) || col.Kind != SchemaShape.Column.VectorKind.Scalar)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.inputColumnName, new RootCauseLocalizationInputDataViewType().ToString(), col.GetTypeString());

                result[colInfo.outputColumnName] = new SchemaShape.Column(colInfo.outputColumnName, col.Kind, col.ItemType, col.IsKey, col.Annotations);
            }

            return new SchemaShape(result.Values);
        }
    }
}
