// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Transforms;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// This interface is used by Maml components (the <see cref="EvaluateCommand"/>, the <see cref="CrossValidationCommand"/>
    /// and the <see cref="EvaluateTransform"/> to evaluate, print and save the results.
    /// The input <see cref="RoleMappedData"/> to the <see cref="IEvaluator.Evaluate"/> and the <see cref="IEvaluator.GetPerInstanceMetrics"/> methods
    /// should be assumed to contain only the following column roles: label, group, weight and name. Any other columns needed for
    /// evaluation should be searched for by name in the <see cref="ISchema"/>.
    /// </summary>
    public interface IMamlEvaluator : IEvaluator
    {
        /// <summary>
        /// Print the aggregate metrics to the console.
        /// </summary>
        void PrintFoldResults(IChannel ch, Dictionary<string, IDataView> metrics);

        /// <summary>
        /// Combine the overall metrics from multiple folds into a single data view.
        /// </summary>
        /// <param name="metrics"></param>
        /// <returns></returns>
        IDataView GetOverallResults(params IDataView[] metrics);

        /// <summary>
        /// Handles custom metrics (such as p/r curves for binary classification, or group summary results for ranking) from one
        /// or more folds. Implementations of this method typically creates a single data view for the custom metric and saves it
        /// to a user specified file.
        /// </summary>
        void PrintAdditionalMetrics(IChannel ch, params Dictionary<string, IDataView>[] metrics);

        /// <summary>
        /// Create a data view containing only the columns that are saved as per-instance results by Maml commands.
        /// </summary>
        IDataView GetPerInstanceDataViewToSave(RoleMappedData perInstance);
    }

    /// <summary>
    /// A base class implementation of <see cref="IMamlEvaluator"/>. The <see cref="Evaluate"/> and <see cref="GetPerInstanceMetrics"/>
    /// methods create a new <see cref="RoleMappedData"/> containing all the columns needed for evaluation, and call the corresponding
    /// methods on an <see cref="IEvaluator"/> of the appropriate type.
    /// </summary>
    public abstract class MamlEvaluatorBase : IMamlEvaluator
    {
        public abstract class ArgumentsBase : EvaluateInputBase
        {
            // Standard columns.

            [Argument(ArgumentType.AtMostOnce, HelpText = "Column to use for labels.", ShortName = "lab")]
            public string LabelColumn;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Weight column name.", ShortName = "weight")]
            public string WeightColumn;

            // Score columns.

            [Argument(ArgumentType.AtMostOnce, HelpText = "Score column name.", ShortName = "score")]
            public string ScoreColumn;

            // Stratification columns.

            [Argument(ArgumentType.Multiple, HelpText = "Stratification column name.", ShortName = "strat")]
            public string[] StratColumn;
        }

        public static RoleMappedSchema.ColumnRole Strat = "Strat";
        protected readonly IHost Host;

        protected readonly string ScoreColumnKind;
        protected readonly string ScoreCol;
        protected readonly string LabelCol;
        protected readonly string WeightCol;
        protected readonly string[] StratCols;

        protected abstract IEvaluator Evaluator { get; }

        protected MamlEvaluatorBase(ArgumentsBase args, IHostEnvironment env, string scoreColumnKind, string registrationName)
        {
            Contracts.CheckValue(env, nameof(env));
            Host = env.Register(registrationName);
            ScoreColumnKind = scoreColumnKind;
            ScoreCol = args.ScoreColumn;
            LabelCol = args.LabelColumn;
            WeightCol = args.WeightColumn;
            StratCols = args.StratColumn;
        }

        public Dictionary<string, IDataView> Evaluate(RoleMappedData data)
        {
            data = new RoleMappedData(data.Data, GetInputColumnRoles(data.Schema, needStrat: true));
            return Evaluator.Evaluate(data);
        }

        protected IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> GetInputColumnRoles(RoleMappedSchema schema, bool needStrat = false, bool needName = false)
        {
            Host.CheckValue(schema, nameof(schema));

            var roles = !needStrat || StratCols == null
                ? Enumerable.Empty<KeyValuePair<RoleMappedSchema.ColumnRole, string>>()
                : StratCols.Select(col => RoleMappedSchema.CreatePair(Strat, col));

            if (needName && schema.Name != null)
                roles = roles.Prepend(RoleMappedSchema.ColumnRole.Name.Bind(schema.Name.Name));

            return roles.Concat(GetInputColumnRolesCore(schema));
        }

        /// <summary>
        /// All the input columns needed by an evaluator should be added here.
        /// The base class ipmlementation gets the score column, the label column (if exists) and the weight column (if exists).
        /// Override if additional columns are needed.
        /// </summary>
        protected virtual IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> GetInputColumnRolesCore(RoleMappedSchema schema)
        {
            // Get the score column information.
            var scoreInfo = EvaluateUtils.GetScoreColumnInfo(Host, schema.Schema, ScoreCol, nameof(ArgumentsBase.ScoreColumn),
                ScoreColumnKind);
            yield return RoleMappedSchema.CreatePair(MetadataUtils.Const.ScoreValueKind.Score, scoreInfo.Name);

            // Get the label column information.
            string label = EvaluateUtils.GetColName(LabelCol, schema.Label, DefaultColumnNames.Label);
            yield return RoleMappedSchema.ColumnRole.Label.Bind(label);

            string weight = EvaluateUtils.GetColName(WeightCol, schema.Weight, null);
            if (!string.IsNullOrEmpty(weight))
                yield return RoleMappedSchema.ColumnRole.Weight.Bind(weight);
        }

        public virtual IEnumerable<MetricColumn> GetOverallMetricColumns()
        {
            return Evaluator.GetOverallMetricColumns();
        }

        public void PrintFoldResults(IChannel ch, Dictionary<string, IDataView> metrics)
        {
            Host.CheckValue(ch, nameof(ch));
            Host.CheckValue(metrics, nameof(metrics));
            PrintFoldResultsCore(ch, metrics);
        }

        /// <summary>
        /// This method simply prints the overall metrics using EvaluateUtils.PrintConfusionMatrixAndPerFoldResults.
        /// Override if something else is needed.
        /// </summary>
        protected virtual void PrintFoldResultsCore(IChannel ch, Dictionary<string, IDataView> metrics)
        {
            ch.AssertValue(ch);
            ch.AssertValue(metrics);

            IDataView fold;
            if (!metrics.TryGetValue(MetricKinds.OverallMetrics, out fold))
                throw ch.Except("No overall metrics found");

            string weightedMetrics;
            string unweightedMetrics = MetricWriter.GetPerFoldResults(Host, fold, out weightedMetrics);
            if (!string.IsNullOrEmpty(weightedMetrics))
                ch.Info(weightedMetrics);
            ch.Info(unweightedMetrics);
        }

        public IDataView GetOverallResults(params IDataView[] metrics)
        {
            Host.CheckNonEmpty(metrics, nameof(metrics));
            var overall = CombineOverallMetricsCore(metrics);
            return GetOverallResultsCore(overall);
        }

        protected virtual IDataView CombineOverallMetricsCore(IDataView[] metrics)
        {
            return EvaluateUtils.ConcatenateOverallMetrics(Host, metrics);
        }

        protected virtual IDataView GetOverallResultsCore(IDataView overall)
        {
            return overall;
        }

        public void PrintAdditionalMetrics(IChannel ch, params Dictionary<string, IDataView>[] metrics)
        {
            Host.CheckValue(ch, nameof(ch));
            Host.CheckNonEmpty(metrics, nameof(metrics));
            PrintAdditionalMetricsCore(ch, metrics);
        }

        /// <summary>
        /// This method simply prints the overall metrics using EvaluateUtils.PrintOverallMetrics.
        /// Override if something else is needed.
        /// </summary>
        protected virtual void PrintAdditionalMetricsCore(IChannel ch, Dictionary<string, IDataView>[] metrics)
        {
        }

        public IDataTransform GetPerInstanceMetrics(RoleMappedData scoredData)
        {
            Host.AssertValue(scoredData);

            var schema = scoredData.Schema;
            var dataEval = new RoleMappedData(scoredData.Data, GetInputColumnRoles(schema));
            return Evaluator.GetPerInstanceMetrics(dataEval);
        }

        private IDataView WrapPerInstance(RoleMappedData perInst)
        {
            var idv = perInst.Data;

            // Make a list of column names that Maml outputs as part of the per-instance data view, and then wrap
            // the per-instance data computed by the evaluator in a SelectColumnsTransform.
            var cols = new List<(string Source, string Name)>();
            var colsToKeep = new List<string>();

            // If perInst is the result of cross-validation and contains a fold Id column, include it.
            int foldCol;
            if (perInst.Schema.Schema.TryGetColumnIndex(MetricKinds.ColumnNames.FoldIndex, out foldCol))
                colsToKeep.Add(MetricKinds.ColumnNames.FoldIndex);

            // Maml always outputs a name column, if it doesn't exist add a GenerateNumberTransform.
            if (perInst.Schema.Name == null)
            {
                var args = new NumberGeneratingTransformer.Arguments();
                args.Column = new[] { new NumberGeneratingTransformer.Column() { Name = "Instance" } };
                args.UseCounter = true;
                idv = new NumberGeneratingTransformer(Host, args, idv);
                colsToKeep.Add("Instance");
            }
            else
            {
                cols.Add((perInst.Schema.Name.Name, "Instance"));
                colsToKeep.Add("Instance");
            }

            // Maml outputs the weight column if it exists.
            if (perInst.Schema.Weight != null)
                colsToKeep.Add(perInst.Schema.Weight.Name);

            // Get the other columns from the evaluator.
            foreach (var col in GetPerInstanceColumnsToSave(perInst.Schema))
                colsToKeep.Add(col);

            idv = new ColumnsCopyingTransformer(Host, cols.ToArray()).Transform(idv);
            idv = SelectColumnsTransform.CreateKeep(Host, idv, colsToKeep.ToArray());
            return GetPerInstanceMetricsCore(idv, perInst.Schema);
        }

        /// <summary>
        /// The perInst dataview contains all a name column (called Instance), the FoldId, Label and Weight columns if
        /// they exist, and all the columns returned by <see cref="GetPerInstanceColumnsToSave"/>.
        /// It should be overridden only if additional processing is needed, such as dropping slots in the "top k scores" column
        /// in the multi-class case.
        /// </summary>
        protected virtual IDataView GetPerInstanceMetricsCore(IDataView perInst, RoleMappedSchema schema)
        {
            return perInst;
        }

        public IDataView GetPerInstanceDataViewToSave(RoleMappedData perInstance)
        {
            Host.CheckValue(perInstance, nameof(perInstance));
            var data = new RoleMappedData(perInstance.Data, GetInputColumnRoles(perInstance.Schema, needName: true));
            return WrapPerInstance(data);
        }

        /// <summary>
        /// Returns the names of the columns that should be saved in the per-instance results file. These can include
        /// the columns generated by the corresponding <see cref="IRowMapper"/>, or any of the input columns used by
        /// it. The Name and Weight columns should not be included, since the base class includes them automatically.
        /// </summary>
        protected abstract IEnumerable<string> GetPerInstanceColumnsToSave(RoleMappedSchema schema);
    }
}
