// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML.CommandLine;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Transforms;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// This interface is used by Maml components (the <see cref="EvaluateCommand"/>, the <see cref="CrossValidationCommand"/>
    /// and the <see cref="EvaluateTransform"/> to evaluate, print and save the results.
    /// The input <see cref="RoleMappedData"/> to the <see cref="IEvaluator.Evaluate"/> and the <see cref="IEvaluator.GetPerInstanceMetrics"/> methods
    /// should be assumed to contain only the following column roles: label, group, weight and name. Any other columns needed for
    /// evaluation should be searched for by name in the <see cref="RoleMappedData.Schema"/>.
    /// </summary>
    [BestFriend]
    internal interface IMamlEvaluator : IEvaluator
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
    /// A base class implementation of <see cref="IMamlEvaluator"/>. The <see cref="Evaluate"/> and <see cref="IEvaluator.GetPerInstanceMetrics"/>
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

        internal static RoleMappedSchema.ColumnRole Strat = "Strat";
        [BestFriend]
        private protected readonly IHost Host;

        [BestFriend]
        private protected readonly string ScoreColumnKind;
        [BestFriend]
        private protected readonly string ScoreCol;
        [BestFriend]
        private protected readonly string LabelCol;
        [BestFriend]
        private protected readonly string WeightCol;
        [BestFriend]
        private protected readonly string[] StratCols;

        [BestFriend]
        private protected abstract IEvaluator Evaluator { get; }

        [BestFriend]
        private protected MamlEvaluatorBase(ArgumentsBase args, IHostEnvironment env, string scoreColumnKind, string registrationName)
        {
            Contracts.CheckValue(env, nameof(env));
            Host = env.Register(registrationName);
            ScoreColumnKind = scoreColumnKind;
            ScoreCol = args.ScoreColumn;
            LabelCol = args.LabelColumn;
            WeightCol = args.WeightColumn;
            StratCols = args.StratColumn;
        }

        Dictionary<string, IDataView> IEvaluator.Evaluate(RoleMappedData data)
        {
            data = new RoleMappedData(data.Data, GetInputColumnRoles(data.Schema, needStrat: true));
            return Evaluator.Evaluate(data);
        }

        [BestFriend]
        private protected IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> GetInputColumnRoles(RoleMappedSchema schema, bool needStrat = false, bool needName = false)
        {
            Host.CheckValue(schema, nameof(schema));

            var roles = !needStrat || StratCols == null
                ? Enumerable.Empty<KeyValuePair<RoleMappedSchema.ColumnRole, string>>()
                : StratCols.Select(col => RoleMappedSchema.CreatePair(Strat, col));

            if (needName && schema.Name.HasValue)
                roles = MetadataUtils.Prepend(roles, RoleMappedSchema.ColumnRole.Name.Bind(schema.Name.Value.Name));

            return roles.Concat(GetInputColumnRolesCore(schema));
        }

        /// <summary>
        /// All the input columns needed by an evaluator should be added here.
        /// The base class ipmlementation gets the score column, the label column (if exists) and the weight column (if exists).
        /// Override if additional columns are needed.
        /// </summary>
        [BestFriend]
        private protected virtual IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> GetInputColumnRolesCore(RoleMappedSchema schema)
        {
            // Get the score column information.
            var scoreCol = EvaluateUtils.GetScoreColumn(Host, schema.Schema, ScoreCol, nameof(ArgumentsBase.ScoreColumn),
                ScoreColumnKind);
            yield return RoleMappedSchema.CreatePair(MetadataUtils.Const.ScoreValueKind.Score, scoreCol.Name);

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

        void IMamlEvaluator.PrintFoldResults(IChannel ch, Dictionary<string, IDataView> metrics)
        {
            Host.CheckValue(ch, nameof(ch));
            Host.CheckValue(metrics, nameof(metrics));
            PrintFoldResultsCore(ch, metrics);
        }

        /// <summary>
        /// This method simply prints the overall metrics using EvaluateUtils.PrintConfusionMatrixAndPerFoldResults.
        /// Override if something else is needed.
        /// </summary>
        [BestFriend]
        private protected virtual void PrintFoldResultsCore(IChannel ch, Dictionary<string, IDataView> metrics)
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

        IDataView IMamlEvaluator.GetOverallResults(params IDataView[] metrics)
        {
            Host.CheckNonEmpty(metrics, nameof(metrics));
            var overall = CombineOverallMetricsCore(metrics);
            return GetOverallResultsCore(overall);
        }

        [BestFriend]
        private protected virtual IDataView CombineOverallMetricsCore(IDataView[] metrics)
        {
            return EvaluateUtils.ConcatenateOverallMetrics(Host, metrics);
        }

        [BestFriend]
        private protected virtual IDataView GetOverallResultsCore(IDataView overall)
        {
            return overall;
        }

        void IMamlEvaluator.PrintAdditionalMetrics(IChannel ch, params Dictionary<string, IDataView>[] metrics)
        {
            Host.CheckValue(ch, nameof(ch));
            Host.CheckNonEmpty(metrics, nameof(metrics));
            PrintAdditionalMetricsCore(ch, metrics);
        }

        /// <summary>
        /// This method simply prints the overall metrics using EvaluateUtils.PrintOverallMetrics.
        /// Override if something else is needed.
        /// </summary>
        [BestFriend]
        private protected virtual void PrintAdditionalMetricsCore(IChannel ch, Dictionary<string, IDataView>[] metrics)
        {
        }

        IDataTransform IEvaluator.GetPerInstanceMetrics(RoleMappedData scoredData)
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
            var cols = new List<(string name, string source)>();
            var colsToKeep = new List<string>();

            // If perInst is the result of cross-validation and contains a fold Id column, include it.
            int foldCol;
            if (perInst.Schema.Schema.TryGetColumnIndex(MetricKinds.ColumnNames.FoldIndex, out foldCol))
                colsToKeep.Add(MetricKinds.ColumnNames.FoldIndex);

            // Maml always outputs a name column, if it doesn't exist add a GenerateNumberTransform.
            if (perInst.Schema.Name?.Name is string nameName)
            {
                cols.Add(("Instance", nameName));
                colsToKeep.Add("Instance");
            }
            else
            {
                var args = new GenerateNumberTransform.Arguments();
                args.Column = new[] { new GenerateNumberTransform.Column() { Name = "Instance" } };
                args.UseCounter = true;
                idv = new GenerateNumberTransform(Host, args, idv);
                colsToKeep.Add("Instance");
            }

            // Maml outputs the weight column if it exists.
            if (perInst.Schema.Weight?.Name is string weightName)
                colsToKeep.Add(weightName);

            // Get the other columns from the evaluator.
            foreach (var col in GetPerInstanceColumnsToSave(perInst.Schema))
                colsToKeep.Add(col);

            idv = new ColumnCopyingTransformer(Host, cols.ToArray()).Transform(idv);
            idv = ColumnSelectingTransformer.CreateKeep(Host, idv, colsToKeep.ToArray());
            return GetPerInstanceMetricsCore(idv, perInst.Schema);
        }

        /// <summary>
        /// The perInst dataview contains all a name column (called Instance), the FoldId, Label and Weight columns if
        /// they exist, and all the columns returned by <see cref="GetPerInstanceColumnsToSave"/>.
        /// It should be overridden only if additional processing is needed, such as dropping slots in the "top k scores" column
        /// in the multi-class case.
        /// </summary>
        [BestFriend]
        private protected virtual IDataView GetPerInstanceMetricsCore(IDataView perInst, RoleMappedSchema schema)
        {
            return perInst;
        }

        IDataView IMamlEvaluator.GetPerInstanceDataViewToSave(RoleMappedData perInstance)
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
        [BestFriend]
        private protected abstract IEnumerable<string> GetPerInstanceColumnsToSave(RoleMappedSchema schema);
    }
}
