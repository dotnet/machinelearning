// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.StaticPipe.Runtime;

[assembly: LoadableClass(CategoricalTransform.Summary, typeof(IDataTransform), typeof(CategoricalTransform), typeof(CategoricalTransform.Arguments), typeof(SignatureDataTransform),
    CategoricalTransform.UserName, "CategoricalTransform", "CatTransform", "Categorical", "Cat")]

[assembly: LoadableClass(typeof(void), typeof(Categorical), null, typeof(SignatureEntryPointModule), "Categorical")]
namespace Microsoft.ML.Runtime.Data
{
    /// <include file='doc.xml' path='doc/members/member[@name="CategoricalOneHotVectorizer"]/*' />
    public sealed class CategoricalTransform : ITransformer, ICanSaveModel
    {
        public enum OutputKind : byte
        {
            /// <summary>
            /// Output is a bag (multi-set) vector
            /// </summary>
            [TGUI(Label = "Output is a bag (multi-set) vector")]
            Bag = 1,

            /// <summary>
            /// Output is an indicator vector
            /// </summary>
            [TGUI(Label = "Output is an indicator vector")]
            Ind = 2,

            /// <summary>
            /// Output is a key value
            /// </summary>
            [TGUI(Label = "Output is a key value")]
            Key = 3,

            /// <summary>
            /// Output is binary encoded
            /// </summary>
            [TGUI(Label = "Output is binary encoded")]
            Bin = 4,
        }

        public sealed class Column : TermTransform.ColumnBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Output kind: Bag (multi-set vector), Ind (indicator vector), Key (index), or Binary encoded indicator vector", ShortName = "kind")]
            public OutputKind? OutputKind;

            public static Column Parse(string str)
            {
                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            protected override bool TryParse(string str)
            {
                Contracts.AssertNonEmpty(str);

                // We accept N:K:S where N is the new column name, K is the output kind,
                // and S is source column names.
                if (!TryParse(str, out string extra))
                    return false;
                if (extra == null)
                    return true;
                if (!Enum.TryParse(extra, true, out OutputKind kind))
                    return false;
                OutputKind = kind;
                return true;
            }

            public bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (OutputKind == null)
                    return TryUnparseCore(sb);
                var kind = OutputKind.Value;
                if (!Enum.IsDefined(typeof(OutputKind), kind))
                    return false;
                string extra = OutputKind.Value.ToString();
                return TryUnparseCore(sb, extra);
            }
        }

        public sealed class Arguments : TermTransform.ArgumentsBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Output kind: Bag (multi-set vector), Ind (indicator vector), or Key (index)",
                ShortName = "kind", SortOrder = 102)]
            public OutputKind OutputKind = CategoricalEstimator.Defaults.OutKind;

            public Arguments()
            {
                // Unlike in the term transform, we want the text key values for the categorical transform
                // to default to true.
                TextKeyValues = true;
            }
        }

        internal const string Summary = "Converts the categorical value into an indicator array by building a dictionary of categories based on the "
            + "data and using the id in the dictionary as the index in the array.";

        public const string UserName = "Categorical Transform";

        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register("Categorical");
            h.CheckValue(args, nameof(args));
            h.CheckValue(input, nameof(input));
            h.CheckUserArg(Utils.Size(args.Column) > 0, nameof(args.Column));

            var columns = new List<CategoricalEstimator.ColumnInfo>();
            foreach (var column in args.Column)
            {
                var col = new CategoricalEstimator.ColumnInfo(
                    column.Source ?? column.Name,
                    column.Name,
                    column.OutputKind ?? args.OutputKind,
                    column.MaxNumTerms ?? args.MaxNumTerms,
                    column.Sort ?? args.Sort,
                    column.Term ?? args.Term);
                col.SetTerms(column.Terms ?? args.Terms);
                columns.Add(col);
            }
            return new CategoricalEstimator(env, columns.ToArray(), args.DataFile, args.TermsColumn, args.Loader).Fit(input).Transform(input) as IDataTransform;
        }

        private readonly TransformerChain<ITransformer> _transformer;

        public CategoricalTransform(TermEstimator term, IEstimator<ITransformer> toVector, IDataView input)
        {
            if (toVector != null)
                _transformer = term.Append(toVector).Fit(input);
            else
                _transformer = new TransformerChain<ITransformer>(term.Fit(input));
        }

        public Schema GetOutputSchema(Schema inputSchema) => _transformer.GetOutputSchema(inputSchema);

        public IDataView Transform(IDataView input) => _transformer.Transform(input);

        public void Save(ModelSaveContext ctx) => _transformer.Save(ctx);

        public bool IsRowToRowMapper => _transformer.IsRowToRowMapper;

        public IRowToRowMapper GetRowToRowMapper(Schema inputSchema) => _transformer.GetRowToRowMapper(inputSchema);
    }
    /// <summary>
    /// Estimator which takes set of columns and produce for each column indicator array.
    /// </summary>
    public sealed class CategoricalEstimator : IEstimator<CategoricalTransform>
    {
        internal static class Defaults
        {
            public const CategoricalTransform.OutputKind OutKind = CategoricalTransform.OutputKind.Ind;
        }

        public class ColumnInfo : TermTransform.ColumnInfo
        {
            public readonly CategoricalTransform.OutputKind OutputKind;
            public ColumnInfo(string input, string output, CategoricalTransform.OutputKind outputKind = Defaults.OutKind,
                int maxNumTerms = TermEstimator.Defaults.MaxNumTerms, TermTransform.SortOrder sort = TermEstimator.Defaults.Sort,
                string[] term = null)
                : base(input, output, maxNumTerms, sort, term, true)
            {
                OutputKind = outputKind;
            }

            internal void SetTerms(string terms)
            {
                Terms = terms;
            }

        }

        private readonly IHost _host;
        private readonly IEstimator<ITransformer> _toSomething;
        private TermEstimator _term;

        /// A helper method to create <see cref="CategoricalEstimator"/> for public facing API.
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Name of the column to be transformed.</param>
        /// <param name="output">Name of the output column. If this is <c>null</c>, <paramref name="input"/> is used.</param>
        /// <param name="outputKind">The type of output expected.</param>
        public CategoricalEstimator(IHostEnvironment env, string input,
            string output = null, CategoricalTransform.OutputKind outputKind = Defaults.OutKind)
            : this(env, new ColumnInfo(input, output ?? input, outputKind))
        {
        }

        public CategoricalEstimator(IHostEnvironment env, ColumnInfo[] columns,
            string file = null, string termsColumn = null,
            IComponentFactory<IMultiStreamSource, IDataLoader> loaderFactory = null)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(TermEstimator));
            _term = new TermEstimator(_host, columns, file, termsColumn, loaderFactory);
            var binaryCols = new List<(string input, string output)>();
            var cols = new List<(string input, string output, bool bag)>();
            for (int i = 0; i < columns.Length; i++)
            {
                var column = columns[i];
                CategoricalTransform.OutputKind kind = columns[i].OutputKind;
                switch (kind)
                {
                    default:
                        throw _host.ExceptUserArg(nameof(column.OutputKind));
                    case CategoricalTransform.OutputKind.Key:
                        continue;
                    case CategoricalTransform.OutputKind.Bin:
                        binaryCols.Add((column.Output, column.Output));
                        break;
                    case CategoricalTransform.OutputKind.Ind:
                        cols.Add((column.Output, column.Output, false));
                        break;
                    case CategoricalTransform.OutputKind.Bag:
                        cols.Add((column.Output, column.Output, true));
                        break;
                }
            }
            IEstimator<ITransformer> toBinVector = null;
            IEstimator<ITransformer> toVector = null;
            if (binaryCols.Count > 0)
                toBinVector = new KeyToBinaryVectorEstimator(_host, binaryCols.Select(x => new KeyToBinaryVectorTransform.ColumnInfo(x.input, x.output)).ToArray());
            if (cols.Count > 0)
                toVector = new KeyToVectorEstimator(_host, cols.Select(x => new KeyToVectorTransform.ColumnInfo(x.input, x.output, x.bag)).ToArray());

            if (toBinVector != null && toVector != null)
                _toSomething = toVector.Append(toBinVector);
            else
            {
                if (toBinVector != null)
                    _toSomething = toBinVector;
                else
                    _toSomething = toVector;
            }
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema) => _term.Append(_toSomething).GetOutputSchema(inputSchema);

        public CategoricalTransform Fit(IDataView input) => new CategoricalTransform(_term, _toSomething, input);

        internal void WrapTermWithDelegate(Action<TermTransform> onFit)
        {
            _term = (TermEstimator)_term.WithOnFitDelegate(onFit);
        }
    }

    public static class Categorical
    {
        [TlcModule.EntryPoint(Name = "Transforms.CategoricalOneHotVectorizer",
            Desc = CategoricalTransform.Summary,
            UserName = CategoricalTransform.UserName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/member[@name=""CategoricalOneHotVectorizer""]/*' />",
                                 @"<include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/example[@name=""CategoricalOneHotVectorizer""]/*' />"})]
        public static CommonOutputs.TransformOutput CatTransformDict(IHostEnvironment env, CategoricalTransform.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("CatTransformDict");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var xf = CategoricalTransform.Create(host, input, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModel(env, xf, input.Data), OutputData = xf };
        }

        [TlcModule.EntryPoint(Name = "Transforms.CategoricalHashOneHotVectorizer",
            Desc = CategoricalHashTransform.Summary,
            UserName = CategoricalHashTransform.UserName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/member[@name=""CategoricalHashOneHotVectorizer""]/*' />",
                                 @"<include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/example[@name=""CategoricalHashOneHotVectorizer""]/*' />"})]
        public static CommonOutputs.TransformOutput CatTransformHash(IHostEnvironment env, CategoricalHashTransform.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("CatTransformDict");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var xf = CategoricalHashTransform.Create(host, input, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModel(env, xf, input.Data), OutputData = xf };
        }

        [TlcModule.EntryPoint(Name = "Transforms.TextToKeyConverter",
            Desc = TermTransform.Summary,
            UserName = TermTransform.FriendlyName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Data/Transforms/doc.xml' path='doc/members/member[@name=""TextToKey""]/*' />",
                                 @"<include file='../Microsoft.ML.Data/Transforms/doc.xml' path='doc/members/example[@name=""TextToKey""]/*' />" })]
        public static CommonOutputs.TransformOutput TextToKey(IHostEnvironment env, TermTransform.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("Term");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var xf = TermTransform.Create(host, input, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModel(env, xf, input.Data), OutputData = xf };
        }

        [TlcModule.EntryPoint(Name = "Transforms.KeyToTextConverter",
            Desc = "KeyToValueTransform utilizes KeyValues metadata to map key indices to the corresponding values in the KeyValues metadata.",
            UserName = KeyToValueTransform.UserName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/member[@name=""KeyToText""]/*' />" })]
        public static CommonOutputs.TransformOutput KeyToText(IHostEnvironment env, KeyToValueTransform.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("KeyToValue");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var xf = KeyToValueTransform.Create(host, input, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModel(env, xf, input.Data), OutputData = xf };
        }
    }

    public static class CategoricalStaticExtensions
    {
        public enum OneHotVectorOutputKind : byte
        {
            /// <summary>
            /// Output is a bag (multi-set) vector
            /// </summary>
            Bag = 1,

            /// <summary>
            /// Output is an indicator vector
            /// </summary>
            Ind = 2,

            /// <summary>
            /// Output is binary encoded
            /// </summary>
            Bin = 4,
        }

        public enum OneHotScalarOutputKind : byte
        {
            /// <summary>
            /// Output is an indicator vector
            /// </summary>
            Ind = 2,

            /// <summary>
            /// Output is binary encoded
            /// </summary>
            Bin = 4,
        }

        private const KeyValueOrder DefSort = (KeyValueOrder)TermEstimator.Defaults.Sort;
        private const int DefMax = TermEstimator.Defaults.MaxNumTerms;
        private const OneHotVectorOutputKind DefOut = (OneHotVectorOutputKind)CategoricalEstimator.Defaults.OutKind;

        private struct Config
        {
            public readonly KeyValueOrder Order;
            public readonly int Max;
            public readonly OneHotVectorOutputKind OutputKind;
            public readonly Action<TermTransform.TermMap> OnFit;

            public Config(OneHotVectorOutputKind outputKind, KeyValueOrder order, int max, Action<TermTransform.TermMap> onFit)
            {
                OutputKind = outputKind;
                Order = order;
                Max = max;
                OnFit = onFit;
            }
        }

        private static Action<TermTransform.TermMap> Wrap<T>(ToKeyFitResult<T>.OnFit onFit)
        {
            if (onFit == null)
                return null;
            // The type T asociated with the delegate will be the actual value type once #863 goes in.
            // However, until such time as #863 goes in, it would be too awkward to attempt to extract the metadata.
            // For now construct the useless object then pass it into the delegate.
            return map => onFit(new ToKeyFitResult<T>(map));
        }

        private interface ICategoricalCol
        {
            PipelineColumn Input { get; }
            Config Config { get; }
        }

        private sealed class ImplScalar<T> : Vector<float>, ICategoricalCol
        {
            public PipelineColumn Input { get; }
            public Config Config { get; }
            public ImplScalar(PipelineColumn input, Config config) : base(Rec.Inst, input)
            {
                Input = input;
                Config = config;
            }
        }

        private sealed class ImplVector<T> : Vector<float>, ICategoricalCol
        {
            public PipelineColumn Input { get; }
            public Config Config { get; }
            public ImplVector(PipelineColumn input, Config config) : base(Rec.Inst, input)
            {
                Input = input;
                Config = config;
            }
        }

        private sealed class Rec : EstimatorReconciler
        {
            public static readonly Rec Inst = new Rec();

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env, PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames, IReadOnlyDictionary<PipelineColumn, string> outputNames, IReadOnlyCollection<string> usedNames)
            {
                var infos = new CategoricalEstimator.ColumnInfo[toOutput.Length];
                Action<TermTransform> onFit = null;
                for (int i = 0; i < toOutput.Length; ++i)
                {
                    var tcol = (ICategoricalCol)toOutput[i];
                    infos[i] = new CategoricalEstimator.ColumnInfo(inputNames[tcol.Input], outputNames[toOutput[i]], (CategoricalTransform.OutputKind)tcol.Config.OutputKind,
                        tcol.Config.Max, (TermTransform.SortOrder)tcol.Config.Order);
                    if (tcol.Config.OnFit != null)
                    {
                        int ii = i; // Necessary because if we capture i that will change to toOutput.Length on call.
                        onFit += tt => tcol.Config.OnFit(tt.GetTermMap(ii));
                    }
                }
                var est = new CategoricalEstimator(env, infos);
                if (onFit != null)
                    est.WrapTermWithDelegate(onFit);
                return est;
            }
        }

        /// <summary>
        /// Converts the categorical value into an indicator array by building a dictionary of categories based on the data and using the id in the dictionary as the index in the array.
        /// </summary>
        /// <param name="input">Incoming data.</param>
        /// <param name="outputKind">Specify the output type of indicator array: array or binary encoded data.</param>
        /// <param name="order">How the Id for each value would be assigined: by occurrence or by value.</param>
        /// <param name="maxItems">Maximum number of ids to keep during data scanning.</param>
        /// <param name="onFit">Called upon fitting with the learnt enumeration on the dataset.</param>
        public static Vector<float> OneHotEncoding(this Scalar<string> input, OneHotScalarOutputKind outputKind = (OneHotScalarOutputKind)DefOut, KeyValueOrder order = DefSort,
            int maxItems = DefMax, ToKeyFitResult<ReadOnlyMemory<char>>.OnFit onFit = null)
        {
            Contracts.CheckValue(input, nameof(input));
            return new ImplScalar<string>(input, new Config((OneHotVectorOutputKind)outputKind, order, maxItems, Wrap(onFit)));
        }

        /// <summary>
        /// Converts the categorical value into an indicator array by building a dictionary of categories based on the data and using the id in the dictionary as the index in the array.
        /// </summary>
        /// <param name="input">Incoming data.</param>
        /// <param name="outputKind">Specify the output type of indicator array: Multiarray, array or binary encoded data.</param>
        /// <param name="order">How the Id for each value would be assigined: by occurrence or by value.</param>
        /// <param name="maxItems">Maximum number of ids to keep during data scanning.</param>
        /// <param name="onFit">Called upon fitting with the learnt enumeration on the dataset.</param>
        public static Vector<float> OneHotEncoding(this Vector<string> input, OneHotVectorOutputKind outputKind = DefOut, KeyValueOrder order = DefSort, int maxItems = DefMax,
            ToKeyFitResult<ReadOnlyMemory<char>>.OnFit onFit = null)
        {
            Contracts.CheckValue(input, nameof(input));
            return new ImplVector<string>(input, new Config(outputKind, order, maxItems, Wrap(onFit)));
        }
    }
}
