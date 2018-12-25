// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Transforms.Categorical;
using Microsoft.ML.Transforms.Conversions;

[assembly: LoadableClass(OneHotEncodingTransformer.Summary, typeof(IDataTransform), typeof(OneHotEncodingTransformer), typeof(OneHotEncodingTransformer.Arguments), typeof(SignatureDataTransform),
    OneHotEncodingTransformer.UserName, "CategoricalTransform", "CatTransform", "Categorical", "Cat")]

[assembly: LoadableClass(typeof(void), typeof(Categorical), null, typeof(SignatureEntryPointModule), "Categorical")]

namespace Microsoft.ML.Transforms.Categorical
{
    /// <include file='doc.xml' path='doc/members/member[@name="CategoricalOneHotVectorizer"]/*' />
    public sealed class OneHotEncodingTransformer : ITransformer, ICanSaveModel
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

        public sealed class Column : ValueToKeyMappingTransformer.ColumnBase
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

        public sealed class Arguments : ValueToKeyMappingTransformer.ArgumentsBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Output kind: Bag (multi-set vector), Ind (indicator vector), or Key (index)",
                ShortName = "kind", SortOrder = 102)]
            public OutputKind OutputKind = OneHotEncodingEstimator.Defaults.OutKind;

            public Arguments()
            {
                // Unlike in the term transform, we want the text key values for the categorical transform
                // to default to true.
                TextKeyValues = true;
            }
        }

        internal const string Summary = "Converts the categorical value into an indicator array by building a dictionary of categories based on the "
            + "data and using the id in the dictionary as the index in the array.";

        internal const string UserName = "Categorical Transform";

        internal static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register("Categorical");
            h.CheckValue(args, nameof(args));
            h.CheckValue(input, nameof(input));
            h.CheckUserArg(Utils.Size(args.Column) > 0, nameof(args.Column));

            var columns = new List<OneHotEncodingEstimator.ColumnInfo>();
            foreach (var column in args.Column)
            {
                var col = new OneHotEncodingEstimator.ColumnInfo(
                    column.Source ?? column.Name,
                    column.Name,
                    column.OutputKind ?? args.OutputKind,
                    column.MaxNumTerms ?? args.MaxNumTerms,
                    column.Sort ?? args.Sort,
                    column.Term ?? args.Term);
                col.SetTerms(column.Terms ?? args.Terms);
                columns.Add(col);
            }
            return new OneHotEncodingEstimator(env, columns.ToArray(), args.DataFile, args.TermsColumn, args.Loader).Fit(input).Transform(input) as IDataTransform;
        }

        private readonly TransformerChain<ITransformer> _transformer;

        public OneHotEncodingTransformer(ValueToKeyMappingEstimator term, IEstimator<ITransformer> toVector, IDataView input)
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
    public sealed class OneHotEncodingEstimator : IEstimator<OneHotEncodingTransformer>
    {
        [BestFriend]
        internal static class Defaults
        {
            public const OneHotEncodingTransformer.OutputKind OutKind = OneHotEncodingTransformer.OutputKind.Ind;
        }

        /// <summary>
        /// Describes how the transformer handles one column pair.
        /// </summary>
        public class ColumnInfo : ValueToKeyMappingTransformer.ColumnInfo
        {
            public readonly OneHotEncodingTransformer.OutputKind OutputKind;
            /// <summary>
            /// Describes how the transformer handles one column pair.
            /// </summary>
            /// <param name="input">Name of input column.</param>
            /// <param name="output">Name of the column resulting from the transformation of <paramref name="input"/>. Null means <paramref name="input"/> is replaced.</param>
            /// <param name="outputKind">Output kind: Bag (multi-set vector), Ind (indicator vector), Key (index), or Binary encoded indicator vector.</param>
            /// <param name="maxNumTerms">Maximum number of terms to keep per column when auto-training.</param>
            /// <param name="sort">How items should be ordered when vectorized. If <see cref="ValueToKeyMappingTransformer.SortOrder.Occurrence"/> choosen they will be in the order encountered.
            /// If <see cref="ValueToKeyMappingTransformer.SortOrder.Value"/>, items are sorted according to their default comparison, for example, text sorting will be case sensitive (for example, 'A' then 'Z' then 'a').</param>
            /// <param name="term">List of terms.</param>
            public ColumnInfo(string input, string output=null,
                OneHotEncodingTransformer.OutputKind outputKind = Defaults.OutKind,
                int maxNumTerms = ValueToKeyMappingEstimator.Defaults.MaxNumTerms, ValueToKeyMappingTransformer.SortOrder sort = ValueToKeyMappingEstimator.Defaults.Sort,
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
        private ValueToKeyMappingEstimator _term;

        /// Initializes an instance of the <see cref="OneHotEncodingEstimator"/>.
        /// <param name="env">Host Environment.</param>
        /// <param name="inputColumn">Name of the column to be transformed.</param>
        /// <param name="outputColumn">Name of the output column. If this is <c>null</c>, <paramref name="inputColumn"/> is used.</param>
        /// <param name="outputKind">The type of output expected.</param>
        public OneHotEncodingEstimator(IHostEnvironment env, string inputColumn,
            string outputColumn = null, OneHotEncodingTransformer.OutputKind outputKind = Defaults.OutKind)
            : this(env, new[] { new ColumnInfo(inputColumn, outputColumn ?? inputColumn, outputKind) })
        {
        }

        public OneHotEncodingEstimator(IHostEnvironment env, ColumnInfo[] columns,
            string file = null, string termsColumn = null,
            IComponentFactory<IMultiStreamSource, IDataLoader> loaderFactory = null)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(OneHotEncodingEstimator));
            _term = new ValueToKeyMappingEstimator(_host, columns, file, termsColumn, loaderFactory);
            var binaryCols = new List<(string input, string output)>();
            var cols = new List<(string input, string output, bool bag)>();
            for (int i = 0; i < columns.Length; i++)
            {
                var column = columns[i];
                OneHotEncodingTransformer.OutputKind kind = columns[i].OutputKind;
                switch (kind)
                {
                    default:
                        throw _host.ExceptUserArg(nameof(column.OutputKind));
                    case OneHotEncodingTransformer.OutputKind.Key:
                        continue;
                    case OneHotEncodingTransformer.OutputKind.Bin:
                        binaryCols.Add((column.Output, column.Output));
                        break;
                    case OneHotEncodingTransformer.OutputKind.Ind:
                        cols.Add((column.Output, column.Output, false));
                        break;
                    case OneHotEncodingTransformer.OutputKind.Bag:
                        cols.Add((column.Output, column.Output, true));
                        break;
                }
            }
            IEstimator<ITransformer> toBinVector = null;
            IEstimator<ITransformer> toVector = null;
            if (binaryCols.Count > 0)
                toBinVector = new KeyToBinaryVectorMappingEstimator(_host, binaryCols.Select(x => new KeyToBinaryVectorMappingTransformer.ColumnInfo(x.input, x.output)).ToArray());
            if (cols.Count > 0)
                toVector = new KeyToVectorMappingEstimator(_host, cols.Select(x => new KeyToVectorMappingTransformer.ColumnInfo(x.input, x.output, x.bag)).ToArray());

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

        public OneHotEncodingTransformer Fit(IDataView input) => new OneHotEncodingTransformer(_term, _toSomething, input);

        [BestFriend]
        internal void WrapTermWithDelegate(Action<ValueToKeyMappingTransformer> onFit)
        {
            _term = (ValueToKeyMappingEstimator)_term.WithOnFitDelegate(onFit);
        }
    }

    public static class Categorical
    {
        [TlcModule.EntryPoint(Name = "Transforms.CategoricalOneHotVectorizer",
            Desc = OneHotEncodingTransformer.Summary,
            UserName = OneHotEncodingTransformer.UserName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/member[@name=""CategoricalOneHotVectorizer""]/*' />",
                                 @"<include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/example[@name=""CategoricalOneHotVectorizer""]/*' />"})]
        public static CommonOutputs.TransformOutput CatTransformDict(IHostEnvironment env, OneHotEncodingTransformer.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("CatTransformDict");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var xf = OneHotEncodingTransformer.Create(host, input, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModelImpl(env, xf, input.Data), OutputData = xf };
        }

        [TlcModule.EntryPoint(Name = "Transforms.CategoricalHashOneHotVectorizer",
            Desc = OneHotHashEncoding.Summary,
            UserName = OneHotHashEncoding.UserName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/member[@name=""CategoricalHashOneHotVectorizer""]/*' />",
                                 @"<include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/example[@name=""CategoricalHashOneHotVectorizer""]/*' />"})]
        public static CommonOutputs.TransformOutput CatTransformHash(IHostEnvironment env, OneHotHashEncoding.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("CatTransformDict");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var xf = OneHotHashEncoding.Create(host, input, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModelImpl(env, xf, input.Data), OutputData = xf };
        }

        [TlcModule.EntryPoint(Name = "Transforms.TextToKeyConverter",
            Desc = ValueToKeyMappingTransformer.Summary,
            UserName = ValueToKeyMappingTransformer.FriendlyName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Data/Transforms/doc.xml' path='doc/members/member[@name=""TextToKey""]/*' />",
                                 @"<include file='../Microsoft.ML.Data/Transforms/doc.xml' path='doc/members/example[@name=""TextToKey""]/*' />" })]
        public static CommonOutputs.TransformOutput TextToKey(IHostEnvironment env, ValueToKeyMappingTransformer.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("Term");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var xf = ValueToKeyMappingTransformer.Create(host, input, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModelImpl(env, xf, input.Data), OutputData = xf };
        }

        [TlcModule.EntryPoint(Name = "Transforms.KeyToTextConverter",
            Desc = "KeyToValueTransform utilizes KeyValues metadata to map key indices to the corresponding values in the KeyValues metadata.",
            UserName = KeyToValueMappingTransformer.UserName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/doc.xml' path='doc/members/member[@name=""KeyToText""]/*' />" })]
        public static CommonOutputs.TransformOutput KeyToText(IHostEnvironment env, KeyToValueMappingTransformer.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("KeyToValue");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var xf = KeyToValueMappingTransformer.Create(host, input, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModelImpl(env, xf, input.Data), OutputData = xf };
        }
    }
}
