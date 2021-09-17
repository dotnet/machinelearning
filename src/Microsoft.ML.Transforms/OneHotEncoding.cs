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
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(OneHotEncodingTransformer.Summary, typeof(IDataTransform), typeof(OneHotEncodingTransformer), typeof(OneHotEncodingTransformer.Options), typeof(SignatureDataTransform),
    OneHotEncodingTransformer.UserName, "CategoricalTransform", "CatTransform", "Categorical", "Cat")]

[assembly: LoadableClass(typeof(void), typeof(Categorical), null, typeof(SignatureEntryPointModule), "Categorical")]

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// <see cref="ITransformer"/> resulting from fitting a <see cref="OneHotEncodingEstimator"/>.
    /// </summary>
    public sealed class OneHotEncodingTransformer : ITransformer
    {
        internal sealed class Column : ValueToKeyMappingTransformer.ColumnBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Output kind: Bag (multi-set vector), Ind (indicator vector), Key (index), or Binary encoded indicator vector", ShortName = "kind")]
            public OneHotEncodingEstimator.OutputKind? OutputKind;

            internal static Column Parse(string str)
            {
                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            private protected override bool TryParse(string str)
            {
                Contracts.AssertNonEmpty(str);

                // We accept N:K:S where N is the new column name, K is the output kind,
                // and S is source column names.
                if (!TryParse(str, out string extra))
                    return false;
                if (extra == null)
                    return true;
                if (!Enum.TryParse(extra, true, out OneHotEncodingEstimator.OutputKind kind))
                    return false;
                OutputKind = kind;
                return true;
            }

            internal bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (OutputKind == null)
                    return TryUnparseCore(sb);
                var kind = OutputKind.Value;
                if (!Enum.IsDefined(typeof(OneHotEncodingEstimator.OutputKind), kind))
                    return false;
                string extra = OutputKind.Value.ToString();
                return TryUnparseCore(sb, extra);
            }
        }

        internal sealed class Options : ValueToKeyMappingTransformer.OptionsBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Output kind: Bag (multi-set vector), Ind (indicator vector), or Key (index)",
                ShortName = "kind", SortOrder = 102)]
            public OneHotEncodingEstimator.OutputKind OutputKind = OneHotEncodingEstimator.Defaults.OutKind;

            public Options()
            {
                // Unlike in the term transform, we want the text key values for the categorical transform
                // to default to true.
                TextKeyValues = true;
            }
        }

        internal const string Summary = "Converts the categorical value into an indicator array by building a dictionary of categories based on the "
            + "data and using the id in the dictionary as the index in the array.";

        internal const string UserName = "Categorical Transform";

        internal static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register("Categorical");
            h.CheckValue(options, nameof(options));
            h.CheckValue(input, nameof(input));
            h.CheckUserArg(Utils.Size(options.Columns) > 0, nameof(options.Columns));

            var columns = new List<OneHotEncodingEstimator.ColumnOptions>();
            foreach (var column in options.Columns)
            {
                var col = new OneHotEncodingEstimator.ColumnOptions(
                    column.Name,
                    column.Source ?? column.Name,
                    column.OutputKind ?? options.OutputKind,
                    column.MaxNumTerms ?? options.MaxNumTerms,
                    column.Sort ?? options.Sort);
                col.SetKeys(column.Terms ?? options.Terms, column.Term ?? options.Term);
                columns.Add(col);
            }
            IDataView keyData = null;
            if (!string.IsNullOrEmpty(options.DataFile))
            {
                using (var ch = h.Start("Load term data"))
                    keyData = ValueToKeyMappingTransformer.GetKeyDataViewOrNull(env, ch, options.DataFile, options.TermsColumn, options.Loader, out bool autoLoaded);
                h.AssertValue(keyData);
            }
            var transformed = new OneHotEncodingEstimator(env, columns.ToArray(), keyData).Fit(input).Transform(input);
            return (IDataTransform)transformed;
        }

        private readonly TransformerChain<ITransformer> _transformer;

        internal OneHotEncodingTransformer(ValueToKeyMappingEstimator term, IEstimator<ITransformer> toVector, IDataView input)
        {
            if (toVector != null)
                _transformer = term.Append(toVector).Fit(input);
            else
                _transformer = new TransformerChain<ITransformer>(term.Fit(input));
        }

        public DataViewSchema GetOutputSchema(DataViewSchema inputSchema) => _transformer.GetOutputSchema(inputSchema);

        public IDataView Transform(IDataView input) => _transformer.Transform(input);

        void ICanSaveModel.Save(ModelSaveContext ctx) => (_transformer as ICanSaveModel).Save(ctx);

        bool ITransformer.IsRowToRowMapper => ((ITransformer)_transformer).IsRowToRowMapper;

        IRowToRowMapper ITransformer.GetRowToRowMapper(DataViewSchema inputSchema) => ((ITransformer)_transformer).GetRowToRowMapper(inputSchema);
    }
    /// <summary>
    /// Converts one or more input columns of categorical values into as many output columns of one-hot encoded vectors.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// ###  Estimator Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Does this estimator need to look at the data to train its parameters? | Yes |
    /// | Input column data type | Vector or scalar of numeric, boolean, [text](xref:Microsoft.ML.Data.TextDataViewType), <xref:System.DateTime> or [key](xref:Microsoft.ML.Data.KeyDataViewType) type. |
    /// | Output column data type | Scalar or vector of [key](xref:Microsoft.ML.Data.KeyDataViewType), or vector of <xref:System.Single> type. |
    /// | Exportable to ONNX | Yes |
    ///
    /// The <xref:Microsoft.ML.Transforms.OneHotEncodingEstimator> builds a dictionary of unique values appearing in the input column.
    /// The resulting <xref:Microsoft.ML.Transforms.OneHotEncodingTransformer> converts one or more input columns into as many output
    /// columns of one-hot encoded vectors.
    ///
    /// The <xref:Microsoft.ML.Transforms.OneHotEncodingEstimator> is often used to convert categorical data into a form that can be
    /// provided to a machine learning algorithm.
    ///
    /// The output of this transform is specified by <xref:Microsoft.ML.Transforms.OneHotEncodingEstimator.OutputKind>:
    ///
    /// - <xref:Microsoft.ML.Transforms.OneHotEncodingEstimator.OutputKind.Indicator> produces an [indicator vector](https://en.wikipedia.org/wiki/Indicator_vector).
    /// Each slot in this vector corresponds to a category in the dictionary, so its length is the size of the built dictionary.
    /// If a value is not found in the dictioray, the output is the zero vector.
    ///
    /// - <xref:Microsoft.ML.Transforms.OneHotEncodingEstimator.OutputKind.Bag> produces one vector such that each slot stores the number
    /// of occurances of the corresponding value in the input vector.
    /// Each slot in this vector corresponds to a value in the dictionary, so its length is the size of the built dictionary.
    /// <xref:Microsoft.ML.Transforms.OneHotEncodingEstimator.OutputKind.Indicator> and <xref:Microsoft.ML.Transforms.OneHotEncodingEstimator.OutputKind.Bag>
    /// differ simply in how the bit-vectors generated from individual slots in the input column are aggregated:
    /// for Indicator they are concatenated and for Bag they are added. When the source column is a Scalar, the Indicator and Bag options are identical.
    ///
    /// - <xref:Microsoft.ML.Transforms.OneHotEncodingEstimator.OutputKind.Key> produces keys in a <xref:Microsoft.ML.Data.KeyDataViewType> column.
    /// If the input column is a vector, the output contains a vectory [key](xref:Microsoft.ML.Data.KeyDataViewType) type, where each slot of the
    /// vector corresponds to the respective slot of the input vector.
    /// If a category is not found in the bulit dictionary, it is assigned the value zero.
    ///
    /// - <xref:Microsoft.ML.Transforms.OneHotEncodingEstimator.OutputKind.Binary> produces a binary encoded vector to represent the values found in the dictionary
    /// that are present in the input column. If a value in the input column is not found in the dictionary, the output is the zero vector.
    ///
    /// The OneHotEncodingTransformer can be applied to one or more columns, in which case it builds and uses a separate dictionary
    /// for each column that it is applied to.
    ///
    /// Check the See Also section for links to usage examples.
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="CategoricalCatalog.OneHotEncoding(TransformsCatalog.CategoricalTransforms, InputOutputColumnPair[], OneHotEncodingEstimator.OutputKind, int, ValueToKeyMappingEstimator.KeyOrdinality, IDataView)"/>
    /// <seealso cref="CategoricalCatalog.OneHotEncoding(TransformsCatalog.CategoricalTransforms, string, string, OneHotEncodingEstimator.OutputKind, int, ValueToKeyMappingEstimator.KeyOrdinality, IDataView)"/>
    public sealed class OneHotEncodingEstimator : IEstimator<OneHotEncodingTransformer>
    {
        [BestFriend]
        internal static class Defaults
        {
            public const OutputKind OutKind = OutputKind.Indicator;
        }

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
            Indicator = 2,

            /// <summary>
            /// Output is a key value
            /// </summary>
            [TGUI(Label = "Output is a key value")]
            Key = 3,

            /// <summary>
            /// Output is binary encoded
            /// </summary>
            [TGUI(Label = "Output is binary encoded")]
            Binary = 4,
        }

        /// <summary>
        /// Describes how the transformer handles one column pair.
        /// </summary>
        [BestFriend]
        internal sealed class ColumnOptions : ValueToKeyMappingEstimator.ColumnOptionsBase
        {
            public readonly OutputKind OutputKind;
            /// <summary>
            /// Describes how the transformer handles one column pair.
            /// </summary>
            /// <param name="name">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
            /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="name"/> will be used as source.</param>
            /// <param name="outputKind">Output kind: Bag (multi-set vector), Ind (indicator vector), Key (index), or Binary encoded indicator vector.</param>
            /// <param name="maximumNumberOfKeys">Maximum number of terms to keep per column when auto-training.</param>
            /// <param name="keyOrdinality">How items should be ordered when vectorized. If <see cref="ValueToKeyMappingEstimator.KeyOrdinality.ByOccurrence"/> choosen they will be in the order encountered.
            /// If <see cref="ValueToKeyMappingEstimator.KeyOrdinality.ByValue"/>, items are sorted according to their default comparison, for example, text sorting will be case sensitive (for example, 'A' then 'Z' then 'a').</param>
            public ColumnOptions(string name, string inputColumnName = null,
                OutputKind outputKind = Defaults.OutKind,
                int maximumNumberOfKeys = ValueToKeyMappingEstimator.Defaults.MaximumNumberOfKeys, ValueToKeyMappingEstimator.KeyOrdinality keyOrdinality = ValueToKeyMappingEstimator.Defaults.Ordinality)
                : base(name, inputColumnName ?? name, maximumNumberOfKeys, keyOrdinality, true)
            {
                OutputKind = outputKind;
            }

            internal void SetKeys(string[] keys, string key)
            {
                Keys = keys;
                Key = key;
            }

        }

        private readonly IHost _host;
        private readonly IEstimator<ITransformer> _toSomething;
        private ValueToKeyMappingEstimator _term;

        /// Initializes an instance of the <see cref="OneHotEncodingEstimator"/>.
        /// <param name="env">Host Environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="outputKind">The type of output expected.</param>
        internal OneHotEncodingEstimator(IHostEnvironment env, string outputColumnName, string inputColumnName = null,
            OutputKind outputKind = Defaults.OutKind)
            : this(env, new[] { new ColumnOptions(outputColumnName, inputColumnName ?? outputColumnName, outputKind) })
        {
        }

        internal OneHotEncodingEstimator(IHostEnvironment env, ColumnOptions[] columns, IDataView keyData = null)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(OneHotEncodingEstimator));
            _term = new ValueToKeyMappingEstimator(_host, columns, keyData);
            var binaryCols = new List<(string outputColumnName, string inputColumnName)>();
            var cols = new List<(string outputColumnName, string inputColumnName, bool bag)>();
            for (int i = 0; i < columns.Length; i++)
            {
                var column = columns[i];
                OutputKind kind = columns[i].OutputKind;
                switch (kind)
                {
                    default:
                        throw _host.ExceptUserArg(nameof(column.OutputKind));
                    case OutputKind.Key:
                        continue;
                    case OutputKind.Binary:
                        binaryCols.Add((column.OutputColumnName, column.OutputColumnName));
                        break;
                    case OutputKind.Indicator:
                        cols.Add((column.OutputColumnName, column.OutputColumnName, false));
                        break;
                    case OutputKind.Bag:
                        cols.Add((column.OutputColumnName, column.OutputColumnName, true));
                        break;
                }
            }
            IEstimator<ITransformer> toBinVector = null;
            IEstimator<ITransformer> toVector = null;
            if (binaryCols.Count > 0)
                toBinVector = new KeyToBinaryVectorMappingEstimator(_host, binaryCols.Select(x => (x.outputColumnName, x.inputColumnName)).ToArray());
            if (cols.Count > 0)
                toVector = new KeyToVectorMappingEstimator(_host, cols.Select(x => new KeyToVectorMappingEstimator.ColumnOptions(x.outputColumnName, x.inputColumnName, x.bag)).ToArray());

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

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            if (_toSomething != null)
                return _term.Append(_toSomething).GetOutputSchema(inputSchema);
            else
                return _term.GetOutputSchema(inputSchema);
        }

        /// <summary>
        /// Trains and returns a <see cref="OneHotEncodingTransformer"/>.
        /// </summary>
        public OneHotEncodingTransformer Fit(IDataView input) => new OneHotEncodingTransformer(_term, _toSomething, input);

        [BestFriend]
        internal void WrapTermWithDelegate(Action<ValueToKeyMappingTransformer> onFit)
        {
            _term = (ValueToKeyMappingEstimator)_term.WithOnFitDelegate(onFit);
        }
    }

    internal static class Categorical
    {
        [TlcModule.EntryPoint(Name = "Transforms.CategoricalOneHotVectorizer",
            Desc = OneHotEncodingTransformer.Summary,
            UserName = OneHotEncodingTransformer.UserName)]
        public static CommonOutputs.TransformOutput CatTransformDict(IHostEnvironment env, OneHotEncodingTransformer.Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("CatTransformDict");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var xf = OneHotEncodingTransformer.Create(host, input, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModelImpl(env, xf, input.Data), OutputData = xf };
        }

        [TlcModule.EntryPoint(Name = "Transforms.CategoricalHashOneHotVectorizer",
            Desc = OneHotHashEncodingTransformer.Summary,
            UserName = OneHotHashEncodingTransformer.UserName)]
        public static CommonOutputs.TransformOutput CatTransformHash(IHostEnvironment env, OneHotHashEncodingTransformer.Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("CatTransformDict");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var xf = OneHotHashEncodingTransformer.Create(host, input, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModelImpl(env, xf, input.Data), OutputData = xf };
        }

        [TlcModule.EntryPoint(Name = "Transforms.TextToKeyConverter",
            Desc = ValueToKeyMappingTransformer.Summary,
            UserName = ValueToKeyMappingTransformer.FriendlyName)]
        public static CommonOutputs.TransformOutput TextToKey(IHostEnvironment env, ValueToKeyMappingTransformer.Options input)
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
            UserName = KeyToValueMappingTransformer.UserName)]
        public static CommonOutputs.TransformOutput KeyToText(IHostEnvironment env, KeyToValueMappingTransformer.Options input)
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
