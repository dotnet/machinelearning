// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data.StaticPipe.Runtime;

[assembly: LoadableClass(CategoricalTransform.Summary, typeof(IDataTransform), typeof(CategoricalTransform), typeof(CategoricalTransform.Arguments), typeof(SignatureDataTransform),
    CategoricalTransform.UserName, "CategoricalTransform", "CatTransform", "Categorical", "Cat")]

[assembly: LoadableClass(typeof(void), typeof(Categorical), null, typeof(SignatureEntryPointModule), "Categorical")]
namespace Microsoft.ML.Runtime.Data
{
    /// <include file='doc.xml' path='doc/members/member[@name="CategoricalOneHotVectorizer"]/*' />
    public static class CategoricalTransform
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
                string extra;
                if (!base.TryParse(str, out extra))
                    return false;
                if (extra == null)
                    return true;

                OutputKind kind;
                if (!Enum.TryParse(extra, true, out kind))
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
                col.SetTerms(column.Terms);
                columns.Add(col);
            }
            return Create(env, input, columns.ToArray()) as IDataTransform;
        }

        public static IDataView Create(IHostEnvironment env, IDataView input, params CategoricalEstimator.ColumnInfo[] columns)
        {
            return new CategoricalEstimator(env, columns).Fit(input).Transform(input);
        }
    }

    public sealed class CategoricalEstimator : IEstimator<ITransformer>
    {
        public static class Defaults
        {
            public const CategoricalTransform.OutputKind OutKind = CategoricalTransform.OutputKind.Ind;
        }

        public class ColumnInfo : TermTransform.ColumnInfo
        {
            public readonly CategoricalTransform.OutputKind OutputKind;
            public ColumnInfo(string input, string output, CategoricalTransform.OutputKind outputKind = Defaults.OutKind, int maxNumTerms = TermEstimator.Defaults.MaxNumTerms, TermTransform.SortOrder sort = TermEstimator.Defaults.Sort, string[] term = null)
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
        private readonly ColumnInfo[] _columns;
        private readonly IEstimator<ITransformer> _estimatorChain;

        public CategoricalEstimator(IHostEnvironment env, IDataView input, string name, string source = null, CategoricalTransform.OutputKind outputKind = Defaults.OutKind)
            : this(env, new ColumnInfo(source ?? name, name, outputKind))
        {
        }

        public CategoricalEstimator(IHostEnvironment env, params ColumnInfo[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(TermEstimator));
            _columns = columns.ToArray();
            var termEst = new TermEstimator(_host, _columns);

            var cols = new List<KeyToVectorTransform.ColumnInfo>();
            bool binaryEncoding = false;
            for (int i = 0; i < _columns.Length; i++)
            {
                var column = _columns[i];

                bool bag;

                CategoricalTransform.OutputKind kind = _columns[i].OutputKind;
                switch (kind)
                {
                    default:
                        throw _host.ExceptUserArg(nameof(column.OutputKind));
                    case CategoricalTransform.OutputKind.Key:
                        continue;
                    case CategoricalTransform.OutputKind.Bin:
                        binaryEncoding = true;
                        bag = false;
                        break;
                    case CategoricalTransform.OutputKind.Ind:
                        bag = false;
                        break;
                    case CategoricalTransform.OutputKind.Bag:
                        bag = true;
                        break;
                }
                var col = new KeyToVectorTransform.ColumnInfo(column.Output, column.Output, bag);
                cols.Add(col);

                if (binaryEncoding)
                {
                    var keyToBinEst = new KeyToBinaryVectorEstimator(_host, cols.Select(x => new KeyToBinaryVectorTransform.ColumnInfo(x.Input, x.Output)).ToArray());
                    _estimatorChain = termEst.Append(keyToBinEst);
                }
                else
                {
                    var keyToVecEst = new KeyToVectorEstimator(_host, cols.Select(x => new KeyToVectorTransform.ColumnInfo(x.Input, x.Output, x.Bag)).ToArray());
                    _estimatorChain = termEst.Append(keyToVecEst);
                }
            }
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            return _estimatorChain.GetOutputSchema(inputSchema);
        }

        public ITransformer Fit(IDataView input)
        {
            return _estimatorChain.Fit(input);
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
            UserName = TermTransform.UserName,
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

    public enum OneHotOutputKind : byte
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
        /// Output is a key value
        /// </summary>
        Key = 3,

        /// <summary>
        /// Output is binary encoded
        /// </summary>
        Bin = 4,
    }

    public static partial class CategoricalStaticExtensions
    {
        // I am not certain I see a good way to cover the distinct types beyond complete enumeration.
        // Raw generics would allow illegal possible inputs, e.g., Scalar<Bitmap>. So, this is a partial
        // class, and all the public facing extension methods for each possible type are in a T4 generated result.

        private const KeyValueOrder DefSort = (KeyValueOrder)TermEstimator.Defaults.Sort;
        private const int DefMax = TermEstimator.Defaults.MaxNumTerms;
        private const OneHotOutputKind DefOut = (OneHotOutputKind)CategoricalEstimator.Defaults.OutKind;

        private struct Config
        {
            public readonly KeyValueOrder Order;
            public readonly int Max;
            public readonly OneHotOutputKind OutputKind;

            public Config(KeyValueOrder order, int max, OneHotOutputKind outputKind)
            {
                Order = order;
                Max = max;
                OutputKind = outputKind;
            }
        }

        private interface IOneHotCol
        {
            PipelineColumn Input { get; }
            Config Config { get; }
        }

        private sealed class ImplScalar<T> :Vector<float>, IOneHotCol
        {
            public PipelineColumn Input { get; }
            public Config Config { get; }
            public ImplScalar(PipelineColumn input, Config config) : base(Rec.Inst, input)
            {
                Input = input;
                Config = config;
            }
        }

        private sealed class ImplVector<T> : Vector<float>, IOneHotCol
        {
            public PipelineColumn Input { get; }
            public Config Config { get; }
            public ImplVector(PipelineColumn input, Config config) : base(Rec.Inst, input)
            {
                Input = input;
                Config = config;
            }
        }

        private sealed class ImplVarVector<T> : VarVector<float>, IOneHotCol
        {
            public PipelineColumn Input { get; }
            public Config Config { get; }
            public ImplVarVector(PipelineColumn input, Config config) : base(Rec.Inst, input)
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
                for (int i = 0; i < toOutput.Length; ++i)
                {
                    var tcol = (IOneHotCol)toOutput[i];
                    infos[i] = new CategoricalEstimator.ColumnInfo(inputNames[tcol.Input], outputNames[toOutput[i]], (CategoricalTransform.OutputKind)tcol.Config.OutputKind,
                        tcol.Config.Max, (TermTransform.SortOrder)tcol.Config.Order);
                }
                return new CategoricalEstimator(env, infos);
            }
        }
    }
}
