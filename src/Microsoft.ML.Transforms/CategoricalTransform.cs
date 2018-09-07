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

        private static class Defaults
        {
            public const OutputKind OutKind = OutputKind.Ind;
        }

        public sealed class Arguments : TermTransform.ArgumentsBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Output kind: Bag (multi-set vector), Ind (indicator vector), or Key (index)",
                ShortName = "kind", SortOrder = 102)]
            public OutputKind OutputKind = Defaults.OutKind;

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

        /// <summary>
        /// A helper method to create <see cref="CategoricalTransform"/> for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="name">Name of the output column.</param>
        /// <param name="source">Name of the column to be transformed. If this is null '<paramref name="name"/>' will be used.</param>
        /// <param name="outputKind">The type of output expected.</param>
        public static IDataTransform Create(IHostEnvironment env, IDataView input, string name, string source = null, OutputKind outputKind = Defaults.OutKind)
        {
            var args = new Arguments()
            {
                Column = new[] { new Column(){
                        Source = source ?? name,
                        Name = name
                    }
                },
                OutputKind = outputKind
            };
            return Create(env, args, input);
        }

        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register("Categorical");
            h.CheckValue(args, nameof(args));
            h.CheckValue(input, nameof(input));
            h.CheckUserArg(Utils.Size(args.Column) > 0, nameof(args.Column));
            return CreateTransformCore(
                args.OutputKind,
                args.Column,
                args.Column.Select(col => col.OutputKind).ToList(),
                 TermTransform.Create(h, args, args.Column, input),
                h,
                env);
        }

        public static IDataTransform CreateTransformCore(
            OutputKind argsOutputKind,
            OneToOneColumn[] columns,
            List<OutputKind?> columnOutputKinds,
            IDataTransform input,
            IHost h,
            IHostEnvironment env,
            CategoricalHashTransform.Arguments catHashArgs = null)
        {
            Contracts.CheckValue(columns, nameof(columns));
            Contracts.CheckValue(columnOutputKinds, nameof(columnOutputKinds));
            Contracts.CheckParam(columns.Length == columnOutputKinds.Count, nameof(columns));

            using (var ch = h.Start("Create Transform Core"))
            {
                // Create the KeyToVectorTransform, if needed.
                List<KeyToVectorTransform.Column> cols = new List<KeyToVectorTransform.Column>();
                bool binaryEncoding = argsOutputKind == OutputKind.Bin;
                for (int i = 0; i < columns.Length; i++)
                {
                    var column = columns[i];
                    if (!column.TrySanitize())
                        throw h.ExceptUserArg(nameof(Column.Name));

                    bool? bag;
                    OutputKind kind = columnOutputKinds[i].HasValue ? columnOutputKinds[i].Value : argsOutputKind;
                    switch (kind)
                    {
                        default:
                            throw env.ExceptUserArg(nameof(Column.OutputKind));
                        case OutputKind.Key:
                            continue;
                        case OutputKind.Bin:
                            binaryEncoding = true;
                            bag = false;
                            break;
                        case OutputKind.Ind:
                            bag = false;
                            break;
                        case OutputKind.Bag:
                            bag = true;
                            break;
                    }
                    var col = new KeyToVectorTransform.Column();
                    col.Name = column.Name;
                    col.Source = column.Name;
                    col.Bag = bag;
                    cols.Add(col);
                }

                if (cols.Count == 0)
                    return input;

                IDataTransform transform;
                if (binaryEncoding)
                {
                    if ((catHashArgs?.InvertHash ?? 0) != 0)
                        ch.Warning("Invert hashing is being used with binary encoding.");

                    var keyToBinaryArgs = new KeyToBinaryVectorTransform.Arguments();
                    keyToBinaryArgs.Column = cols.ToArray();
                    transform = new KeyToBinaryVectorTransform(h, keyToBinaryArgs, input);
                }
                else
                {
                    var keyToVecArgs = new KeyToVectorTransform.Arguments
                    {
                        Bag = argsOutputKind == OutputKind.Bag,
                        Column = cols.ToArray()
                    };

                    transform = new KeyToVectorTransform(h, keyToVecArgs, input);
                }

                ch.Done();
                return transform;
            }
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
}
