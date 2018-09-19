// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(CategoricalHashTransform.Summary, typeof(IDataTransform), typeof(CategoricalHashTransform), typeof(CategoricalHashTransform.Arguments), typeof(SignatureDataTransform),
    CategoricalHashTransform.UserName, "CategoricalHashTransform", "CatHashTransform", "CategoricalHash", "CatHash")]

namespace Microsoft.ML.Runtime.Data
{
    /// <include file='doc.xml' path='doc/members/member[@name="CategoricalHashOneHotVectorizer"]/*' />
    public static class CategoricalHashTransform
    {
        public const int NumBitsLim = 31; // can't convert 31-bit hashes to indicator vectors, so max is 30

        public sealed class Column : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce,
                HelpText = "The number of bits to hash into. Must be between 1 and 30, inclusive.",
                ShortName = "bits")]
            public int? HashBits;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Hashing seed")]
            public uint? Seed;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether the position of each term should be included in the hash", ShortName = "ord")]
            public bool? Ordered;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.",
                ShortName = "ih")]
            public int? InvertHash;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Output kind: Bag (multi-set vector), Ind (indicator vector), or Key (index)",
                ShortName = "kind", SortOrder = 102)]
            public CategoricalTransform.OutputKind? OutputKind;

            public static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            protected override bool TryParse(string str)
            {
                Contracts.AssertNonEmpty(str);

                // We accept N:B:S where N is the new column name, B is the number of bits,
                // and S is source column names.
                if (!TryParse(str, out string extra))
                    return false;
                if (extra == null)
                    return true;
                if (!int.TryParse(extra, out int bits))
                    return false;
                HashBits = bits;
                return true;
            }

            public bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (Seed != null || Ordered != null || InvertHash != null)
                    return false;
                if (HashBits == null)
                    return TryUnparseCore(sb);
                string extra = HashBits.Value.ToString();
                return TryUnparseCore(sb, extra);
            }
        }

        private static class Defaults
        {
            public const int HashBits = 16;
            public const uint Seed = 314489979;
            public const bool Ordered = true;
            public const int InvertHash = 0;
            public const CategoricalTransform.OutputKind OutputKind = CategoricalTransform.OutputKind.Bag;
        }

        /// <summary>
        /// This class is a merger of <see cref="HashConverterTransformer.Arguments"/> and <see cref="KeyToVectorTransform.Arguments"/>
        /// with join option removed
        /// </summary>
        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:hashBits:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of bits to hash into. Must be between 1 and 30, inclusive.",
                ShortName = "bits", SortOrder = 2)]
            public int HashBits = Defaults.HashBits;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Hashing seed")]
            public uint Seed = Defaults.Seed;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether the position of each term should be included in the hash", ShortName = "ord")]
            public bool Ordered = Defaults.Ordered;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.",
                ShortName = "ih")]
            public int InvertHash = Defaults.InvertHash;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Output kind: Bag (multi-set vector), Ind (indicator vector), or Key (index)",
                ShortName = "kind", SortOrder = 102)]
            public CategoricalTransform.OutputKind OutputKind = Defaults.OutputKind;
        }

        internal const string Summary = "Converts the categorical value into an indicator array by hashing the value and using the hash as an index in the "
            + "bag. If the input column is a vector, a single indicator bag is returned for it.";

        public const string UserName = "Categorical Hash Transform";

        /// <summary>
        /// A helper method to create <see cref="CategoricalHashTransform"/> for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="name">Name of the output column.</param>
        /// <param name="source">Name of the column to be transformed. If this is null '<paramref name="name"/>' will be used.</param>
        /// <param name="hashBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="invertHash">Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.</param>
        /// <param name="outputKind">The type of output expected.</param>
        public static IDataTransform Create(IHostEnvironment env,
            IDataView input,
            string name,
            string source = null,
            int hashBits = Defaults.HashBits,
            int invertHash = Defaults.InvertHash,
            CategoricalTransform.OutputKind outputKind = Defaults.OutputKind)
        {
            var args = new Arguments()
            {
                Column = new[] { new Column(){
                    Source = source ?? name,
                    Name = name
                    }
                },
                HashBits = hashBits,
                InvertHash = invertHash,
                OutputKind = outputKind
            };
            return Create(env, args, input);
        }

        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register("CategoricalHash");
            using (var ch = h.Start("CategoricalHash"))
            {
                h.CheckValue(args, nameof(args));
                h.CheckValue(input, nameof(input));
                h.CheckUserArg(Utils.Size(args.Column) > 0, nameof(args.Column), "Columns must be specified");
                if (args.HashBits < 1 || args.HashBits >= NumBitsLim)
                    throw h.ExceptUserArg(nameof(args.HashBits), "Number of bits must be between 1 and {0}", NumBitsLim - 1);

                // creating the Hash function
                var hashArgs = new HashConverterTransformer.Arguments
                {
                    HashBits = args.HashBits,
                    Seed = args.Seed,
                    Ordered = args.Ordered,
                    InvertHash = args.InvertHash,
                    Column = new HashConverterTransformer.Column[args.Column.Length]
                };
                for (int i = 0; i < args.Column.Length; i++)
                {
                    var column = args.Column[i];
                    if (!column.TrySanitize())
                        throw h.ExceptUserArg(nameof(Column.Name));
                    h.Assert(!string.IsNullOrWhiteSpace(column.Name));
                    h.Assert(!string.IsNullOrWhiteSpace(column.Source));
                    hashArgs.Column[i] = new HashConverterTransformer.Column
                    {
                        HashBits = column.HashBits,
                        Seed = column.Seed,
                        Ordered = column.Ordered,
                        Name = column.Name,
                        Source = column.Source,
                        InvertHash = column.InvertHash,
                    };
                }

                return CreateTransformCore(
                    args.OutputKind, args.Column,
                    args.Column.Select(col => col.OutputKind).ToList(),
                    HashConverterTransformer.Create(h, hashArgs, input),
                    h,
                    args);
            }
        }

        private static IDataTransform CreateTransformCore(CategoricalTransform.OutputKind argsOutputKind, OneToOneColumn[] columns,
            List<CategoricalTransform.OutputKind?> columnOutputKinds, IDataTransform input, IHost h, Arguments catHashArgs = null)
        {
            Contracts.CheckValue(columns, nameof(columns));
            Contracts.CheckValue(columnOutputKinds, nameof(columnOutputKinds));
            Contracts.CheckParam(columns.Length == columnOutputKinds.Count, nameof(columns));

            using (var ch = h.Start("Create Transform Core"))
            {
                // Create the KeyToVectorTransform, if needed.
                var cols = new List<KeyToVectorTransform.Column>();
                bool binaryEncoding = argsOutputKind == CategoricalTransform.OutputKind.Bin;
                for (int i = 0; i < columns.Length; i++)
                {
                    var column = columns[i];
                    if (!column.TrySanitize())
                        throw h.ExceptUserArg(nameof(Column.Name));

                    bool? bag;
                    CategoricalTransform.OutputKind kind = columnOutputKinds[i] ?? argsOutputKind;
                    switch (kind)
                    {
                        default:
                            throw ch.ExceptUserArg(nameof(Column.OutputKind));
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

                    var keyToBinaryVecCols = cols.Select(x => new KeyToBinaryVectorTransform.ColumnInfo(x.Source, x.Name)).ToArray();
                    transform = KeyToBinaryVectorTransform.Create(h, input, keyToBinaryVecCols);
                }
                else
                {
                    var keyToVecCols = cols.Select(x => new KeyToVectorTransform.ColumnInfo(x.Source, x.Name, x.Bag ?? argsOutputKind == CategoricalTransform.OutputKind.Bag)).ToArray();

                    transform = KeyToVectorTransform.Create(h, input, keyToVecCols);
                }

                ch.Done();
                return transform;
            }
        }
    }
}
