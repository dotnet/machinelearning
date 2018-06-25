// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Linq;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Internal.Internallearn;

[assembly: LoadableClass(CategoricalHashTransform.Summary, typeof(IDataTransform), typeof(CategoricalHashTransform), typeof(CategoricalHashTransform.Arguments), typeof(SignatureDataTransform),
    CategoricalHashTransform.UserName, "CategoricalHashTransform", "CatHashTransform", "CategoricalHash", "CatHash")]

namespace Microsoft.ML.Runtime.Data
{
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
                string extra;
                if (!base.TryParse(str, out extra))
                    return false;
                if (extra == null)
                    return true;

                int bits;
                if (!int.TryParse(extra, out bits))
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

        /// <summary>
        /// This class is a merger of <see cref="HashTransform.Arguments"/> and <see cref="KeyToVectorTransform.Arguments"/>
        /// with join option removed
        /// </summary>
        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:hashBits:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of bits to hash into. Must be between 1 and 30, inclusive.",
                ShortName = "bits", SortOrder = 2)]
            public int HashBits = 16;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Hashing seed")]
            public uint Seed = 314489979;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether the position of each term should be included in the hash", ShortName = "ord")]
            public bool Ordered = true;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.",
                ShortName = "ih")]
            public int InvertHash;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Output kind: Bag (multi-set vector), Ind (indicator vector), or Key (index)",
                ShortName = "kind", SortOrder = 102)]
            public CategoricalTransform.OutputKind OutputKind = CategoricalTransform.OutputKind.Bag;
        }

        internal const string Summary = "Converts the categorical value into an indicator array by hashing the value and using the hash as an index in the "
            + "bag. If the input column is a vector, a single indicator bag is returned for it.";

        public const string UserName = "Categorical Hash Transform";

        public static IDataTransform Create(IHostEnvironment env, IDataView input, params string[] inputColumns)
        {
            var inputOutputColumns = new(string inputColumn, string outputColumn)[inputColumns.Length];
            for (int i = 0; i < inputColumns.Length; i++)
            {
                inputOutputColumns[i].inputColumn = inputOutputColumns[i].outputColumn = inputColumns[i];
            }
            return Create(env, input, inputOutputColumns);
        }

        public static IDataTransform Create(IHostEnvironment env, IDataView input, params (string inputColumn, string outputColumn)[] inputOutputColumns)
        {
            Column[] cols = new Column[inputOutputColumns.Length];
            for (int i = 0; i < inputOutputColumns.Length; i++)
            {
                cols[i] = new Column();
                cols[i].Source = inputOutputColumns[i].inputColumn;
                cols[i].Name = inputOutputColumns[i].outputColumn;
            }
            var args = new Arguments()
            {
                Column = cols
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
                var hashArgs = new HashTransform.Arguments
                {
                    HashBits = args.HashBits,
                    Seed = args.Seed,
                    Ordered = args.Ordered,
                    InvertHash = args.InvertHash,
                    Column = new HashTransform.Column[args.Column.Length]
                };
                for (int i = 0; i < args.Column.Length; i++)
                {
                    var column = args.Column[i];
                    if (!column.TrySanitize())
                        throw h.ExceptUserArg(nameof(Column.Name));
                    h.Assert(!string.IsNullOrWhiteSpace(column.Name));
                    h.Assert(!string.IsNullOrWhiteSpace(column.Source));
                    hashArgs.Column[i] = new HashTransform.Column
                    {
                        HashBits = column.HashBits,
                        Seed = column.Seed,
                        Ordered = column.Ordered,
                        Name = column.Name,
                        Source = column.Source,
                        InvertHash = column.InvertHash,
                    };
                }

                return CategoricalTransform.CreateTransformCore(
                    args.OutputKind,args.Column,
                    args.Column.Select(col => col.OutputKind).ToList(),
                    new HashTransform(h, hashArgs, input),
                    h,
                    env,
                    args);
            }
        }
    }
}
