//// Licensed to the .NET Foundation under one or more agreements.
//// The .NET Foundation licenses this file to you under the MIT license.
//// See the LICENSE file in the project root for more information.

//using System.Text;
//using Microsoft.ML;
//using Microsoft.ML.CommandLine;
//using Microsoft.ML.Data;
//using Microsoft.ML.Internal.Utilities;
//using Microsoft.ML.Runtime;

//[assembly: LoadableClass(typeof(IDataTransform), typeof(DraculaTransform), typeof(DraculaTransform.Arguments), typeof(SignatureDataTransform),
//    "Dracula Transform", "DraculaTransform", "Dracula")]

//namespace Microsoft.ML.Transforms
//{
//    internal class DraculaEstimator : IEstimator<ITransformer>
//    {
//        /// <summary>
//        /// This is a merger of arguments for <see cref="CountTableTransformer"/> and <see cref="HashJoiningTransform"/>
//        /// </summary>
//        public sealed class Options
//        {
//            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)",
//                ShortName = "col", SortOrder = 1)]
//            public Column[] Column;

//            [Argument(ArgumentType.Multiple, HelpText = "Count table settings", ShortName = "table")]
//            public IComponentFactory<CountTableBuilderBase> CountTable = new CMCountTableBuilder.Arguments();

//            [Argument(ArgumentType.AtMostOnce, HelpText = "The coefficient with which to apply the prior smoothing to the features", ShortName = "prior")]
//            public float PriorCoefficient = 1;

//            [Argument(ArgumentType.AtMostOnce, HelpText = "Laplacian noise diversity/scale-parameter. Suggest keeping it less than 1.", ShortName = "laplace")]
//            public float LaplaceScale = 0;

//            [Argument(ArgumentType.AtMostOnce, HelpText = "Seed for the random generator for the laplacian noise.", ShortName = "seed")]
//            public int Seed = 314489979;

//            [Argument(ArgumentType.Required, HelpText = "Label column", ShortName = "label,lab", Purpose = SpecialPurpose.ColumnName)]
//            public string LabelColumn;

//            [Argument(ArgumentType.AtMostOnce, HelpText = "Optional text file to load counts from", ShortName = "extfile")]
//            public string ExternalCountsFile;

//            [Argument(ArgumentType.AtMostOnce, HelpText = "Comma-separated list of column IDs in the external count file", ShortName = "extschema")]
//            public string ExternalCountsSchema;

//            [Argument(ArgumentType.AtMostOnce, HelpText = "Keep counts for all columns in one shared count table", ShortName = "shared")]
//            public bool SharedTable = false;

//            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether the values need to be combined for a single hash", SortOrder = 3)]
//            public bool Join = true;

//            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of bits to hash into. Must be between 1 and 31, inclusive.",
//                ShortName = "bits")]
//            public int NumberOfBits = HashJoiningTransform.NumBitsLim - 1;

//            [Argument(ArgumentType.AtMostOnce, HelpText = "Hashing seed")]
//            public uint HashingSeed = 314489979;
//        }

//        public sealed class Column : OneToOneColumn
//        {
//            [Argument(ArgumentType.Multiple, HelpText = "Count table settings", ShortName = "table")]
//            public IComponentFactory<CountTableBuilderBase> CountTable;

//            [Argument(ArgumentType.AtMostOnce, HelpText = "The coefficient with which to apply the prior smoothing to the features", ShortName = "prior")]
//            public float? PriorCoefficient;

//            [Argument(ArgumentType.AtMostOnce, HelpText = "Laplacian noise diversity/scale-parameter. Suggest keeping it less than 1.", ShortName = "laplace")]
//            public float? LaplaceScale;

//            [Argument(ArgumentType.AtMostOnce, HelpText = "Seed for the random generator for the laplacian noise.", ShortName = "seed")]
//            public int? Seed;

//            // REVIEW petelu: rename to 'combine' (with 'join' as a secondary name) once possible
//            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether the values need to be combined for a single hash")]
//            public bool? Join;

//            [Argument(ArgumentType.AtMostOnce, HelpText = "Which slots should be combined together. Example: 0,3,5;0,1;3;2,1,0. Overrides 'join'.")]
//            public string CustomSlotMap;

//            public static Column Parse(string str)
//            {
//                var res = new Column();
//                if (res.TryParse(str))
//                    return res;
//                return null;
//            }
//        }

//        private readonly IHost _host;
//        private readonly Options _options;

//        public DraculaEstimator(IHostEnvironment env, Options options)
//        {
//            Contracts.CheckValue(env, nameof(env));
//            _host = env.Register(nameof(DraculaEstimator));
//            _host.CheckValue(options, nameof(options));
//            _host.CheckUserArg(Utils.Size(options.Column) > 0, nameof(options.Column), "Columns must be specified");
//            _host.CheckUserArg(!string.IsNullOrWhiteSpace(options.LabelColumn), nameof(options.LabelColumn), "Must specify the label column name");

//            _options = options;
//        }

//        public ITransformer Fit(IDataView input)
//        {
//            // creating bin mapper (HashJoinFunction)
//            var hashJoinArgs = new HashJoiningTransform.Arguments
//            {
//                NumberOfBits = _options.NumberOfBits,
//                Join = _options.Join,
//                Seed = _options.HashingSeed,
//                Ordered = false,
//            };

//            hashJoinArgs.Columns = new HashJoiningTransform.Column[_options.Column.Length];
//            for (int i = 0; i < _options.Column.Length; i++)
//            {
//                var column = _options.Column[i];
//                if (!column.TrySanitize())
//                    throw _host.ExceptUserArg(nameof(Column.Name));
//                hashJoinArgs.Columns[i] =
//                    new HashJoiningTransform.Column
//                    {
//                        Join = column.Join,
//                        CustomSlotMap = column.CustomSlotMap,
//                        Name = column.Source,
//                        Source = column.Source,
//                    };
//            }

//            IDataTransform hashJoinTransform = new HashJoiningTransform(_host, hashJoinArgs, input);

//            // creating count table transform
//            CountTableEstimator estimator;
//            if (_options.SharedTable)
//            {
//                var columns = new CountTableEstimator.SharedColumnOptions[_options.Column.Length];
//                for (int i = 0; i < _options.Column.Length; i++)
//                {
//                    var column = _options.Column[i];
//                    columns[i] = new CountTableEstimator.SharedColumnOptions(
//                        column.Name,
//                        column.Source,
//                        column.PriorCoefficient ?? _options.PriorCoefficient,
//                        column.LaplaceScale ?? _options.LaplaceScale,
//                        column.Seed ?? _options.Seed);
//                }
//                var builder = _options.CountTable;
//                _host.CheckValue(builder, nameof(_options.CountTable));
//                estimator = new CountTableEstimator(_host, _options.LabelColumn, builder.CreateComponent(_host), columns);
//            }
//            else
//            {
//                var columns = new CountTableEstimator.ColumnOptions[_options.Column.Length];
//                for (int i = 0; i < _options.Column.Length; i++)
//                {
//                    var column = _options.Column[i];
//                    var builder = column.CountTable ?? _options.CountTable;
//                    _host.CheckValue(builder, nameof(_options.CountTable));
//                    columns[i] = new CountTableEstimator.ColumnOptions(
//                        column.Name,
//                        column.Source,
//                        builder.CreateComponent(_host),
//                        column.PriorCoefficient ?? _options.PriorCoefficient,
//                        column.LaplaceScale ?? _options.LaplaceScale,
//                        column.Seed ?? _options.Seed);
//                }
//                estimator = new CountTableEstimator(_host, _options.LabelColumn, _options.ExternalCountsFile, _options.ExternalCountsSchema, columns);
//            }
//            var countTableArgs =
//                new CountTableTransformer.Options
//                {
//                    CountTable = _options.CountTable,
//                    PriorCoefficient = _options.PriorCoefficient,
//                    LaplaceScale = _options.LaplaceScale,
//                    LabelColumn = _options.LabelColumn,
//                    ExternalCountsFile = _options.ExternalCountsFile,
//                    ExternalCountsSchema = _options.ExternalCountsSchema,
//                    SharedTable = _options.SharedTable,
//                    Seed = _options.Seed,
//                    Columns = new CountTableTransformer.Column[_options.Column.Length],
//                };
//            for (int i = 0; i < _options.Column.Length; i++)
//            {
//                var column = _options.Column[i];
//                countTableArgs.Columns[i] =
//                    new CountTableTransformer.Column
//                    {
//                        Name = column.Name,
//                        Source = column.Name, // this is intentional! if args are Src:Dst, then HashJoin transforms Src:Dst and count table transforms Dst:Dst
//                        CountTable = column.CountTable,
//                        PriorCoefficient = column.PriorCoefficient,
//                        LaplaceScale = column.LaplaceScale
//                    };
//            }

//            // train count table with the output of the hash join function
//            return new CountTableEstimator(_host,  CountTableTransformer(_host, countTableArgs, hashJoinTransform);
//        }

//        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
//        {
//            throw new System.NotImplementedException();
//        }
//    }

//    public static class DraculaTransform
//    {
//        public sealed class Column : OneToOneColumn
//        {
//            [Argument(ArgumentType.Multiple, HelpText = "Count table settings", ShortName = "table")]
//            public ICountTableBuilderFactory CountTable;

//            [Argument(ArgumentType.Multiple, HelpText = "Featurizer for counts", ShortName = "feat")]
//            public ICountFeaturizerFactory Featurizer;

//            // REVIEW petelu: rename to 'combine' (with 'join' as a secondary name) once possible
//            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether the values need to be combined for a single hash")]
//            public bool? Join;

//            [Argument(ArgumentType.AtMostOnce, HelpText = "Which slots should be combined together. Example: 0,3,5;0,1;3;2,1,0. Overrides 'join'.")]
//            public string CustomSlotMap;

//            public static Column Parse(string str)
//            {
//                var res = new Column();
//                if (res.TryParse(str))
//                    return res;
//                return null;
//            }

//            public bool TryUnparse(StringBuilder sb)
//            {
//                Contracts.AssertValue(sb);
//                if (CountTable != null || Featurizer != null || Join != null || !string.IsNullOrEmpty(CustomSlotMap))
//                    return false;
//                return TryUnparseCore(sb);
//            }
//        }

//        /// <summary>
//        /// This is a merger of arguments for <see cref="CountTableTransform"/> and <see cref="HashJoinTransform"/>
//        /// </summary>
//        public sealed class Arguments
//        {
//            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)",
//                ShortName = "col", SortOrder = 1)]
//            public Column[] Column;

//            [Argument(ArgumentType.Multiple, HelpText = "Count table settings", ShortName = "table")]
//            public ICountTableBuilderFactory CountTable = new CMCountTableBuilder.Arguments();

//            [Argument(ArgumentType.Multiple, HelpText = "Featurizer for counts", ShortName = "feat")]
//            public ICountFeaturizerFactory Featurizer = new DraculaFeaturizer.Arguments();

//            [Argument(ArgumentType.Required, HelpText = "Label column", ShortName = "label,lab", SortOrder = 2, Purpose = SpecialPurpose.ColumnName)]
//            public string LabelColumn;

//            [Argument(ArgumentType.AtMostOnce, HelpText = "Optional text file to load counts from", ShortName = "extfile")]
//            public string ExternalCountsFile;

//            [Argument(ArgumentType.AtMostOnce, HelpText = "Comma-separated list of column IDs in the external count file", ShortName = "extschema")]
//            public string ExternalCountsSchema;

//            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether the values need to be combined for a single hash", SortOrder = 3)]
//            public bool Join = true;

//            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of bits to hash into. Must be between 1 and 31, inclusive.",
//                ShortName = "bits")]
//            public int HashBits = HashJoinTransform.NumBitsLim - 1;

//            [Argument(ArgumentType.AtMostOnce, HelpText = "Hashing seed")]
//            public uint Seed = 314489979;

//            [Argument(ArgumentType.AtMostOnce, HelpText = "Keep counts for all columns in one shared count table", ShortName = "shared")]
//            public bool SharedTable = false;
//        }

//        private const string RegistrationName = "DraculaTransform";

//        internal const string Summary = "Transforms the categorical column into the set of features: count of each label class, "
//            + "log-odds for each label class, back-off indicator. The columns can be of arbitrary type.";

//        /// <summary>
//        /// Initialize row function from arguments and train it (or load count tables from a file and skip training)
//        /// </summary>
//        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
//        {
//            Contracts.CheckValue(env, nameof(env));
//            var h = env.Register(RegistrationName);
//            h.CheckValue(args, nameof(args));
//            h.CheckValue(input, nameof(input));
//            h.CheckUserArg(Utils.Size(args.Column) > 0, nameof(args.Column), "Columns must be specified");
//            h.CheckUserArg(!string.IsNullOrWhiteSpace(args.LabelColumn), nameof(args.LabelColumn), "Must specify the label column name");

//            // creating bin mapper (HashJoinFunction)
//            var hashJoinArgs = new HashJoinTransform.Arguments
//            {
//                HashBits = args.HashBits,
//                Join = args.Join,
//                Seed = args.Seed,
//                Ordered = false,
//            };

//            hashJoinArgs.Column = new HashJoinTransform.Column[args.Column.Length];
//            for (int i = 0; i < args.Column.Length; i++)
//            {
//                var column = args.Column[i];
//                if (!column.TrySanitize())
//                    throw h.ExceptUserArg(nameof(Column.Name));
//                hashJoinArgs.Column[i] =
//                    new HashJoinTransform.Column
//                    {
//                        Join = column.Join,
//                        CustomSlotMap = column.CustomSlotMap,
//                        Name = column.Name,
//                        Source = column.Source,
//                    };
//            }

//            var hashJoinTransform = new HashJoinTransform(h, hashJoinArgs, input);

//            // creating count table transform
//            var countTableArgs =
//                new CountTableTransform.Arguments
//                {
//                    CountTable = args.CountTable,
//                    Featurizer = args.Featurizer,
//                    LabelColumn = args.LabelColumn,
//                    ExternalCountsFile = args.ExternalCountsFile,
//                    ExternalCountsSchema = args.ExternalCountsSchema,
//                    SharedTable = args.SharedTable,
//                    Column = new CountTableTransform.Column[args.Column.Length],
//                };
//            for (int i = 0; i < args.Column.Length; i++)
//            {
//                var column = args.Column[i];
//                countTableArgs.Column[i] =
//                    new CountTableTransform.Column
//                    {
//                        Name = column.Name,
//                        Source = column.Name, // this is intentional! if args are Src:Dst, then HashJoin transforms Src:Dst and count table transforms Dst:Dst
//                        CountTable = column.CountTable,
//                        Featurizer = column.Featurizer
//                    };
//            }

//            // train count table with the output of the hash join function
//            return new CountTableTransform(h, countTableArgs, hashJoinTransform);
//        }
//    }
//}
