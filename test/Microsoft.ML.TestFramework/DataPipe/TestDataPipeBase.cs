// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.TestFramework;
using Xunit;

namespace Microsoft.ML.Runtime.RunTests
{
    public abstract partial class TestDataPipeBase : TestDataViewBase
    {
        public const string IrisDataPath = "iris.data";

        protected static TextLoader.Arguments MakeIrisTextLoaderArgs()
        {
            return new TextLoader.Arguments()
            {
                Separator = "comma",
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("SepalLength", DataKind.R4, 0),
                    new TextLoader.Column("SepalWidth", DataKind.R4, 1),
                    new TextLoader.Column("PetalLength", DataKind.R4, 2),
                    new TextLoader.Column("PetalWidth",DataKind.R4, 3),
                    new TextLoader.Column("Label", DataKind.Text, 4)
                }
            };
        }

        /// <summary>
        /// 'Workout test' for an estimator.
        /// Checks the following traits:
        /// - the estimator is applicable to the validFitInput and validForFitNotValidForTransformInput, and not applicable to validTransformInput and invalidInput;
        /// - the fitted transformer is applicable to validFitInput and validTransformInput, and not applicable to invalidInput and validForFitNotValidForTransformInput;
        /// - fitted transformer can be saved and re-loaded into the transformer with the same behavior.
        /// - schema propagation for fitted transformer conforms to schema propagation of estimator.
        /// </summary>
        protected void TestEstimatorCore(IEstimator<ITransformer> estimator,
            IDataView validFitInput, IDataView validTransformInput = null, IDataView invalidInput = null, IDataView validForFitNotValidForTransformInput = null)
        {
            Contracts.AssertValue(estimator);
            Contracts.AssertValue(validFitInput);
            Contracts.AssertValueOrNull(validTransformInput);
            Contracts.AssertValueOrNull(invalidInput);
            Action<Action> mustFail = (Action action) =>
            {
                try
                {
                    action();
                    Assert.False(true);
                }
                catch (ArgumentOutOfRangeException) { }
                catch (InvalidOperationException) { }
                catch (TargetInvocationException ex)
                {
                    Exception e;
                    for (e = ex; e.InnerException != null; e = e.InnerException)
                    {
                    }
                    Assert.True(e is ArgumentOutOfRangeException || e is InvalidOperationException);
                    Assert.True(e.IsMarked());
                }
            };

            // Schema propagation tests for estimator.
            var outSchemaShape = estimator.GetOutputSchema(SchemaShape.Create(validFitInput.Schema));
            if (validTransformInput != null)
            {
                mustFail(() => estimator.GetOutputSchema(SchemaShape.Create(validTransformInput.Schema)));
                mustFail(() => estimator.Fit(validTransformInput));
            }

            if (invalidInput != null)
            {
                mustFail(() => estimator.GetOutputSchema(SchemaShape.Create(invalidInput.Schema)));
                mustFail(() => estimator.Fit(invalidInput));
            }

            if (validForFitNotValidForTransformInput != null)
            {
                estimator.GetOutputSchema(SchemaShape.Create(validForFitNotValidForTransformInput.Schema));
                estimator.Fit(validForFitNotValidForTransformInput);
            }

            var transformer = estimator.Fit(validFitInput);
            // Save and reload.
            string modelPath = GetOutputPath(FullTestName + "-model.zip");
            using (var fs = File.Create(modelPath))
                transformer.SaveTo(Env, fs);

            ITransformer loadedTransformer;
            using (var fs = File.OpenRead(modelPath))
                loadedTransformer = TransformerChain.LoadFrom(Env, fs);
            DeleteOutputPath(modelPath);

            // Run on train data.
            Action<IDataView> checkOnData = (IDataView data) =>
            {
                var schema = transformer.GetOutputSchema(data.Schema);

                // If it's a row to row mapper, then the output schema should be the same.
                if (transformer.IsRowToRowMapper)
                {
                    var mapper = transformer.GetRowToRowMapper(data.Schema);
                    Check(mapper.InputSchema == data.Schema, "InputSchemas were not identical to actual input schema");
                    CheckSameSchemas(schema, mapper.Schema);
                }
                else
                {
                    mustFail(() => transformer.GetRowToRowMapper(data.Schema));
                }

                // Loaded transformer needs to have the same schema propagation.
                CheckSameSchemas(schema, loadedTransformer.GetOutputSchema(data.Schema));

                var scoredTrain = transformer.Transform(data);
                var scoredTrain2 = loadedTransformer.Transform(data);

                // The schema of the transformed data must match the schema provided by schema propagation.
                CheckSameSchemas(schema, scoredTrain.Schema);

                // The schema and data of scored dataset must be identical between loaded
                // and original transformer.
                // This in turn means that the schema of loaded transformer matches for 
                // Transform and GetOutputSchema calls.
                CheckSameSchemas(scoredTrain.Schema, scoredTrain2.Schema);
                CheckSameValues(scoredTrain, scoredTrain2, exactDoubles: false);
            };

            checkOnData(validFitInput);

            if (validTransformInput != null)
                checkOnData(validTransformInput);

            if (invalidInput != null)
            {
                mustFail(() => transformer.GetOutputSchema(invalidInput.Schema));
                mustFail(() => transformer.Transform(invalidInput));
                mustFail(() => loadedTransformer.GetOutputSchema(invalidInput.Schema));
                mustFail(() => loadedTransformer.Transform(invalidInput));
            }
            if (validForFitNotValidForTransformInput != null)
            {
                mustFail(() => transformer.GetOutputSchema(validForFitNotValidForTransformInput.Schema));
                mustFail(() => transformer.Transform(validForFitNotValidForTransformInput));
                mustFail(() => loadedTransformer.GetOutputSchema(validForFitNotValidForTransformInput.Schema));
                mustFail(() => loadedTransformer.Transform(validForFitNotValidForTransformInput));
            }

            // Schema verification between estimator and transformer.
            var scoredTrainSchemaShape = SchemaShape.Create(transformer.GetOutputSchema(validFitInput.Schema));
            CheckSameSchemaShape(outSchemaShape, scoredTrainSchemaShape);
        }

        private void CheckSameSchemaShape(SchemaShape promised, SchemaShape delivered)
        {
            Assert.True(promised.Columns.Length == delivered.Columns.Length);
            var sortedCols1 = promised.Columns.OrderBy(x => x.Name);
            var sortedCols2 = delivered.Columns.OrderBy(x => x.Name);

            foreach (var (x, y) in sortedCols1.Zip(sortedCols2, (x, y) => (x, y)))
            {
                Assert.Equal(x.Name, y.Name);
                // We want the 'promised' metadata to be a superset of 'delivered'.
                Assert.True(y.IsCompatibleWith(x), $"Mismatch on {x.Name}");
            }
        }

        // REVIEW: incorporate the testing for re-apply logic here?
        /// <summary>
        /// Create PipeDataLoader from the given args, save it, re-load it, verify that the data of
        /// the loaded pipe matches the original.
        /// * pathData defaults to breast-cancer.txt.
        /// * actLoader is invoked for extra validation (if non-null).
        /// </summary>
        protected IDataLoader TestCore(string pathData, bool keepHidden, string[] argsPipe,
            Action<IDataLoader> actLoader = null, string suffix = "", string suffixBase = null, bool checkBaseline = true,
            bool forceDense = false, bool logCurs = false, ConsoleEnvironment env = null, bool roundTripText = true,
            bool checkTranspose = false, bool checkId = true, bool baselineSchema = true)
        {
            Contracts.AssertValue(Env);
            if (env == null)
                env = Env;

            MultiFileSource files;
            IDataLoader compositeLoader;
            var pipe1 = compositeLoader = CreatePipeDataLoader(env, pathData, argsPipe, out files);

            if (actLoader != null)
                actLoader(compositeLoader);

            // Re-apply pipe to the loader and check equality.
            var comp = compositeLoader as CompositeDataLoader;
            IDataView srcLoader = null;
            if (comp != null)
            {
                srcLoader = comp.View;
                while (srcLoader is IDataTransform)
                    srcLoader = ((IDataTransform)srcLoader).Source;
                var reappliedPipe = ApplyTransformUtils.ApplyAllTransformsToData(env, comp.View, srcLoader);
                if (!CheckMetadataTypes(reappliedPipe.Schema))
                    Failed();

                if (!CheckSameSchemas(pipe1.Schema, reappliedPipe.Schema))
                    Failed();
                else if (!CheckSameValues(pipe1, reappliedPipe, checkId: checkId))
                    Failed();
            }

            if (logCurs)
            {
                string name = TestName + suffix + "-CursLog" + ".txt";
                string pathLog = DeleteOutputPath("SavePipe", name);

                using (var writer = OpenWriter(pathLog))
                using (env.RedirectChannelOutput(writer, writer))
                {
                    long count = 0;
                    // Set the concurrency to 1 for this; restore later.
                    int conc = env.ConcurrencyFactor;
                    env.ConcurrencyFactor = 1;
                    using (var curs = pipe1.GetRowCursor(c => true, null))
                    {
                        while (curs.MoveNext())
                        {
                            count++;
                        }
                    }
                    writer.WriteLine("Cursored through {0} rows", count);
                    env.ConcurrencyFactor = conc;
                }

                CheckEqualityNormalized("SavePipe", name);
            }

            var pathModel = SavePipe(pipe1, suffix);
            var pipe2 = LoadPipe(pathModel, env, files);
            if (!CheckMetadataTypes(pipe2.Schema))
                Failed();

            if (!CheckSameSchemas(pipe1.Schema, pipe2.Schema))
                Failed();
            else if (!CheckSameValues(pipe1, pipe2, checkId: checkId))
                Failed();

            if (pipe1.Schema.ColumnCount > 0)
            {
                // The text saver fails if there are no columns, so we cannot check in that case.
                if (!SaveLoadText(pipe1, env, keepHidden, suffix, suffixBase, checkBaseline, forceDense, roundTripText))
                    Failed();
                // The transpose saver likewise fails for the same reason.
                if (checkTranspose && !SaveLoadTransposed(pipe1, env, suffix))
                    Failed();
            }
            if (!SaveLoad(pipe1, env, suffix))
                Failed();

            // Check that the pipe doesn't shuffle when it cannot :).
            if (srcLoader != null)
            {
                // First we need to cache the data so it can be shuffled.
                var cachedData = new CacheDataView(env, srcLoader, null);
                var newPipe = ApplyTransformUtils.ApplyAllTransformsToData(env, comp.View, cachedData);
                if (!newPipe.CanShuffle)
                {
                    using (var c1 = newPipe.GetRowCursor(col => true, new SysRandom(123)))
                    using (var c2 = newPipe.GetRowCursor(col => true))
                    {
                        if (!CheckSameValues(c1, c2, true, true, true))
                            Failed();
                    }
                }

                // Join all filler threads of CacheDataView prior to the disposal of _wrt. 
                // Otherwise it may writes to a closed stream.
                cachedData.Wait();
            }

            // Baseline the schema, including metadata.
            if (baselineSchema)
            {
                string name = TestName + suffix + "-Schema" + ".txt";
                string path = DeleteOutputPath("SavePipe", name);
                using (var writer = OpenWriter(path))
                {
                    ShowSchemaCommand.RunOnData(writer,
                        new ShowSchemaCommand.Arguments() { ShowMetadataValues = true, ShowSteps = true },
                        pipe1);
                }
                if (!CheckEquality("SavePipe", name))
                    Log("*** ShowSchema failed on pipe1");
                else
                {
                    path = DeleteOutputPath("SavePipe", name);
                    using (var writer = OpenWriter(path))
                    {
                        ShowSchemaCommand.RunOnData(writer,
                            new ShowSchemaCommand.Arguments() { ShowMetadataValues = true, ShowSteps = true },
                            pipe2);
                    }
                    if (!CheckEquality("SavePipe", name))
                        Log("*** ShowSchema failed on pipe2");
                }
            }

            // REVIEW: What about tests for ensuring that shuffling produces an actual shuffled version?

            return pipe1;
        }

        protected IDataLoader CreatePipeDataLoader(IHostEnvironment env, string pathData, string[] argsPipe, out MultiFileSource files)
        {
            VerifyArgParsing(env, argsPipe);

            // Default to breast-cancer.txt.
            if (string.IsNullOrEmpty(pathData))
                pathData = GetDataPath("breast-cancer.txt");

            files = new MultiFileSource(pathData == "<none>" ? null : pathData);
            var sub = new SubComponent<IDataLoader, SignatureDataLoader>("Pipe", argsPipe);
            var pipe = sub.CreateInstance(env, files);
            if (!CheckMetadataTypes(pipe.Schema))
                Failed();

            return pipe;
        }

        /// <summary>
        /// Apply pipe's transforms and optionally ChooseColumns transform to newView, 
        /// and test if pipe and newPipe have the same schema and values.
        /// </summary>
        protected void TestApplyTransformsToData(IHostEnvironment env, IDataLoader pipe, IDataView newView, string chooseArgs = null)
        {
            Contracts.AssertValue(pipe);
            Contracts.AssertValue(newView);

            IDataView view = pipe;
            newView = ApplyTransformUtils.ApplyAllTransformsToData(env, view, newView);
            if (!string.IsNullOrWhiteSpace(chooseArgs))
            {
                var component = new SubComponent<IDataTransform, SignatureDataTransform>("Choose", chooseArgs);
                view = component.CreateInstance(env, view);
                newView = component.CreateInstance(env, newView);
            }

            if (!CheckSameSchemas(view.Schema, newView.Schema))
                Failed();
            else if (!CheckSameValues(view, newView))
                Failed();
        }

        protected void VerifyArgParsing(IHostEnvironment env, string[] strs)
        {
            string str = CmdParser.CombineSettings(strs);
            var args = new CompositeDataLoader.Arguments();
            if (!CmdParser.ParseArguments(Env, str, args))
            {
                Fail("Parsing arguments failed!");
                return;
            }

            // For the loader and each transform, verify that custom unparsing is correct.
            VerifyCustArgs(env, args.Loader);
            foreach (var kvp in args.Transform)
                VerifyCustArgs(env, kvp.Value);
        }

        protected void VerifyCustArgs<TArg, TRes>(IHostEnvironment env, IComponentFactory<TArg, TRes> factory)
            where TRes : class
        {
            if (factory is ICommandLineComponentFactory commandLineFactory)
            {
                var str = commandLineFactory.GetSettingsString();
                var info = env.ComponentCatalog.GetLoadableClassInfo(commandLineFactory.Name, commandLineFactory.SignatureType);
                Assert.NotNull(info);
                var def = info.CreateArguments();

                var a1 = info.CreateArguments();
                CmdParser.ParseArguments(Env, str, a1);

                // Get both the expanded and custom forms.
                string exp1 = CmdParser.GetSettings(Env, a1, def, SettingsFlags.Default | SettingsFlags.NoUnparse);
                string cust = CmdParser.GetSettings(Env, a1, def);

                // Map cust back to an object, then get its full form.
                var a2 = info.CreateArguments();
                CmdParser.ParseArguments(Env, cust, a2);
                string exp2 = CmdParser.GetSettings(Env, a2, def, SettingsFlags.Default | SettingsFlags.NoUnparse);

                if (exp1 != exp2)
                    Fail("Custom unparse failed on '{0}' starting with '{1}': '{2}' vs '{3}'", commandLineFactory.Name, str, exp1, exp2);
            }
            else
            {
                Fail($"TestDataPipeBase was called with a non command line loader or transform '{factory}'");
            }
        }

        protected bool SaveLoadText(IDataView view, IHostEnvironment env,
            bool hidden = true, string suffix = "", string suffixBase = null,
            bool checkBaseline = true, bool forceDense = false, bool roundTrip = true,
            bool outputSchema = true, bool outputHeader = true)
        {
            TextSaver saver = new TextSaver(env, new TextSaver.Arguments() { Dense = forceDense, OutputSchema = outputSchema, OutputHeader = outputHeader });
            var schema = view.Schema;
            List<int> savable = new List<int>();
            for (int c = 0; c < schema.ColumnCount; ++c)
            {
                ColumnType type = schema.GetColumnType(c);
                if (saver.IsColumnSavable(type) && (hidden || !schema.IsHidden(c)))
                    savable.Add(c);
            }

            string name = TestName + suffix + "-Data" + ".txt";
            string pathData = DeleteOutputPath("SavePipe", name);

            string argsLoader;
            using (var stream = File.Create(pathData))
                saver.SaveData(out argsLoader, stream, view, savable.ToArray());

            if (checkBaseline)
            {
                string nameBase = suffixBase != null ? TestName + suffixBase + "-Data" + ".txt" : name;
                CheckEquality("SavePipe", name, nameBase);
            }

            if (!roundTrip)
                return true;

            if (savable.Count < view.Schema.ColumnCount)
            {
                // Restrict the comparison to the subset of columns we were able to save.
                var chooseargs = new ChooseColumnsByIndexTransform.Arguments();
                chooseargs.Index = savable.ToArray();
                view = new ChooseColumnsByIndexTransform(env, chooseargs, view);
            }

            var args = new TextLoader.Arguments();
            if (!CmdParser.ParseArguments(Env, argsLoader, args))
            {
                Fail("Couldn't parse the args '{0}' in '{1}'", argsLoader, pathData);
                return Failed();
            }

            // Note that we don't pass in "args", but pass in a default args so we test
            // the auto-schema parsing.
            var loadedData = TextLoader.ReadFile(env, new TextLoader.Arguments(), new MultiFileSource(pathData));
            if (!CheckMetadataTypes(loadedData.Schema))
                Failed();

            if (!CheckSameSchemas(view.Schema, loadedData.Schema, exactTypes: false, keyNames: false))
                return Failed();
            if (!CheckSameValues(view, loadedData, exactTypes: false, exactDoubles: false, checkId: false))
                return Failed();
            return true;
        }

        protected string SavePipe(IDataLoader pipe, string suffix = "", string dir = "Pipeline")
        {
            string name = TestName + suffix + ".zip";
            string pathModel = DeleteOutputPath("SavePipe", name);

            using (var file = Env.CreateOutputFile(pathModel))
            using (var strm = file.CreateWriteStream())
            using (var rep = RepositoryWriter.CreateNew(strm, Env))
            {
                ModelSaveContext.SaveModel(rep, pipe, dir);
                rep.Commit();
            }
            return pathModel;
        }

        protected string[] Concat(string[] args1, string[] args2)
        {
            string[] res = new string[args1.Length + args2.Length];
            Array.Copy(args1, res, args1.Length);
            Array.Copy(args2, 0, res, args1.Length, args2.Length);
            return res;
        }

        protected IDataLoader LoadPipe(string pathModel, IHostEnvironment env, IMultiStreamSource files)
        {
            using (var file = Env.OpenInputFile(pathModel))
            using (var strm = file.OpenReadStream())
            using (var rep = RepositoryReader.Open(strm, env))
            {
                IDataLoader pipe;
                ModelLoadContext.LoadModel<IDataLoader, SignatureLoadDataLoader>(env,
                    out pipe, rep, "Pipeline", files);
                return pipe;
            }
        }

        protected bool CheckMetadataTypes(ISchema sch)
        {
            var hs = new HashSet<string>();
            for (int col = 0; col < sch.ColumnCount; col++)
            {
                var typeSlot = sch.GetMetadataTypeOrNull(MetadataUtils.Kinds.SlotNames, col);
                var typeKeys = sch.GetMetadataTypeOrNull(MetadataUtils.Kinds.KeyValues, col);
                var all = sch.GetMetadataTypes(col);

                hs.Clear();
                foreach (var kvp in all)
                {
                    if (kvp.Key == null || kvp.Value == null)
                    {
                        Fail("Null returned from GetMetadataTypes");
                        return Failed();
                    }
                    if (!hs.Add(kvp.Key))
                    {
                        Fail("Duplicate metadata type: {0}", kvp.Key);
                        return Failed();
                    }
                    if (kvp.Key == MetadataUtils.Kinds.SlotNames)
                    {
                        if (typeSlot == null || !typeSlot.Equals(kvp.Value))
                        {
                            Fail("SlotNames types don't match");
                            return Failed();
                        }
                        typeSlot = null;
                        continue;
                    }
                    if (kvp.Key == MetadataUtils.Kinds.KeyValues)
                    {
                        if (typeKeys == null || !typeKeys.Equals(kvp.Value))
                        {
                            Fail("KeyValues types don't match");
                            return Failed();
                        }
                        typeKeys = null;
                        continue;
                    }

                    var type = sch.GetMetadataTypeOrNull(kvp.Key, col);
                    if (type == null || !type.Equals(kvp.Value))
                    {
                        Fail("{0} types don't match", kvp.Key);
                        return Failed();
                    }
                }

                if (!Check(typeSlot == null, "SlotNames not in GetMetadataTypes"))
                    return Failed();
                if (!Check(typeKeys == null, "KeyValues not in GetMetadataTypes"))
                    return Failed();
            }

            return true;
        }

        protected bool CheckSameSchemas(ISchema sch1, ISchema sch2, bool exactTypes = true, bool keyNames = true)
        {
            if (sch1.ColumnCount != sch2.ColumnCount)
            {
                Fail("column count mismatch: {0} vs {1}", sch1.ColumnCount, sch2.ColumnCount);
                return Failed();
            }

            for (int col = 0; col < sch1.ColumnCount; col++)
            {
                string name1 = sch1.GetColumnName(col);
                string name2 = sch2.GetColumnName(col);
                if (name1 != name2)
                {
                    Fail("column name mismatch at index {0}: {1} vs {2}", col, name1, name2);
                    return Failed();
                }
                var type1 = sch1.GetColumnType(col);
                var type2 = sch2.GetColumnType(col);
                if (!EqualTypes(type1, type2, exactTypes))
                {
                    Fail("column type mismatch at index {0}", col);
                    return Failed();
                }

                // This ensures that the two schemas map names to the same column indices.
                int col1, col2;
                bool f1 = sch1.TryGetColumnIndex(name1, out col1);
                bool f2 = sch2.TryGetColumnIndex(name2, out col2);
                if (!Check(f1, "TryGetColumnIndex unexpectedly failed"))
                    return Failed();
                if (!Check(f2, "TryGetColumnIndex unexpectedly failed"))
                    return Failed();
                if (col1 != col2)
                {
                    Fail("TryGetColumnIndex on '{0}' produced different results: '{1}' vs '{2}'", name1, col1, col2);
                    return Failed();
                }

                // This checks that an unknown metadata kind does the right thing.
                if (!CheckMetadataNames("PurpleDragonScales", -1, sch1, sch2, col, exactTypes, true))
                    return Failed();

                int size = type1.IsVector ? type1.VectorSize : -1;
                if (!CheckMetadataNames(MetadataUtils.Kinds.SlotNames, size, sch1, sch2, col, exactTypes, true))
                    return Failed();

                if (!keyNames)
                    continue;

                size = type1.ItemType.IsKey ? type1.ItemType.KeyCount : -1;
                if (!CheckMetadataNames(MetadataUtils.Kinds.KeyValues, size, sch1, sch2, col, exactTypes, false))
                    return Failed();
            }

            return true;
        }

        protected bool CheckMetadataNames(string kind, int size, ISchema sch1, ISchema sch2, int col, bool exactTypes, bool mustBeText)
        {
            var names1 = default(VBuffer<ReadOnlyMemory<char>>);
            var names2 = default(VBuffer<ReadOnlyMemory<char>>);

            var t1 = sch1.GetMetadataTypeOrNull(kind, col);
            var t2 = sch2.GetMetadataTypeOrNull(kind, col);
            if ((t1 == null) != (t2 == null))
            {
                Fail("Different null-ness of {0} metadata types", kind);
                return Failed();
            }
            if (t1 == null)
            {
                if (!CheckMetadataCallFailure(kind, sch1, col, ref names1))
                    return Failed();
                if (!CheckMetadataCallFailure(kind, sch2, col, ref names2))
                    return Failed();
                return true;
            }
            if (!EqualTypes(t1, t2, exactTypes))
            {
                Fail("Different {0} metadata types: {0} vs {1}", kind, t1, t2);
                return Failed();
            }
            if (!t1.ItemType.IsText)
            {
                if (!mustBeText)
                {
                    Log("Metadata '{0}' was not text so skipping comparison", kind);
                    return true; // REVIEW: Do something a bit more clever here.
                }
                Fail("Unexpected {0} metadata type", kind);
                return Failed();
            }

            if (size != t1.VectorSize)
            {
                Fail("{0} metadata type wrong size: {1} vs {2}", kind, t1.VectorSize, size);
                return Failed();
            }

            sch1.GetMetadata(kind, col, ref names1);
            sch2.GetMetadata(kind, col, ref names2);
            if (!CompareVec(in names1, in names2, size, (a, b) => a.Span.SequenceEqual(b.Span)))
            {
                Fail("Different {0} metadata values", kind);
                return Failed();
            }
            return true;
        }

        protected bool CheckMetadataCallFailure(string kind, ISchema sch, int col, ref VBuffer<ReadOnlyMemory<char>> names)
        {
            try
            {
                sch.GetMetadata(kind, col, ref names);
                Fail("Getting {0} metadata unexpectedly succeeded", kind);
                return Failed();
            }
            catch (InvalidOperationException ex)
            {
                if (ex.Message != "Invalid call to GetMetadata")
                {
                    Fail("Message from GetMetadata failed call doesn't match expected message: {0}", ex.Message);
                    return Failed();
                }
            }
            return true;
        }

        protected bool SaveLoad(IDataView view, IHostEnvironment env, string suffix = "")
        {
            var saverArgs = new BinarySaver.Arguments();
            saverArgs.MaxBytesPerBlock = null;
            saverArgs.MaxRowsPerBlock = 100;
            BinarySaver saver = new BinarySaver(env, saverArgs);

            var schema = view.Schema;
            List<int> savable = new List<int>();
            for (int c = 0; c < schema.ColumnCount; ++c)
            {
                ColumnType type = schema.GetColumnType(c);
                if (saver.IsColumnSavable(type))
                    savable.Add(c);
            }

            string name = TestName + suffix + "-Data" + ".idv";
            string pathData = DeleteOutputPath("SavePipe", name);

            using (var stream = File.Create(pathData))
            {
                saver.SaveData(stream, view, savable.ToArray());
                Log("View saved in {0} bytes", stream.Length);
            }

            if (savable.Count < view.Schema.ColumnCount)
            {
                // Restrict the comparison to the subset of columns we were able to save.
                var chooseargs = new ChooseColumnsByIndexTransform.Arguments();
                chooseargs.Index = savable.ToArray();
                view = new ChooseColumnsByIndexTransform(env, chooseargs, view);
            }

            var args = new BinaryLoader.Arguments();
            using (BinaryLoader loader = new BinaryLoader(env, args, pathData))
            {
                if (!CheckMetadataTypes(loader.Schema))
                    return Failed();

                if (!CheckSameSchemas(view.Schema, loader.Schema))
                    return Failed();
                if (!CheckSameValues(view, loader, checkId: false))
                    return Failed();
            }
            return true;
        }

        protected bool SaveLoadTransposed(IDataView view, IHostEnvironment env, string suffix = "")
        {
            var saverArgs = new TransposeSaver.Arguments();
            saverArgs.WriteRowData = false; // Force it to use this the re-transposition logic.
            TransposeSaver saver = new TransposeSaver(env, saverArgs);

            var schema = view.Schema;
            List<int> savable = new List<int>();
            for (int c = 0; c < schema.ColumnCount; ++c)
            {
                ColumnType type = schema.GetColumnType(c);
                if (saver.IsColumnSavable(type))
                    savable.Add(c);
            }
            if (savable.Count == 0)
            {
                Log("No columns were savable in transposed saver, skipping");
                return true;
            }

            string name = TestName + suffix + "-Data" + ".tdv";
            string pathData = DeleteOutputPath("SavePipe", name);

            using (var stream = File.Create(pathData))
            {
                saver.SaveData(stream, view, savable.ToArray());
                Log("View saved in {0} bytes", stream.Length);
            }

            if (savable.Count < view.Schema.ColumnCount)
            {
                // Restrict the comparison to the subset of columns we were able to save.
                var chooseargs = new ChooseColumnsByIndexTransform.Arguments();
                chooseargs.Index = savable.ToArray();
                view = new ChooseColumnsByIndexTransform(env, chooseargs, view);
            }

            var args = new TransposeLoader.Arguments();
            MultiFileSource src = new MultiFileSource(pathData);
            TransposeLoader loader = new TransposeLoader(env, args, src);
            if (!CheckMetadataTypes(loader.Schema))
                return Failed();

            if (!CheckSameSchemas(view.Schema, loader.Schema))
                return Failed();
            if (!CheckSameValues(view, loader, checkId: false))
                return Failed();
            return true;
        }
    }

    public abstract partial class TestDataViewBase : BaseTestBaseline
    {

        public class SentimentData
        {
            [ColumnName("Label")]
            public bool Sentiment;
            public string SentimentText;
        }

        public class SentimentPrediction
        {
            [ColumnName("PredictedLabel")]
            public bool Sentiment;

            public float Score;
        }

        private static TextLoader.Arguments MakeSentimentTextLoaderArgs()
        {
            return new TextLoader.Arguments()
            {
                Separator = "tab",
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("Label", DataKind.BL, 0),
                    new TextLoader.Column("SentimentText", DataKind.Text, 1)
                }
            };
        }

        protected bool Failed()
        {
            Contracts.Assert(!IsPassing);
            return false;
        }

        protected bool EqualTypes(ColumnType type1, ColumnType type2, bool exactTypes)
        {
            Contracts.AssertValue(type1);
            Contracts.AssertValue(type2);

            return exactTypes ? type1.Equals(type2) : type1.SameSizeAndItemType(type2);
        }

        protected bool CheckSameValues(IDataView view1, IDataView view2, bool exactTypes = true, bool exactDoubles = true, bool checkId = true)
        {
            Contracts.Assert(view1.Schema.ColumnCount == view2.Schema.ColumnCount);

            bool all = true;
            bool tmp;

            using (var curs1 = view1.GetRowCursor(col => true))
            using (var curs2 = view2.GetRowCursor(col => true))
            {
                Check(curs1.Schema == view1.Schema, "Schema of view 1 and its cursor differed");
                Check(curs2.Schema == view2.Schema, "Schema of view 2 and its cursor differed");
                tmp = CheckSameValues(curs1, curs2, exactTypes, exactDoubles, checkId, true);
            }
            Check(tmp, "All same failed");
            all &= tmp;

            using (var curs1 = view1.GetRowCursor(col => true))
            using (var curs2 = view2.GetRowCursor(col => (col & 1) == 0, null))
            {
                Check(curs1.Schema == view1.Schema, "Schema of view 1 and its cursor differed");
                Check(curs2.Schema == view2.Schema, "Schema of view 2 and its cursor differed");
                tmp = CheckSameValues(curs1, curs2, exactTypes, exactDoubles, checkId, false);
            }
            Check(tmp, "Even same failed");
            all &= tmp;

            using (var curs1 = view1.GetRowCursor(col => true))
            using (var curs2 = view2.GetRowCursor(col => (col & 1) != 0, null))
            {
                Check(curs1.Schema == view1.Schema, "Schema of view 1 and its cursor differed");
                Check(curs2.Schema == view2.Schema, "Schema of view 2 and its cursor differed");
                tmp = CheckSameValues(curs1, curs2, exactTypes, exactDoubles, checkId, false);
            }
            Check(tmp, "Odd same failed");

            using (var curs1 = view1.GetRowCursor(col => true))
            {
                Check(curs1.Schema == view1.Schema, "Schema of view 1 and its cursor differed");
                tmp = CheckSameValues(curs1, view2, exactTypes, exactDoubles, checkId);
            }
            Check(tmp, "Single value same failed");

            all &= tmp;
            return all;
        }

        protected bool CheckSameValues(IRowCursor curs1, IRowCursor curs2, bool exactTypes, bool exactDoubles, bool checkId, bool checkIdCollisions = true)
        {
            Contracts.Assert(curs1.Schema.ColumnCount == curs2.Schema.ColumnCount);

            // Get the comparison delegates for each column.
            int colLim = curs1.Schema.ColumnCount;
            Func<bool>[] comps = new Func<bool>[colLim];
            for (int col = 0; col < colLim; col++)
            {
                var f1 = curs1.IsColumnActive(col);
                var f2 = curs2.IsColumnActive(col);

                if (f1 && f2)
                {
                    var type1 = curs1.Schema.GetColumnType(col);
                    var type2 = curs2.Schema.GetColumnType(col);
                    if (!EqualTypes(type1, type2, exactTypes))
                    {
                        Fail("Different types");
                        return Failed();
                    }
                    comps[col] = GetColumnComparer(curs1, curs2, col, type1, exactDoubles);
                }
            }
            ValueGetter<UInt128> idGetter = null;
            Func<bool> idComp = checkId ? GetIdComparer(curs1, curs2, out idGetter) : null;
            HashSet<UInt128> idsSeen = null;
            if (checkIdCollisions && idGetter == null)
                idGetter = curs1.GetIdGetter();
            long idCollisions = 0;
            UInt128 id = default(UInt128);

            for (; ; )
            {
                bool f1 = curs1.MoveNext();
                bool f2 = curs2.MoveNext();
                if (f1 != f2)
                {
                    if (f1)
                        Fail("Left has more rows at position: {0}", curs1.Position);
                    else
                        Fail("Right has more rows at position: {0}", curs2.Position);
                    return Failed();
                }

                if (!f1)
                {
                    if (idCollisions > 0)
                        Fail("{0} id collisions among {1} items", idCollisions, Utils.Size(idsSeen) + idCollisions);
                    return idCollisions == 0;
                }
                else if (checkIdCollisions)
                {
                    idGetter(ref id);
                    if (!Utils.Add(ref idsSeen, id))
                    {
                        if (idCollisions == 0)
                            Log("Id collision {0} at {1}, further collisions will not be logged", id, curs1.Position);
                        idCollisions++;
                    }
                }

                Contracts.Assert(curs1.Position == curs2.Position);

                for (int col = 0; col < colLim; col++)
                {
                    var comp = comps[col];
                    if (comp != null && !comp())
                    {
                        Fail("Different values in column {0} of row {1}", col, curs1.Position);
                        return Failed();
                    }
                    if (idComp != null && !idComp())
                    {
                        Fail("Different values in ID of row {0}", curs1.Position);
                        return Failed();
                    }
                }
            }
        }

        protected bool CheckSameValues(IRowCursor curs1, IDataView view2, bool exactTypes = true, bool exactDoubles = true, bool checkId = true)
        {
            Contracts.Assert(curs1.Schema.ColumnCount == view2.Schema.ColumnCount);

            // Get a cursor for each column.
            int colLim = curs1.Schema.ColumnCount;
            var cursors = new IRowCursor[colLim];
            try
            {
                for (int col = 0; col < colLim; col++)
                {
                    // curs1 should have all columns active (for simplicity of the code here).
                    Contracts.Assert(curs1.IsColumnActive(col));
                    cursors[col] = view2.GetRowCursor(c => c == col);
                }

                // Get the comparison delegates for each column.
                Func<bool>[] comps = new Func<bool>[colLim];
                // We have also one ID comparison delegate for each cursor.
                Func<bool>[] idComps = new Func<bool>[cursors.Length];
                for (int col = 0; col < colLim; col++)
                {
                    Contracts.Assert(cursors[col] != null);
                    var type1 = curs1.Schema.GetColumnType(col);
                    var type2 = cursors[col].Schema.GetColumnType(col);
                    if (!EqualTypes(type1, type2, exactTypes))
                    {
                        Fail("Different types");
                        return Failed();
                    }
                    comps[col] = GetColumnComparer(curs1, cursors[col], col, type1, exactDoubles);
                    ValueGetter<UInt128> idGetter;
                    idComps[col] = checkId ? GetIdComparer(curs1, cursors[col], out idGetter) : null;
                }

                for (; ; )
                {
                    bool f1 = curs1.MoveNext();
                    for (int col = 0; col < colLim; col++)
                    {
                        bool f2 = cursors[col].MoveNext();
                        if (f1 != f2)
                        {
                            if (f1)
                                Fail("Left has more rows at position: {0}", curs1.Position);
                            else
                                Fail("Right {0} has more rows at position: {1}", col, cursors[2].Position);
                            return Failed();
                        }
                    }

                    if (!f1)
                        return true;

                    for (int col = 0; col < colLim; col++)
                    {
                        Contracts.Assert(curs1.Position == cursors[col].Position);
                        var comp = comps[col];
                        if (comp != null && !comp())
                        {
                            Fail("Different values in column {0} of row {1}", col, curs1.Position);
                            return Failed();
                        }
                        comp = idComps[col];
                        if (comp != null && !comp())
                        {
                            Fail("Different values in ID values for column {0} cursor of row {1}", col, curs1.Position);
                            return Failed();
                        }
                    }
                }
            }
            finally
            {
                for (int col = 0; col < colLim; col++)
                {
                    var c = cursors[col];
                    if (c != null)
                        c.Dispose();
                }
            }
        }

        protected Func<bool> GetIdComparer(IRow r1, IRow r2, out ValueGetter<UInt128> idGetter)
        {
            var g1 = r1.GetIdGetter();
            idGetter = g1;
            var g2 = r2.GetIdGetter();
            UInt128 v1 = default(UInt128);
            UInt128 v2 = default(UInt128);
            return
                () =>
                {
                    g1(ref v1);
                    g2(ref v2);
                    return v1.Equals(v2);
                };
        }

        protected Func<bool> GetColumnComparer(IRow r1, IRow r2, int col, ColumnType type, bool exactDoubles)
        {
            if (!type.IsVector)
            {
                switch (type.RawKind)
                {
                    case DataKind.I1:
                        return GetComparerOne<sbyte>(r1, r2, col, (x, y) => x == y);
                    case DataKind.U1:
                        return GetComparerOne<byte>(r1, r2, col, (x, y) => x == y);
                    case DataKind.I2:
                        return GetComparerOne<short>(r1, r2, col, (x, y) => x == y);
                    case DataKind.U2:
                        return GetComparerOne<ushort>(r1, r2, col, (x, y) => x == y);
                    case DataKind.I4:
                        return GetComparerOne<int>(r1, r2, col, (x, y) => x == y);
                    case DataKind.U4:
                        return GetComparerOne<uint>(r1, r2, col, (x, y) => x == y);
                    case DataKind.I8:
                        return GetComparerOne<long>(r1, r2, col, (x, y) => x == y);
                    case DataKind.U8:
                        return GetComparerOne<ulong>(r1, r2, col, (x, y) => x == y);
                    case DataKind.R4:
                        return GetComparerOne<Single>(r1, r2, col, (x, y) => FloatUtils.GetBits(x) == FloatUtils.GetBits(y));
                    case DataKind.R8:
                        if (exactDoubles)
                            return GetComparerOne<Double>(r1, r2, col, (x, y) => FloatUtils.GetBits(x) == FloatUtils.GetBits(y));
                        else
                            return GetComparerOne<Double>(r1, r2, col, EqualWithEpsDouble);
                    case DataKind.Text:
                        return GetComparerOne<ReadOnlyMemory<char>>(r1, r2, col, (a ,b) => a.Span.SequenceEqual(b.Span));
                    case DataKind.Bool:
                        return GetComparerOne<bool>(r1, r2, col, (x, y) => x == y);
                    case DataKind.TimeSpan:
                        return GetComparerOne<TimeSpan>(r1, r2, col, (x, y) => x == y);
                    case DataKind.DT:
                        return GetComparerOne<DateTime>(r1, r2, col, (x, y) => x == y);
                    case DataKind.DZ:
                        return GetComparerOne<DateTimeOffset>(r1, r2, col, (x, y) => x.Equals(y));
                    case DataKind.UG:
                        return GetComparerOne<UInt128>(r1, r2, col, (x, y) => x.Equals(y));
                    case (DataKind)0:
                        // We cannot compare custom types (including image).
                        return () => true;
                }
            }
            else
            {
                int size = type.VectorSize;
                Contracts.Assert(size >= 0);
                switch (type.ItemType.RawKind)
                {
                    case DataKind.I1:
                        return GetComparerVec<sbyte>(r1, r2, col, size, (x, y) => x == y);
                    case DataKind.U1:
                        return GetComparerVec<byte>(r1, r2, col, size, (x, y) => x == y);
                    case DataKind.I2:
                        return GetComparerVec<short>(r1, r2, col, size, (x, y) => x == y);
                    case DataKind.U2:
                        return GetComparerVec<ushort>(r1, r2, col, size, (x, y) => x == y);
                    case DataKind.I4:
                        return GetComparerVec<int>(r1, r2, col, size, (x, y) => x == y);
                    case DataKind.U4:
                        return GetComparerVec<uint>(r1, r2, col, size, (x, y) => x == y);
                    case DataKind.I8:
                        return GetComparerVec<long>(r1, r2, col, size, (x, y) => x == y);
                    case DataKind.U8:
                        return GetComparerVec<ulong>(r1, r2, col, size, (x, y) => x == y);
                    case DataKind.R4:
                        if (exactDoubles)
                            return GetComparerVec<Single>(r1, r2, col, size, (x, y) => FloatUtils.GetBits(x) == FloatUtils.GetBits(y));
                        else
                            return GetComparerVec<Single>(r1, r2, col, size, EqualWithEpsSingle);
                    case DataKind.R8:
                        if (exactDoubles)
                            return GetComparerVec<Double>(r1, r2, col, size, (x, y) => FloatUtils.GetBits(x) == FloatUtils.GetBits(y));
                        else
                            return GetComparerVec<Double>(r1, r2, col, size, EqualWithEpsDouble);
                    case DataKind.Text:
                        return GetComparerVec<ReadOnlyMemory<char>>(r1, r2, col, size, (a, b) => a.Span.SequenceEqual(b.Span));
                    case DataKind.Bool:
                        return GetComparerVec<bool>(r1, r2, col, size, (x, y) => x == y);
                    case DataKind.TimeSpan:
                        return GetComparerVec<TimeSpan>(r1, r2, col, size, (x, y) => x == y);
                    case DataKind.DT:
                        return GetComparerVec<DateTime>(r1, r2, col, size, (x, y) => x == y);
                    case DataKind.DZ:
                        return GetComparerVec<DateTimeOffset>(r1, r2, col, size, (x, y) => x.Equals(y));
                    case DataKind.UG:
                        return GetComparerVec<UInt128>(r1, r2, col, size, (x, y) => x.Equals(y));
                }
            }

#if !CORECLR // REVIEW: Port Picture type to CoreTLC.
            if (type is PictureType)
            {
                var g1 = r1.GetGetter<Picture>(col);
                var g2 = r2.GetGetter<Picture>(col);
                Picture v1 = null;
                Picture v2 = null;
                return
                    () =>
                    {
                        g1(ref v1);
                        g2(ref v2);
                        return ComparePicture(v1, v2);
                    };
            }
#endif

            throw Contracts.Except("Unknown type in GetColumnComparer: '{0}'", type);
        }

        private const Double DoubleEps = 1e-9;

        private static bool EqualWithEpsDouble(Double x, Double y)
        {
            // bitwise comparison is needed because Abs(Inf-Inf) and Abs(NaN-NaN) are not 0s.
            return FloatUtils.GetBits(x) == FloatUtils.GetBits(y) || Math.Abs(x - y) < DoubleEps;
        }

        private const float SingleEps = 1e-6f;

        private static bool EqualWithEpsSingle(float x, float y)
        {
            // bitwise comparison is needed because Abs(Inf-Inf) and Abs(NaN-NaN) are not 0s.
            return FloatUtils.GetBits(x) == FloatUtils.GetBits(y) || Math.Abs(x - y) < SingleEps;
        }

        protected Func<bool> GetComparerOne<T>(IRow r1, IRow r2, int col, Func<T, T, bool> fn)
        {
            var g1 = r1.GetGetter<T>(col);
            var g2 = r2.GetGetter<T>(col);
            T v1 = default(T);
            T v2 = default(T);
            return
                () =>
                {
                    g1(ref v1);
                    g2(ref v2);
                    if (!fn(v1, v2))
                        return false;
                    return true;
                };
        }

        protected Func<bool> GetComparerVec<T>(IRow r1, IRow r2, int col, int size, Func<T, T, bool> fn)
        {
            var g1 = r1.GetGetter<VBuffer<T>>(col);
            var g2 = r2.GetGetter<VBuffer<T>>(col);
            var v1 = default(VBuffer<T>);
            var v2 = default(VBuffer<T>);
            return
                () =>
                {
                    g1(ref v1);
                    g2(ref v2);
                    return CompareVec<T>(in v1, in v2, size, fn);
                };
        }

        protected bool CompareVec<T>(in VBuffer<T> v1, in VBuffer<T> v2, int size, Func<T, T, bool> fn)
        {
            return CompareVec(in v1, in v2, size, (i, x, y) => fn(x, y));
        }

        protected bool CompareVec<T>(in VBuffer<T> v1, in VBuffer<T> v2, int size, Func<int, T, T, bool> fn)
        {
            Contracts.Assert(size == 0 || v1.Length == size);
            Contracts.Assert(size == 0 || v2.Length == size);
            Contracts.Assert(v1.Length == v2.Length);

            if (v1.IsDense && v2.IsDense)
            {
                for (int i = 0; i < v1.Length; i++)
                {
                    var x1 = v1.Values[i];
                    var x2 = v2.Values[i];
                    if (!fn(i, x1, x2))
                        return false;
                }
                return true;
            }

            Contracts.Assert(!v1.IsDense || !v2.IsDense);
            int iiv1 = 0;
            int iiv2 = 0;
            for (; ; )
            {
                int iv1 = v1.IsDense ? iiv1 : iiv1 < v1.Count ? v1.Indices[iiv1] : v1.Length;
                int iv2 = v2.IsDense ? iiv2 : iiv2 < v2.Count ? v2.Indices[iiv2] : v2.Length;
                T x1, x2;
                int iv;
                if (iv1 == iv2)
                {
                    if (iv1 == v1.Length)
                        return true;
                    x1 = v1.Values[iiv1];
                    x2 = v2.Values[iiv2];
                    iv = iv1;
                    iiv1++;
                    iiv2++;
                }
                else if (iv1 < iv2)
                {
                    x1 = v1.Values[iiv1];
                    x2 = default(T);
                    iv = iv1;
                    iiv1++;
                }
                else
                {
                    x1 = default(T);
                    x2 = v2.Values[iiv2];
                    iv = iv2;
                    iiv2++;
                }
                if (!fn(iv, x1, x2))
                    return false;
            }
        }

        // Verifies the equality of the values returned by the single valued getters passed in as parameters.
        protected void VerifyOneEquality<T>(ValueGetter<T> oneGetter, ValueGetter<T> oneNGetter)
        {
            T f1 = default(T);
            T f1n = default(T);
            oneGetter(ref f1);
            oneNGetter(ref f1n);
            Assert.Equal(f1, f1n);
        }

        // Verifies the equality of the values returned by the vector valued getters passed in as parameters using the provided compare function.
        protected void VerifyVecEquality<T>(ValueGetter<VBuffer<T>> vecGetter, ValueGetter<VBuffer<T>> vecNGetter, Func<int, T, T, bool> compare, int size)
        {
            VBuffer<T> fv = default(VBuffer<T>);
            VBuffer<T> fvn = default(VBuffer<T>);
            vecGetter(ref fv);
            vecNGetter(ref fvn);
            Assert.True(CompareVec(in fv, in fvn, size, compare));
        }

#if !CORECLR
        // REVIEW: Port Picture type to Core TLC.
        protected bool ComparePicture(Picture v1, Picture v2)
        {
            if (v1 == null || v2 == null)
                return v1 == v2;

            var p1 = v1.Contents.Pixels;
            var p2 = v2.Contents.Pixels;

            if (p1.Width != p2.Width)
                return false;
            if (p1.Height != p2.Height)
                return false;
            if (p1.PixelFormat != p2.PixelFormat)
                return false;

            for (int y = 0; y < p1.Height; y++)
            {
                for (int x = 0; x < p1.Width; x++)
                {
                    var x1 = p1.GetPixel(x, y);
                    var x2 = p2.GetPixel(x, y);
                    if (x1 != x2)
                        return false;
                }
            }
            return true;
        }
#endif
    }
}
