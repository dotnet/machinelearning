// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.TestFramework;
using Xunit;

namespace Microsoft.ML.RunTests
{
    public abstract partial class TestDataPipeBase : TestDataViewBase
    {
        public const string IrisDataPath = "iris.data";

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
            ML.Model.Save(transformer, validFitInput.Schema, modelPath);

            ITransformer loadedTransformer;
            DataViewSchema loadedInputSchema;
            using (var fs = File.OpenRead(modelPath))
                loadedTransformer = ML.Model.Load(fs, out loadedInputSchema);
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
                    CheckSameSchemas(schema, mapper.OutputSchema);
                }
                else
                {
                    mustFail(() => transformer.GetRowToRowMapper(data.Schema));
                }

                // Loaded transformer needs to have the same schema propagation.
                CheckSameSchemas(schema, loadedTransformer.GetOutputSchema(data.Schema));
                // Loaded schema needs to have the same schema as data.
                CheckSameSchemas(data.Schema, loadedInputSchema);

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
            Assert.True(promised.Count == delivered.Count);
            var sortedCols1 = promised.OrderBy(x => x.Name);
            var sortedCols2 = delivered.OrderBy(x => x.Name);

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
        internal ILegacyDataLoader TestCore(string pathData, bool keepHidden, string[] argsPipe,
            Action<ILegacyDataLoader> actLoader = null, string suffix = "", string suffixBase = null, bool checkBaseline = true,
            bool forceDense = false, bool logCurs = false, bool roundTripText = true,
            bool checkTranspose = false, bool checkId = true, bool baselineSchema = true, int digitsOfPrecision = DigitsOfPrecision)
        {
            Contracts.AssertValue(Env);

            MultiFileSource files;
            ILegacyDataLoader compositeLoader;
            var pipe1 = compositeLoader = CreatePipeDataLoader(_env, pathData, argsPipe, out files);

            actLoader?.Invoke(compositeLoader);

            // Re-apply pipe to the loader and check equality.
            var comp = compositeLoader as LegacyCompositeDataLoader;
            IDataView srcLoader = null;
            if (comp != null)
            {
                srcLoader = comp.View;
                while (srcLoader is IDataTransform)
                    srcLoader = ((IDataTransform)srcLoader).Source;
                var reappliedPipe = ApplyTransformUtils.ApplyAllTransformsToData(_env, comp.View, srcLoader);
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
                using (_env.RedirectChannelOutput(writer, writer))
                {
                    long count = 0;
                    using (var curs = pipe1.GetRowCursorForAllColumns())
                    {
                        while (curs.MoveNext())
                        {
                            count++;
                        }
                    }
                    writer.WriteLine("Cursored through {0} rows", count);
                }

                CheckEqualityNormalized("SavePipe", name, digitsOfPrecision: digitsOfPrecision);
            }

            var pathModel = SavePipe(pipe1, suffix);
            var pipe2 = LoadPipe(pathModel, _env, files);
            if (!CheckMetadataTypes(pipe2.Schema))
                Failed();

            if (!CheckSameSchemas(pipe1.Schema, pipe2.Schema))
                Failed();
            else if (!CheckSameValues(pipe1, pipe2, checkId: checkId))
                Failed();

            if (pipe1.Schema.Count > 0)
            {
                // The text saver fails if there are no columns, so we cannot check in that case.
                if (!SaveLoadText(pipe1, _env, keepHidden, suffix, suffixBase, checkBaseline, forceDense, roundTripText, digitsOfPrecision: digitsOfPrecision))
                    Failed();
                // The transpose saver likewise fails for the same reason.
                if (checkTranspose && !SaveLoadTransposed(pipe1, _env, suffix))
                    Failed();
            }
            if (!SaveLoad(pipe1, _env, suffix))
                Failed();

            // Check that the pipe doesn't shuffle when it cannot :).
            if (srcLoader != null)
            {
                // First we need to cache the data so it can be shuffled.
                var cachedData = new CacheDataView(_env, srcLoader, null);
                var newPipe = ApplyTransformUtils.ApplyAllTransformsToData(_env, comp.View, cachedData);
                if (!newPipe.CanShuffle)
                {
                    using (var c1 = newPipe.GetRowCursor(newPipe.Schema, new Random(123)))
                    using (var c2 = newPipe.GetRowCursorForAllColumns())
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
                if (!CheckEquality("SavePipe", name, digitsOfPrecision: digitsOfPrecision))
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
                    if (!CheckEquality("SavePipe", name, digitsOfPrecision: digitsOfPrecision))
                        Log("*** ShowSchema failed on pipe2");
                }
            }

            // REVIEW: What about tests for ensuring that shuffling produces an actual shuffled version?

            return pipe1;
        }

        private ILegacyDataLoader CreatePipeDataLoader(IHostEnvironment env, string pathData, string[] argsPipe, out MultiFileSource files)
        {
            VerifyArgParsing(env, argsPipe);

            // Default to breast-cancer.txt.
            if (string.IsNullOrEmpty(pathData))
                pathData = GetDataPath("breast-cancer.txt");

            files = new MultiFileSource(pathData == "<none>" ? null : pathData);
            var sub = new SubComponent<ILegacyDataLoader, SignatureDataLoader>("Pipe", argsPipe);
            var pipe = sub.CreateInstance(env, files);
            if (!CheckMetadataTypes(pipe.Schema))
                Failed();

            return pipe;
        }

        protected void VerifyArgParsing(IHostEnvironment env, string[] strs)
        {
            string str = CmdParser.CombineSettings(strs);
            var args = new LegacyCompositeDataLoader.Arguments();
            if (!CmdParser.ParseArguments(Env, str, args))
            {
                Fail("Parsing arguments failed!");
                return;
            }

            // For the loader and each transform, verify that custom unparsing is correct.
            VerifyCustArgs(env, args.Loader);
            foreach (var kvp in args.Transforms)
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
            bool outputSchema = true, bool outputHeader = true, int digitsOfPrecision = DigitsOfPrecision)
        {
            TextSaver saver = new TextSaver(env, new TextSaver.Arguments() { Dense = forceDense, OutputSchema = outputSchema, OutputHeader = outputHeader });
            var schema = view.Schema;
            List<int> savable = new List<int>();
            for (int c = 0; c < schema.Count; ++c)
            {
                DataViewType type = schema[c].Type;
                if (saver.IsColumnSavable(type) && (hidden || !schema[c].IsHidden))
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
                CheckEquality("SavePipe", name, nameBase, digitsOfPrecision: digitsOfPrecision);
            }

            if (!roundTrip)
                return true;

            if (savable.Count < view.Schema.Count)
            {
                // Restrict the comparison to the subset of columns we were able to save.
                var chooseargs = new ChooseColumnsByIndexTransform.Options();
                chooseargs.Indices = savable.ToArray();
                view = new ChooseColumnsByIndexTransform(env, chooseargs, view);
            }

            var args = new TextLoader.Options() { AllowSparse = true, AllowQuoting = true};
            if (!CmdParser.ParseArguments(Env, argsLoader, args))
            {
                Fail("Couldn't parse the args '{0}' in '{1}'", argsLoader, pathData);
                return Failed();
            }

            // Note that we don't pass in "args", but pass in a default args so we test
            // the auto-schema parsing.
            var loadedData = ML.Data.LoadFromTextFile(pathData, options: args);
            if (!CheckMetadataTypes(loadedData.Schema))
                Failed();

            if (!CheckSameSchemas(view.Schema, loadedData.Schema, exactTypes: false, keyNames: false))
                return Failed();
            if (!CheckSameValues(view, loadedData, exactTypes: false, exactDoubles: false, checkId: false))
                return Failed();
            return true;
        }

        protected private string SavePipe(ILegacyDataLoader pipe, string suffix = "", string dir = "Pipeline")
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

        private ILegacyDataLoader LoadPipe(string pathModel, IHostEnvironment env, IMultiStreamSource files)
        {
            using (var file = Env.OpenInputFile(pathModel))
            using (var strm = file.OpenReadStream())
            using (var rep = RepositoryReader.Open(strm, env))
            {
                ILegacyDataLoader pipe;
                ModelLoadContext.LoadModel<ILegacyDataLoader, SignatureLoadDataLoader>(env,
                    out pipe, rep, "Pipeline", files);
                return pipe;
            }
        }

        protected bool CheckMetadataTypes(DataViewSchema sch)
        {
            var hs = new HashSet<string>();
            for (int col = 0; col < sch.Count; col++)
            {
                var typeSlot = sch[col].Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.SlotNames)?.Type;
                var typeKeys = sch[col].Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.KeyValues)?.Type;

                hs.Clear();
                foreach (var metaColumn in sch[col].Annotations.Schema)
                {
                    if (metaColumn.Name == null || metaColumn.Type == null)
                    {
                        Fail("Null returned from GetMetadataTypes");
                        return Failed();
                    }
                    if (!hs.Add(metaColumn.Name))
                    {
                        Fail("Duplicate metadata type: {0}", metaColumn.Name);
                        return Failed();
                    }
                    if (metaColumn.Name == AnnotationUtils.Kinds.SlotNames)
                    {
                        if (typeSlot == null || !typeSlot.Equals(metaColumn.Type))
                        {
                            Fail("SlotNames types don't match");
                            return Failed();
                        }
                        typeSlot = null;
                        continue;
                    }
                    if (metaColumn.Name == AnnotationUtils.Kinds.KeyValues)
                    {
                        if (typeKeys == null || !typeKeys.Equals(metaColumn.Type))
                        {
                            Fail("KeyValues types don't match");
                            return Failed();
                        }
                        typeKeys = null;
                    }
                }

                if (!Check(typeSlot == null, "SlotNames not in GetMetadataTypes"))
                    return Failed();
                if (!Check(typeKeys == null, "KeyValues not in GetMetadataTypes"))
                    return Failed();
            }

            return true;
        }

        protected bool CheckSameSchemas(DataViewSchema sch1, DataViewSchema sch2, bool exactTypes = true, bool keyNames = true)
        {
            if (sch1.Count != sch2.Count)
            {
                Fail("column count mismatch: {0} vs {1}", sch1.Count, sch2.Count);
                return Failed();
            }

            for (int col = 0; col < sch1.Count; col++)
            {
                string name1 = sch1[col].Name;
                string name2 = sch2[col].Name;
                if (name1 != name2)
                {
                    Fail("column name mismatch at index {0}: {1} vs {2}", col, name1, name2);
                    return Failed();
                }
                var type1 = sch1[col].Type;
                var type2 = sch2[col].Type;
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
                if (!CheckMetadataNames("PurpleDragonScales", 0, sch1, sch2, col, exactTypes, true))
                    return Failed();

                ulong vsize = type1 is VectorType vectorType ? (ulong)vectorType.Size : 0;
                if (!CheckMetadataNames(AnnotationUtils.Kinds.SlotNames, vsize, sch1, sch2, col, exactTypes, true))
                    return Failed();

                if (!keyNames)
                    continue;

                ulong ksize = type1.GetItemType() is KeyType keyType ? keyType.Count : 0;
                if (!CheckMetadataNames(AnnotationUtils.Kinds.KeyValues, ksize, sch1, sch2, col, exactTypes, false))
                    return Failed();
            }

            return true;
        }

        protected bool CheckMetadataNames(string kind, ulong size, DataViewSchema sch1, DataViewSchema sch2, int col, bool exactTypes, bool mustBeText)
        {
            var names1 = default(VBuffer<ReadOnlyMemory<char>>);
            var names2 = default(VBuffer<ReadOnlyMemory<char>>);

            var t1 = sch1[col].Annotations.Schema.GetColumnOrNull(kind)?.Type;
            var t2 = sch2[col].Annotations.Schema.GetColumnOrNull(kind)?.Type;
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
            if (size > int.MaxValue)
                Fail(nameof(KeyType) + "." + nameof(KeyType.Count) + "is larger than int.MaxValue");
            if (!EqualTypes(t1, t2, exactTypes))
            {
                Fail("Different {0} metadata types: {0} vs {1}", kind, t1, t2);
                return Failed();
            }
            if (!(t1.GetItemType() is TextDataViewType))
            {
                if (!mustBeText)
                {
                    Log("Metadata '{0}' was not text so skipping comparison", kind);
                    return true; // REVIEW: Do something a bit more clever here.
                }
                Fail("Unexpected {0} metadata type", kind);
                return Failed();
            }

            if ((int)size != t1.GetVectorSize())
            {
                Fail("{0} metadata type wrong size: {1} vs {2}", kind, t1.GetVectorSize(), size);
                return Failed();
            }

            sch1[col].Annotations.GetValue(kind, ref names1);
            sch2[col].Annotations.GetValue(kind, ref names2);
            if (!CompareVec(in names1, in names2, (int)size, (a, b) => a.Span.SequenceEqual(b.Span)))
            {
                Fail("Different {0} metadata values", kind);
                return Failed();
            }
            return true;
        }

        protected bool CheckMetadataCallFailure(string kind, DataViewSchema sch, int col, ref VBuffer<ReadOnlyMemory<char>> names)
        {
            try
            {
                sch[col].Annotations.GetValue(kind, ref names);
                Fail("Getting {0} metadata unexpectedly succeeded", kind);
                return Failed();
            }
            catch (InvalidOperationException ex)
            {
                if (ex.Message != "Invalid call to 'GetValue'")
                {
                    Fail("Message from GetValue failed call doesn't match expected message: {0}", ex.Message);
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
            for (int c = 0; c < schema.Count; ++c)
            {
                DataViewType type = schema[c].Type;
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

            if (savable.Count < view.Schema.Count)
            {
                // Restrict the comparison to the subset of columns we were able to save.
                var chooseargs = new ChooseColumnsByIndexTransform.Options();
                chooseargs.Indices = savable.ToArray();
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
            for (int c = 0; c < schema.Count; ++c)
            {
                DataViewType type = schema[c].Type;
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

            if (savable.Count < view.Schema.Count)
            {
                // Restrict the comparison to the subset of columns we were able to save.
                var chooseargs = new ChooseColumnsByIndexTransform.Options();
                chooseargs.Indices = savable.ToArray();
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

        protected bool Failed()
        {
            Contracts.Assert(!IsPassing);
            return false;
        }

        protected bool EqualTypes(DataViewType type1, DataViewType type2, bool exactTypes)
        {
            Contracts.AssertValue(type1);
            Contracts.AssertValue(type2);

            return exactTypes ? type1.Equals(type2) : type1.SameSizeAndItemType(type2);
        }

        protected bool CheckSameValues(IDataView view1, IDataView view2, bool exactTypes = true, bool exactDoubles = true, bool checkId = true)
        {
            Contracts.Assert(view1.Schema.Count == view2.Schema.Count);

            bool all = true;
            bool tmp;

            using (var curs1 = view1.GetRowCursorForAllColumns())
            using (var curs2 = view2.GetRowCursorForAllColumns())
            {
                Check(curs1.Schema == view1.Schema, "Schema of view 1 and its cursor differed");
                Check(curs2.Schema == view2.Schema, "Schema of view 2 and its cursor differed");
                tmp = CheckSameValues(curs1, curs2, exactTypes, exactDoubles, checkId, true);
            }
            Check(tmp, "All same failed");
            all &= tmp;

            var view2EvenCols = view2.Schema.Where(col => (col.Index & 1) == 0); 
            using (var curs1 = view1.GetRowCursorForAllColumns())
            using (var curs2 = view2.GetRowCursor(view2EvenCols))
            {
                Check(curs1.Schema == view1.Schema, "Schema of view 1 and its cursor differed");
                Check(curs2.Schema == view2.Schema, "Schema of view 2 and its cursor differed");
                tmp = CheckSameValues(curs1, curs2, exactTypes, exactDoubles, checkId, false);
            }
            Check(tmp, "Even same failed");
            all &= tmp;

            var view2OddCols = view2.Schema.Where(col => (col.Index & 1) != 0);
            using (var curs1 = view1.GetRowCursorForAllColumns())
            using (var curs2 = view2.GetRowCursor(view2OddCols))
            {
                Check(curs1.Schema == view1.Schema, "Schema of view 1 and its cursor differed");
                Check(curs2.Schema == view2.Schema, "Schema of view 2 and its cursor differed");
                tmp = CheckSameValues(curs1, curs2, exactTypes, exactDoubles, checkId, false);
            }
            Check(tmp, "Odd same failed");

            using (var curs1 = view1.GetRowCursorForAllColumns())
            {
                Check(curs1.Schema == view1.Schema, "Schema of view 1 and its cursor differed");
                tmp = CheckSameValues(curs1, view2, exactTypes, exactDoubles, checkId);
            }
            Check(tmp, "Single value same failed");

            all &= tmp;
            return all;
        }

        protected bool CheckSameValues(DataViewRowCursor curs1, DataViewRowCursor curs2, bool exactTypes, bool exactDoubles, bool checkId, bool checkIdCollisions = true)
        {
            Contracts.Assert(curs1.Schema.Count == curs2.Schema.Count);

            // Get the comparison delegates for each column.
            int colLim = curs1.Schema.Count;
            Func<bool>[] comps = new Func<bool>[colLim];
            for (int col = 0; col < colLim; col++)
            {
                var f1 = curs1.IsColumnActive(curs1.Schema[col]);
                var f2 = curs2.IsColumnActive(curs2.Schema[col]);

                if (f1 && f2)
                {
                    var type1 = curs1.Schema[col].Type;
                    var type2 = curs2.Schema[col].Type;
                    if (!EqualTypes(type1, type2, exactTypes))
                    {
                        Fail($"Different types {type1} and {type2}");
                        return Failed();
                    }
                    comps[col] = GetColumnComparer(curs1, curs2, col, type1, exactDoubles);
                }
            }
            ValueGetter<DataViewRowId> idGetter = null;
            Func<bool> idComp = checkId ? GetIdComparer(curs1, curs2, out idGetter) : null;
            HashSet<DataViewRowId> idsSeen = null;
            if (checkIdCollisions && idGetter == null)
                idGetter = curs1.GetIdGetter();
            long idCollisions = 0;
            DataViewRowId id = default(DataViewRowId);

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

        protected bool CheckSameValues(DataViewRowCursor curs1, IDataView view2, bool exactTypes = true, bool exactDoubles = true, bool checkId = true)
        {
            Contracts.Assert(curs1.Schema.Count == view2.Schema.Count);

            // Get a cursor for each column.
            int colLim = curs1.Schema.Count;
            var cursors = new DataViewRowCursor[colLim];
            try
            {
                for (int col = 0; col < colLim; col++)
                {
                    // curs1 should have all columns active (for simplicity of the code here).
                    Contracts.Assert(curs1.IsColumnActive(curs1.Schema[col]));
                    cursors[col] = view2.GetRowCursorForAllColumns();
                }

                // Get the comparison delegates for each column.
                Func<bool>[] comps = new Func<bool>[colLim];
                // We have also one ID comparison delegate for each cursor.
                Func<bool>[] idComps = new Func<bool>[cursors.Length];
                for (int col = 0; col < colLim; col++)
                {
                    Contracts.Assert(cursors[col] != null);
                    var type1 = curs1.Schema[col].Type;
                    var type2 = cursors[col].Schema[col].Type;
                    if (!EqualTypes(type1, type2, exactTypes))
                    {
                        Fail("Different types");
                        return Failed();
                    }
                    comps[col] = GetColumnComparer(curs1, cursors[col], col, type1, exactDoubles);
                    ValueGetter<DataViewRowId> idGetter;
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

        protected Func<bool> GetIdComparer(DataViewRow r1, DataViewRow r2, out ValueGetter<DataViewRowId> idGetter)
        {
            var g1 = r1.GetIdGetter();
            idGetter = g1;
            var g2 = r2.GetIdGetter();
            DataViewRowId v1 = default(DataViewRowId);
            DataViewRowId v2 = default(DataViewRowId);
            return
                () =>
                {
                    g1(ref v1);
                    g2(ref v2);
                    return v1.Equals(v2);
                };
        }

        protected Func<bool> GetColumnComparer(DataViewRow r1, DataViewRow r2, int col, DataViewType type, bool exactDoubles)
        {
            if (!(type is VectorType vectorType))
            {
                Type rawType = type.RawType;
                if (rawType == typeof(sbyte))
                    return GetComparerOne<sbyte>(r1, r2, col, (x, y) => x == y);
                else if (rawType == typeof(byte))
                    return GetComparerOne<byte>(r1, r2, col, (x, y) => x == y);
                else if (rawType == typeof(short))
                    return GetComparerOne<short>(r1, r2, col, (x, y) => x == y);
                else if (rawType == typeof(ushort))
                    return GetComparerOne<ushort>(r1, r2, col, (x, y) => x == y);
                else if (rawType == typeof(int))
                    return GetComparerOne<int>(r1, r2, col, (x, y) => x == y);
                else if (rawType == typeof(uint))
                    return GetComparerOne<uint>(r1, r2, col, (x, y) => x == y);
                else if (rawType == typeof(long))
                    return GetComparerOne<long>(r1, r2, col, (x, y) => x == y);
                else if (rawType == typeof(ulong))
                    return GetComparerOne<ulong>(r1, r2, col, (x, y) => x == y);
                else if (rawType == typeof(float))
                    return GetComparerOne<float>(r1, r2, col, (x, y) => FloatUtils.GetBits(x) == FloatUtils.GetBits(y));
                else if (rawType == typeof(double))
                {
                    if (exactDoubles)
                        return GetComparerOne<double>(r1, r2, col, (x, y) => FloatUtils.GetBits(x) == FloatUtils.GetBits(y));
                    else
                        return GetComparerOne<double>(r1, r2, col, EqualWithEpsDouble);
                }
                else if (rawType == typeof(ReadOnlyMemory<char>))
                    return GetComparerOne<ReadOnlyMemory<char>>(r1, r2, col, (a, b) => a.Span.SequenceEqual(b.Span));
                else if (rawType == typeof(bool))
                    return GetComparerOne<bool>(r1, r2, col, (x, y) => x == y);
                else if (rawType == typeof(TimeSpan))
                    return GetComparerOne<TimeSpan>(r1, r2, col, (x, y) => x == y);
                else if (rawType == typeof(DateTime))
                    return GetComparerOne<DateTime>(r1, r2, col, (x, y) => x == y);
                else if (rawType == typeof(DateTimeOffset))
                    return GetComparerOne<DateTimeOffset>(r1, r2, col, (x, y) => x.Equals(y));
                else if (rawType == typeof(DataViewRowId))
                    return GetComparerOne<DataViewRowId>(r1, r2, col, (x, y) => x.Equals(y));
                else
                    return () => true;
            }
            else
            {
                int size = vectorType.Size;
                Contracts.Assert(size >= 0);
                Type itemType = vectorType.ItemType.RawType;

                if (itemType == typeof(sbyte))
                    return GetComparerVec<sbyte>(r1, r2, col, size, (x, y) => x == y);
                else if (itemType == typeof(byte))
                    return GetComparerVec<byte>(r1, r2, col, size, (x, y) => x == y);
                else if (itemType == typeof(short))
                    return GetComparerVec<short>(r1, r2, col, size, (x, y) => x == y);
                else if (itemType == typeof(ushort))
                    return GetComparerVec<ushort>(r1, r2, col, size, (x, y) => x == y);
                else if (itemType == typeof(int))
                    return GetComparerVec<int>(r1, r2, col, size, (x, y) => x == y);
                else if (itemType == typeof(uint))
                    return GetComparerVec<uint>(r1, r2, col, size, (x, y) => x == y);
                else if (itemType == typeof(long))
                    return GetComparerVec<long>(r1, r2, col, size, (x, y) => x == y);
                else if (itemType == typeof(ulong))
                    return GetComparerVec<ulong>(r1, r2, col, size, (x, y) => x == y);
                else if (itemType == typeof(float))
                {
                    if (exactDoubles)
                        return GetComparerVec<float>(r1, r2, col, size, (x, y) => FloatUtils.GetBits(x) == FloatUtils.GetBits(y));
                    else
                        return GetComparerVec<float>(r1, r2, col, size, EqualWithEpsSingle);
                }
                else if (itemType == typeof(double))
                {
                    if (exactDoubles)
                        return GetComparerVec<double>(r1, r2, col, size, (x, y) => FloatUtils.GetBits(x) == FloatUtils.GetBits(y));
                    else
                        return GetComparerVec<double>(r1, r2, col, size, EqualWithEpsDouble);
                }
                else if (itemType == typeof(ReadOnlyMemory<char>))
                    return GetComparerVec<ReadOnlyMemory<char>>(r1, r2, col, size, (a, b) => a.Span.SequenceEqual(b.Span));
                else if (itemType == typeof(bool))
                    return GetComparerVec<bool>(r1, r2, col, size, (x, y) => x == y);
                else if (itemType == typeof(TimeSpan))
                    return GetComparerVec<TimeSpan>(r1, r2, col, size, (x, y) => x == y);
                else if (itemType == typeof(DateTime))
                    return GetComparerVec<DateTime>(r1, r2, col, size, (x, y) => x == y);
                else if (itemType == typeof(DateTimeOffset))
                    return GetComparerVec<DateTimeOffset>(r1, r2, col, size, (x, y) => x.Equals(y));
                else if (itemType == typeof(DataViewRowId))
                    return GetComparerVec<DataViewRowId>(r1, r2, col, size, (x, y) => x.Equals(y));
            }

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

        protected Func<bool> GetComparerOne<T>(DataViewRow r1, DataViewRow r2, int col, Func<T, T, bool> fn)
        {
            var g1 = r1.GetGetter<T>(r1.Schema[col]);
            var g2 = r2.GetGetter<T>(r2.Schema[col]);
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

        protected Func<bool> GetComparerVec<T>(DataViewRow r1, DataViewRow r2, int col, int size, Func<T, T, bool> fn)
        {
            var g1 = r1.GetGetter<VBuffer<T>>(r1.Schema[col]);
            var g2 = r2.GetGetter<VBuffer<T>>(r2.Schema[col]);
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

            var v1Values = v1.GetValues();
            var v2Values = v2.GetValues();

            if (v1.IsDense && v2.IsDense)
            {
                for (int i = 0; i < v1.Length; i++)
                {
                    var x1 = v1Values[i];
                    var x2 = v2Values[i];
                    if (!fn(i, x1, x2))
                        return false;
                }
                return true;
            }

            var v1Indices = v1.GetIndices();
            var v2Indices = v2.GetIndices();

            Contracts.Assert(!v1.IsDense || !v2.IsDense);
            int iiv1 = 0;
            int iiv2 = 0;
            for (; ; )
            {
                int iv1 = v1.IsDense ? iiv1 : iiv1 < v1Indices.Length ? v1Indices[iiv1] : v1.Length;
                int iv2 = v2.IsDense ? iiv2 : iiv2 < v2Indices.Length ? v2Indices[iiv2] : v2.Length;
                T x1, x2;
                int iv;
                if (iv1 == iv2)
                {
                    if (iv1 == v1.Length)
                        return true;
                    x1 = v1Values[iiv1];
                    x2 = v2Values[iiv2];
                    iv = iv1;
                    iiv1++;
                    iiv2++;
                }
                else if (iv1 < iv2)
                {
                    x1 = v1Values[iiv1];
                    x2 = default(T);
                    iv = iv1;
                    iiv1++;
                }
                else
                {
                    x1 = default(T);
                    x2 = v2Values[iiv2];
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
    }
}
