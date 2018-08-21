using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data.StaticPipe.Runtime;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data.StaticPipe
{
    /// <summary>
    /// Utility methods for components that want to expose themselves in the idioms of the statically-typed pipelines.
    /// These utilities are meant to be called by and useful to component authors, not users of those components.
    /// </summary>
    public static class PigstyUtils
    {
        /// <summary>
        /// This is a utility method intended to be used by authors of <see cref="IDataReaderEstimator{TSource,
        /// TReader}"/> components to provide a strongly typed <see cref="DataReaderEstimator{TIn, TTupleShape}"/>.
        /// This analysis tool provides a standard way for readers to exploit statically typed pipelines with the
        /// standard tuple-shape objects without having to write such code themselves.
        /// </summary>
        /// <param name="input">The input that will be used when invoking <paramref name="mapper"/>, which is used
        /// either to produce the input columns.</param>
        /// <param name="baseReconciler">All columns that are yielded by <paramref name="input"/> should produce this
        /// single reconciler. The analysis code in this method will ensure that this is the first object to be
        /// reconciled, before all others.</param>
        /// <param name="mapper">The user provided delegate.</param>
        /// <typeparam name="TReaderEstimatorInputType">The type parameter for the input type to the data reader
        /// estimator.</typeparam>
        /// <typeparam name="TDelegateInput">The input type of the input delegate. This might be some object out of
        /// which one can fetch or else retrieve </typeparam>
        /// <typeparam name="TTupleOutShape"></typeparam>
        /// <returns></returns>
        public static DataReaderEstimator<TReaderEstimatorInputType, TTupleOutShape>
            HelpMe<TReaderEstimatorInputType, TDelegateInput, TTupleOutShape>(
            TDelegateInput input,
            ReaderReconciler<TReaderEstimatorInputType> baseReconciler,
            Func<TDelegateInput, TTupleOutShape> mapper)
        {
            Contracts.CheckValue(mapper, nameof(mapper));

            var method = mapper.Method;
            var output = mapper(input);

            KeyValuePair<string, PipelineColumn>[] outPairs = PipelineColumnAnalyzer.GetNames(output, method.ReturnParameter);

            // Map where the key depends on the set of things in the value. The value contains the yet unresolved dependencies.
            var keyDependsOn = new Dictionary<PipelineColumn, HashSet<PipelineColumn>>();
            // Map where the set of things in the value depend on the key.
            var dependsOnKey = new Dictionary<PipelineColumn, HashSet<PipelineColumn>>();
            // The set of columns detected with zero dependencies.
            var zeroDependencies = new List<PipelineColumn>();

            // First we build up the two structures above, using a queue and visiting from the outputs up.
            var toVisit = new Queue<PipelineColumn>(outPairs.Select(p => p.Value));
            while (toVisit.Count > 0)
            {
                var col = toVisit.Dequeue();
                Contracts.CheckParam(col != null, nameof(mapper), "The delegate seems to have null columns returned somewhere in the pipe");
                if (keyDependsOn.ContainsKey(col))
                    continue; // Already visited.

                var dependsOn = new HashSet<PipelineColumn>();
                foreach (var dep in col.Dependencies ?? Enumerable.Empty<PipelineColumn>())
                {
                    dependsOn.Add(dep);
                    if (!dependsOnKey.TryGetValue(dep, out var dependsOnDep))
                    {
                        dependsOnKey[dep] = dependsOnDep = new HashSet<PipelineColumn>();
                        toVisit.Enqueue(dep);
                    }
                    dependsOnDep.Add(col);
                }
                keyDependsOn[col] = dependsOn;
                if (dependsOn.Count == 0)
                    zeroDependencies.Add(col);
            }

            // Get the base input columns.
            var baseInputs = keyDependsOn.Select(p => p.Key).Where(col => col.ReconcilerObj == baseReconciler).ToArray();

            // The columns that utilize the base reconciler should have no dependencies. This could only happen if
            // the caller of this function has introduced a situation whereby they are claiming they can reconcile
            // to a data-reader object but still have input data dependencies, which does not make sense and
            // indicates that there is a bug in that component code. Unfortunately we can only detect that condition,
            // not determine exactly how it arose, but we can still do so to indicate to the user that there is a
            // problem somewhere in the stack.
            Contracts.CheckParam(baseInputs.All(col => keyDependsOn[col].Count == 0),
                nameof(input), "Bug detected where column producing object was yielding columns with dependencies.");

            // This holds the mappings of columns to names and back. Note that while the same column could be used on
            // the *output*, e.g., you could hypothetically have `(a: r.Foo, b: r.Foo)`, we treat that as the last thing
            // that is done.
            var nameMap = new InvDictionary<string, PipelineColumn>();

            // REVIEW: Need to generalize case where we have input names, e.g., in the below method.

            int tempNum = 0;
            // For all outputs, get potential name collisions with used inputs. Resolve by assigning the input a temporary name.
            foreach (var p in outPairs)
            {
                // TODO: This should be accompanied by an actual CopyColumns estimator, once one exists!! However in the
                // current "fake" world this does not yet exist.

                // If the name for the output is already used by one of the inputs, and this output column does not
                // happen to have the same name, then we need to rename that input to keep it available.
                if (nameMap.TryGetValue(p.Key, out var inputCol) && p.Value != inputCol)
                {
                    Contracts.Assert(inputCol is PipelineColumnAnalyzer.IIsAnalysisColumn);
                    string tempName = $"#Temp_{tempNum++}";
                    Console.WriteLine($"Input/output name collision: Renaming '{p.Key}' to '{tempName}'");
                    nameMap[tempName] = nameMap[p.Key];
                    Contracts.Assert(!nameMap.ContainsKey(p.Key));
                }
                // If we already have a name for this output column, maybe it is used elsewhere. (This can happen when
                // the only thing done with an input is we rename it, or output it twice, or something like this.) In
                // this case it is most appropriate to delay renaming till after all other processing has been done in
                // that case. But otherwise we may as well just take the name.
                if (!nameMap.ContainsKey(p.Value))
                    nameMap[p.Key] = p.Value;
            }

            // First clear the inputs from zero-dependencies yet to be resolved.
            foreach (var col in baseInputs)
            {
                Contracts.Assert(zeroDependencies.Contains(col));
                Contracts.Assert(col.ReconcilerObj == baseReconciler);

                zeroDependencies.Remove(col); // Make more efficient...
                if (!dependsOnKey.TryGetValue(col, out var depends))
                    continue;
                // If any of these base inputs do not have names because, for example, they do not directly appear
                // in the outputs and otherwise do not have names, assign them a name.
                if (!nameMap.ContainsKey(col))
                    nameMap[col] = $"Temp_{tempNum++}";

                foreach (var depender in depends)
                {
                    var dependencies = keyDependsOn[depender];
                    Contracts.Assert(dependencies.Contains(col));
                    dependencies.Remove(col);
                    if (dependencies.Count == 0)
                        zeroDependencies.Add(depender);
                }
                dependsOnKey.Remove(col);
            }

            // Call the reconciler to get the base reader estimator.
            var readerEstimator = baseReconciler.Reconcile(baseInputs, nameMap.AsOther(baseInputs));

            // Next we iteratively find those columns with zero dependencies, "create" them, and if anything depends on
            // these add them to the collection of zero dependencies, etc. etc.
            while (zeroDependencies.Count > 0)
            {
                // All columns with the same reconciler can be transformed together.

                // Note that the following policy of just taking the first group is not optimal. So for example, we
                // could have three columns, (a, b, c). If we had the output (a.X(), b.X() c.Y().X()), then maybe we'd
                // reconcile a.X() and b.X() together, then reconcile c.Y(), then reconcile c.Y().X() alone. Whereas, we
                // could have reconciled c.Y() first, then reconciled a.X(), b.X(), and c.Y().X() together.
                var group = zeroDependencies.GroupBy(p => p.ReconcilerObj).First();
                DataInputReconciler rec = (DataInputReconciler)group.Key;
                PipelineColumn[] cols = group.ToArray();
                // All dependencies should, by this time, have names.
                Contracts.Assert(cols.SelectMany(c => c.Dependencies).All(dep => nameMap.ContainsKey(dep)));
                foreach (var newCol in cols)
                {
                    if (!nameMap.ContainsKey(newCol))
                        nameMap[newCol] = $"#Temp_{tempNum++}";

                }

                var localInputNames = nameMap.AsOther(cols.SelectMany(c => c.Dependencies ?? Enumerable.Empty<PipelineColumn>()));
                var localOutputNames = nameMap.AsOther(cols);
                var result = rec.Reconcile(cols, localInputNames, localOutputNames);

                foreach (var newCol in cols)
                {
                    zeroDependencies.Remove(newCol); // Make more efficient!!

                    // Finally, we find all columns that depend on this one. If this happened to be the last pending
                    // dependency, then we add it to the list.
                    if (dependsOnKey.TryGetValue(newCol, out var depends))
                    {
                        foreach (var depender in depends)
                        {
                            var dependencies = keyDependsOn[depender];
                            Contracts.Assert(dependencies.Contains(newCol));
                            dependencies.Remove(newCol);
                            if (dependencies.Count == 0)
                                zeroDependencies.Add(depender);
                        }
                        dependsOnKey.Remove(newCol);
                    }
                }
            }

            if (keyDependsOn.Any(p => p.Value.Count > 0))
            {
                // This might happen if the user does something incredibly strange, like, say, take some prior
                // lambda, assign a column to a local variable, then re-use it downstream in a different lambdas.
                // The user would have to go to some extraorindary effort to do that, but nonetheless we want to
                // fail with a semi-sensible error message.
                throw Contracts.Except("There were some leftover columns with unresolved dependencies. " +
                    "Did the caller use a " + nameof(PipelineColumn) + " from another lambda?");
            }

            // Now do the final renaming, if any is necessary.
            foreach (var p in outPairs)
            {
                // TODO: Right now we just write stuff out. Once the copy-columns estimator is in place
                // we ought to do this for real.
                Contracts.Assert(nameMap.ContainsKey(p.Value));
                string currentName = nameMap[p.Value];
                if (currentName != p.Key)
                    Console.WriteLine($"Will copy '{currentName}' to '{p.Key}'");
            }

            Console.WriteLine($"Exiting {nameof(HelpMe)} !!!");

            return null;// new FakeEstimator<TTupleOutShape>();
        }
    }
}
