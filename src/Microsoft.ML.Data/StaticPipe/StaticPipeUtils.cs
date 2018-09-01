// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Data.StaticPipe.Runtime
{
    /// <summary>
    /// Utility methods for components that want to expose themselves in the idioms of the statically-typed pipelines.
    /// These utilities are meant to be called by and useful to component authors, not users of those components.
    /// </summary>
    public static class StaticPipeUtils
    {
        /// <summary>
        /// This is a utility method intended to be used by authors of <see cref="IDataReaderEstimator{TSource,
        /// TReader}"/> components to provide a strongly typed <see cref="DataReaderEstimator{TIn, TTupleShape, TDataReader}"/>.
        /// This analysis tool provides a standard way for readers to exploit statically typed pipelines with the
        /// standard tuple-shape objects without having to write such code themselves.
        /// </summary>
        /// <param name="env">Estimators will be instantiated with this environment</param>
        /// /// <param name="ch">Some minor debugging information will be passed along to this channel</param>
        /// <param name="input">The input that will be used when invoking <paramref name="mapper"/>, which is used
        /// either to produce the input columns.</param>
        /// <param name="baseReconciler">All columns that are yielded by <paramref name="input"/> should produce this
        /// single reconciler. The analysis code in this method will ensure that this is the first object to be
        /// reconciled, before all others.</param>
        /// <param name="mapper">The user provided delegate.</param>
        /// <typeparam name="TIn">The type parameter for the input type to the data reader estimator.</typeparam>
        /// <typeparam name="TDelegateInput">The input type of the input delegate. This might be some object out of
        /// which one can fetch or else retrieve </typeparam>
        /// <typeparam name="TTupleOutShape"></typeparam>
        /// <returns></returns>
        public static DataReaderEstimator<TIn, TTupleOutShape, IDataReader<TIn>>
            ReaderEstimatorAnalyzerHelper<TIn, TDelegateInput, TTupleOutShape>(
            IHostEnvironment env,
            IChannel ch,
            TDelegateInput input,
            ReaderReconciler<TIn> baseReconciler,
            Func<TDelegateInput, TTupleOutShape> mapper)
        {
            var readerEstimator = GeneralFunctionAnalyzer(env, ch, input, baseReconciler, mapper, out var est, col => null);
            var schema = StaticSchemaShape.Make<TTupleOutShape>(mapper.Method.ReturnParameter);
            return new DataReaderEstimator<TIn, TTupleOutShape, IDataReader<TIn>>(env, readerEstimator, schema);
        }

        internal static IDataReaderEstimator<TIn, IDataReader<TIn>>
            GeneralFunctionAnalyzer<TIn, TDelegateInput, TTupleOutShape>(
            IHostEnvironment env,
            IChannel ch,
            TDelegateInput input,
            ReaderReconciler<TIn> baseReconciler,
            Func<TDelegateInput, TTupleOutShape> mapper,
            out IEstimator<ITransformer> estimator,
            Func<PipelineColumn, string> inputNameFunction)
        {
            Contracts.CheckValue(mapper, nameof(mapper));

            var method = mapper.Method;
            var output = mapper(input);

            KeyValuePair<string, PipelineColumn>[] outPairs = StaticPipeInternalUtils.GetNamesValues(output, method.ReturnParameter);

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
                ch.CheckParam(col != null, nameof(mapper), "The delegate seems to have null columns returned somewhere in the pipe");
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
            ch.CheckParam(baseInputs.All(col => keyDependsOn[col].Count == 0),
                nameof(input), "Bug detected where column producing object was yielding columns with dependencies.");

            // This holds the mappings of columns to names and back. Note that while the same column could be used on
            // the *output*, e.g., you could hypothetically have `(a: r.Foo, b: r.Foo)`, we treat that as the last thing
            // that is done.
            var nameMap = new BidirectionalDictionary<string, PipelineColumn>();

            // Check to see if we have any set of initial names. This is important in the case where we are mapping
            // in an input data view.
            foreach (var col in baseInputs)
            {
                string inputName = inputNameFunction(col);
                if (inputName != null)
                {
                    ch.Assert(!nameMap.ContainsKey(col));
                    ch.Assert(!nameMap.ContainsKey(inputName));
                    nameMap[col] = inputName;

                    ch.Trace($"Using input with name {inputName}");
                }
            }

            estimator = null;
            var toCopy = new List<(string src, string dst)>();

            int tempNum = 0;
            // For all outputs, get potential name collisions with used inputs. Resolve by assigning the input a temporary name.
            foreach (var p in outPairs)
            {
                // If the name for the output is already used by one of the inputs, and this output column does not
                // happen to have the same name, then we need to rename that input to keep it available.
                if (nameMap.TryGetValue(p.Key, out var inputCol) && p.Value != inputCol)
                {
                    ch.Assert(baseInputs.Contains(inputCol));
                    string tempName = $"#Temp_{tempNum++}";
                    ch.Trace($"Input/output name collision: Renaming '{p.Key}' to '{tempName}'");
                    toCopy.Add((p.Key, tempName));
                    nameMap[tempName] = nameMap[p.Key];
                    ch.Assert(!nameMap.ContainsKey(p.Key));
                }
                // If we already have a name for this output column, maybe it is used elsewhere. (This can happen when
                // the only thing done with an input is we rename it, or output it twice, or something like this.) In
                // this case it is most appropriate to delay renaming till after all other processing has been done in
                // that case. But otherwise we may as well just take the name.
                if (!nameMap.ContainsKey(p.Value))
                    nameMap[p.Key] = p.Value;
            }

            // If any renamings were necessary, create the CopyColumns estimator.
            if (toCopy.Count > 0)
                estimator = new CopyColumnsEstimator(env, toCopy.ToArray());

            // First clear the inputs from zero-dependencies yet to be resolved.
            foreach (var col in baseInputs)
            {
                ch.Assert(zeroDependencies.Contains(col));
                ch.Assert(col.ReconcilerObj == baseReconciler);

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
                    ch.Assert(dependencies.Contains(col));
                    dependencies.Remove(col);
                    if (dependencies.Count == 0)
                        zeroDependencies.Add(depender);
                }
                dependsOnKey.Remove(col);
            }

            // Call the reconciler to get the base reader estimator.
            var readerEstimator = baseReconciler.Reconcile(env, baseInputs, nameMap.AsOther(baseInputs));
            ch.AssertValueOrNull(readerEstimator);

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
                // Beyond that first group that *might* be a data reader reconciler, all subsequent operations will
                // be on where the data is already loaded and so accept data as an input, that is, they should produce
                // an estimator. If this is not the case something seriously wonky is going on, most probably that the
                // user tried to use a column from another source. If this is detected we can produce a sensible error
                // message to tell them not to do this.
                if (!(group.Key is EstimatorReconciler rec))
                {
                    throw ch.Except("Columns from multiple sources were detected. " +
                        "Did the caller use a " + nameof(PipelineColumn) + " from another delegate?");
                }
                PipelineColumn[] cols = group.ToArray();
                // All dependencies should, by this time, have names.
                ch.Assert(cols.SelectMany(c => c.Dependencies).All(dep => nameMap.ContainsKey(dep)));
                foreach (var newCol in cols)
                {
                    if (!nameMap.ContainsKey(newCol))
                        nameMap[newCol] = $"#Temp_{tempNum++}";

                }

                var localInputNames = nameMap.AsOther(cols.SelectMany(c => c.Dependencies ?? Enumerable.Empty<PipelineColumn>()));
                var localOutputNames = nameMap.AsOther(cols);
                var localEstimator = rec.Reconcile(env, cols, localInputNames, localOutputNames);
                readerEstimator = readerEstimator?.Append(localEstimator);
                estimator = estimator?.Append(localEstimator) ?? localEstimator;

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
                throw ch.Except("There were some leftover columns with unresolved dependencies. " +
                    "Did the caller use a " + nameof(PipelineColumn) + " from another delegate?");
            }

            // Now do the final renaming, if any is necessary.
            toCopy.Clear();
            foreach (var p in outPairs)
            {
                // TODO: Right now we just write stuff out. Once the copy-columns estimator is in place
                // we ought to do this for real.
                Contracts.Assert(nameMap.ContainsKey(p.Value));
                string currentName = nameMap[p.Value];
                if (currentName != p.Key)
                {
                    ch.Trace($"Will copy '{currentName}' to '{p.Key}'");
                    toCopy.Add((currentName, p.Key));
                }
            }

            // If any final renamings were necessary, insert the appropriate CopyColumns transform.
            if (toCopy.Count > 0)
            {
                var copyEstimator = new CopyColumnsEstimator(env, toCopy.ToArray());
                if (estimator == null)
                    estimator = copyEstimator;
                else
                    estimator = estimator.Append(copyEstimator);
            }

            ch.Trace($"Exiting {nameof(ReaderEstimatorAnalyzerHelper)}");

            return readerEstimator;
        }

        private sealed class BidirectionalDictionary<T1, T2>
        {
            private readonly Dictionary<T1, T2> _d12;
            private readonly Dictionary<T2, T1> _d21;

            public BidirectionalDictionary()
            {
                _d12 = new Dictionary<T1, T2>();
                _d21 = new Dictionary<T2, T1>();
            }

            public bool ContainsKey(T1 k) => _d12.ContainsKey(k);
            public bool ContainsKey(T2 k) => _d21.ContainsKey(k);

            public bool TryGetValue(T1 k, out T2 v) => _d12.TryGetValue(k, out v);
            public bool TryGetValue(T2 k, out T1 v) => _d21.TryGetValue(k, out v);

            public T1 this[T2 key]
            {
                get => _d21[key];
                set
                {
                    Contracts.CheckValue((object)key, nameof(key));
                    Contracts.CheckValue((object)value, nameof(value));

                    bool removeOldKey = _d12.TryGetValue(value, out var oldKey);
                    if (_d21.TryGetValue(key, out var oldValue))
                        _d12.Remove(oldValue);
                    if (removeOldKey)
                        _d21.Remove(oldKey);

                    _d12[value] = key;
                    _d21[key] = value;
                    Contracts.Assert(_d12.Count == _d21.Count);
                }
            }

            public T2 this[T1 key]
            {
                get => _d12[key];
                set
                {
                    Contracts.CheckValue((object)key, nameof(key));
                    Contracts.CheckValue((object)value, nameof(value));

                    bool removeOldKey = _d21.TryGetValue(value, out var oldKey);
                    if (_d12.TryGetValue(key, out var oldValue))
                        _d21.Remove(oldValue);
                    if (removeOldKey)
                        _d12.Remove(oldKey);

                    _d21[value] = key;
                    _d12[key] = value;

                    Contracts.Assert(_d12.Count == _d21.Count);
                }
            }

            public IReadOnlyDictionary<T1, T2> AsOther(IEnumerable<T1> keys)
            {
                Dictionary<T1, T2> d = new Dictionary<T1, T2>();
                foreach (var v in keys)
                    d[v] = _d12[v];
                return d;
            }

            public IReadOnlyDictionary<T2, T1> AsOther(IEnumerable<T2> keys)
            {
                Dictionary<T2, T1> d = new Dictionary<T2, T1>();
                foreach (var v in keys)
                    d[v] = _d21[v];
                return d;
            }
        }
    }
}
