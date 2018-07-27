using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Core.StrongPipe.Columns;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Core.StrongPipe
{
    public abstract class BlockMaker<TTupleShape>
    {
        public Estimator<TTupleShape, TTupleOutShape, ITransformer> CreateTransform<TTupleOutShape>(Func<TTupleShape, TTupleOutShape> mapper)
        {
            Contracts.CheckValue(mapper, nameof(mapper));

            Console.WriteLine($"Called {nameof(CreateTransform)} !!!");

            var method = mapper.Method;

            // Construct the dummy column structure, then apply the mapping.
            var input = PipelineColumnAnalyzer.CreateAnalysisInstance<TTupleShape>();
            var output = mapper(input);

            // Extract the name/value pairs out of both the input and output.
            KeyValuePair<string, PipelineColumn>[] inPairs = PipelineColumnAnalyzer.GetNames(input, method.GetParameters()[0]);
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

            // The input columns should have no dependencies.
            Contracts.Assert(inPairs.All(p => !keyDependsOn.TryGetValue(p.Value, out var deps) || deps.Count == 0));

            // This holds the mappings of columns to names and back. Note that while the same column could be used on
            // the *output*, e.g., you could hypothetically have `(a: r.Foo, b: r.Foo)`, we treat that as the last thing
            // that is done.
            var nameMap = new InvDictionary<string, PipelineColumn>();

            // Initially we suppose we've only assigned names to the inputs.
            var inputColToName = new Dictionary<PipelineColumn, string>();
            foreach (var p in inPairs)
                inputColToName[p.Value] = p.Key;

            // Get the initial name map.
            foreach (var col in zeroDependencies)
            {
                if (inputColToName.TryGetValue(col, out string inputName))
                {
                    Contracts.Assert(!nameMap.ContainsKey(col));
                    Contracts.Assert(!nameMap.ContainsKey(inputName));
                    nameMap[col] = inputName;

                    Console.WriteLine($"Using input with name {inputName}");
                }
            }

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
            foreach (var p in inPairs)
            {
                zeroDependencies.Remove(p.Value); // Make more efficient...
                if (!dependsOnKey.TryGetValue(p.Value, out var depends))
                    continue;
                Contracts.Assert(nameMap.ContainsKey(p.Value));
                foreach (var depender in depends)
                {
                    var dependencies = keyDependsOn[depender];
                    Contracts.Assert(dependencies.Contains(p.Value));
                    dependencies.Remove(p.Value);
                    if (dependencies.Count == 0)
                        zeroDependencies.Add(depender);
                }
                dependsOnKey.Remove(p.Value);
            }

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
                // This might happen if the user does something incredibly strange, like, say, take one of the prior
                // lambdas, assign it to a local variable, then re-use it downstream in a different lambdas. The user
                // would have to be trying to break the system.
                throw Contracts.Except("There were some leftover columns with unresolved dependencies. " +
                    "Did you use a " + nameof(PipelineColumn) + " from another lambda?");
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

            Console.WriteLine($"Exiting {nameof(CreateTransform)} !!!");

            return new FakeEstimator<TTupleOutShape>();
        }

        private sealed class FakeEstimator<TTupleOutShape>
            : Estimator<TTupleShape, TTupleOutShape, ITransformer>
        {
            protected override ITransformer FitCore(IDataView input)
            {
                throw new NotImplementedException();
            }

            protected override SchemaShape GetOutputSchemaCore(SchemaShape inputSchema)
            {
                throw new NotImplementedException();
            }
        }

        private sealed class InvDictionary<T1, T2>
        {
            private readonly Dictionary<T1, T2> _d12;
            private readonly Dictionary<T2, T1> _d21;

            public InvDictionary()
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
                set {
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
                set {
                    Contracts.CheckValue((object)key, nameof(key));
                    Contracts.CheckValue((object)value, nameof(value));

                    bool removeOldKey = _d21.TryGetValue(value, out var oldKey);
                    if (_d12.TryGetValue(key, out var oldValue))
                        _d21.Remove(oldValue);
                    if (removeOldKey)
                        _d12.Remove(oldKey);

                    _d21[value] = key;
                    _d12[key] = value;
                    if (_d12.Count != _d21.Count)
                        Console.WriteLine("Whoops?");

                    Contracts.Assert(_d12.Count == _d21.Count);
                }
            }

            public Dictionary<T1, T2> AsOther(IEnumerable<T1> keys)
            {
                Dictionary<T1, T2> d = new Dictionary<T1, T2>();
                foreach (var v in keys)
                    d[v] = _d12[v];
                return d;
            }

            public Dictionary<T2, T1> AsOther(IEnumerable<T2> keys)
            {
                Dictionary<T2, T1> d = new Dictionary<T2, T1>();
                foreach (var v in keys)
                    d[v] = _d21[v];
                return d;
            }
        }

    }
}
