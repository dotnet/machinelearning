using Microsoft.ML.Core.Data;
using Microsoft.ML.Core.StrongPipe;
using Microsoft.ML.Core.StrongPipe.Columns;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

// Holds some classes that superficially represent classes, at least sufficiently to give the idea of the statically typed columnar estimator helper API.

namespace FakeStrongPipes
{
    public sealed class TextLoader
    {
        private sealed class Estimator : IDataReaderEstimator<IMultiStreamSource, IDataReader<IMultiStreamSource>>
        {
            public IDataReader<IMultiStreamSource> Fit(IMultiStreamSource input)
                => throw new NotImplementedException();

            public SchemaShape GetOutputSchema()
                => throw new NotImplementedException();
        }

        private sealed class TextReconciler : Reconciler
        {
            public static readonly TextReconciler Inst = new TextReconciler();
        }

        public sealed class Context
        {
            private class MyScalar<T> : Scalar<T>
            {
                public readonly int Ordinal;

                public MyScalar(int ordinal)
                    : base(TextReconciler.Inst, null)
                {
                    Ordinal = ordinal;
                }
            }

            private class MyVector<T> : Vector<T>
            {
                public readonly int? Min;
                public readonly int? Max;

                public MyVector(int? min, int? max)
                    : base(TextReconciler.Inst, null)
                {
                    Min = min;
                    Max = max;
                }
            }

            public Scalar<bool> LoadBool(int ordinal) => Load<bool>(ordinal);
            public Vector<bool> LoadBool(int minOrdinal, int? maxOrdinal) => Load<bool>(minOrdinal, maxOrdinal);
            public Scalar<float> LoadFloat(int ordinal) => Load<float>(ordinal);
            public Vector<float> LoadFloat(int minOrdinal, int? maxOrdinal) => Load<float>(minOrdinal, maxOrdinal);
            public Scalar<double> LoadDouble(int ordinal) => Load<double>(ordinal);
            public Vector<double> LoadDouble(int minOrdinal, int? maxOrdinal) => Load<double>(minOrdinal, maxOrdinal);
            public Scalar<string> LoadText(int ordinal) => Load<string>(ordinal);
            public Vector<string> LoadText(int minOrdinal, int? maxOrdinal) => Load<string>(minOrdinal, maxOrdinal);


            private Scalar<T> Load<T>(int ordinal)
            {
                Contracts.CheckParam(ordinal >= 0, nameof(ordinal), "Should be non-negative");
                return new MyScalar<T>(ordinal);
            }

            private Vector<T> Load<T>(int minOrdinal, int? maxOrdinal)
            {
                Contracts.CheckParam(minOrdinal >= 0, nameof(minOrdinal), "Should be non-negative");
                var v = maxOrdinal >= minOrdinal;
                Contracts.CheckParam(!(maxOrdinal < minOrdinal), nameof(maxOrdinal), "If specified, cannot be less than " + nameof(minOrdinal));
                return new MyVector<T>(minOrdinal, maxOrdinal);
            }
        }

        public static DataReaderEstimator<IMultiStreamSource, TTupleShape> Create<TTupleShape>(Func<Context, TTupleShape> map)
        {
            var meth = map.Method;
            var p = meth.GetParameters()[0];

            return new TypedEstimator<TTupleShape>(null);
        }

        private sealed class TypedEstimator<TTupleShape> : DataReaderEstimator<IMultiStreamSource, TTupleShape>
        {
            private readonly Estimator _estimator;

            public TypedEstimator(Estimator estimator)
            {
                _estimator = estimator;
            }

            protected override IDataReader<IMultiStreamSource> FitCore(IMultiStreamSource input)
                => _estimator.Fit(input);

            protected override SchemaShape GetOutputSchemaCore()
                => _estimator.GetOutputSchema();
        }
    }

    /// <summary>
    /// This is a reconciler that doesn't really do anything, just a fake for testing the infrastructure.
    /// </summary>
    internal sealed class FakeTransformReconciler : DataInputReconciler
    {
        private readonly string _name;

        public FakeTransformReconciler(string name)
        {
            _name = name;
        }

        public override IEstimator<ITransformer> Reconcile(
            PipelineColumn[] toOutput,
            Dictionary<PipelineColumn, string> inputNames,
            Dictionary<PipelineColumn, string> outputNames)
        {
            Console.WriteLine($"Constructing {_name} estimator!");

            foreach (var col in toOutput)
            {
                if ((((IDeps)col).Deps?.Length ?? 0) == 0)
                    Console.WriteLine($"    Will make '{outputNames[col]}' from nothing");
                else
                {
                    Console.WriteLine($"    Will make '{outputNames[col]}' out of " +
                        string.Join(", ", ((IDeps)col).Deps.Select(d => $"'{inputNames[d]}'")));
                }
            }

            return new FakeEstimator();
        }

        private sealed class FakeEstimator : IEstimator<ITransformer>
        {
            public ITransformer Fit(IDataView input) => throw new NotImplementedException();
            public SchemaShape GetOutputSchema(SchemaShape inputSchema) => throw new NotImplementedException();
        }

        private interface IDeps { PipelineColumn[] Deps { get; } }

        private sealed class AScalar<T> : Scalar<T>, IDeps { public AScalar(Reconciler rec, PipelineColumn[] dependencies) : base(rec, dependencies) { Deps = dependencies; } public PipelineColumn[] Deps { get; } }
        private sealed class AVector<T> : Vector<T>, IDeps { public AVector(Reconciler rec, PipelineColumn[] dependencies) : base(rec, dependencies) { Deps = dependencies; } public PipelineColumn[] Deps { get; } }
        private sealed class AVarVector<T> : VarVector<T>, IDeps { public AVarVector(Reconciler rec, PipelineColumn[] dependencies) : base(rec, dependencies) { Deps = dependencies; } public PipelineColumn[] Deps { get; } }
        private sealed class AKey<T> : Key<T>, IDeps { public AKey(Reconciler rec, PipelineColumn[] dependencies) : base(rec, dependencies) { Deps = dependencies; } public PipelineColumn[] Deps { get; } }
        private sealed class AKey<T, TV> : Key<T, TV>, IDeps { public AKey(Reconciler rec, PipelineColumn[] dependencies) : base(rec, dependencies) { Deps = dependencies; } public PipelineColumn[] Deps { get; } }
        private sealed class AVarKey<T> : VarKey<T>, IDeps { public AVarKey(Reconciler rec, PipelineColumn[] dependencies) : base(rec, dependencies) { Deps = dependencies; } public PipelineColumn[] Deps { get; } }

    public Scalar<T> Scalar<T>(params PipelineColumn[] dependencies) => new AScalar<T>(this, dependencies);
        public Vector<T> Vector<T>(params PipelineColumn[] dependencies) => new AVector<T>(this, dependencies);
        public VarVector<T> VarVector<T>(params PipelineColumn[] dependencies) => new AVarVector<T>(this, dependencies);
        public Key<T> Key<T>(params PipelineColumn[] dependencies) => new AKey<T>(this, dependencies);
        public Key<T, TV> Key<T, TV>(params PipelineColumn[] dependencies) => new AKey<T, TV>(this, dependencies);
        public VarKey<T> VarKey<T>(params PipelineColumn[] dependencies) => new AVarKey<T>(this, dependencies);
    }

    public static class ConcatTransformExtensions
    {
        private static FakeTransformReconciler _rec = new FakeTransformReconciler("Concat");

        public sealed class ScalarOrVector<T> : ScalarOrVectorOrVarVector<T>
        {
            private ScalarOrVector(PipelineColumn col) : base(col) { }
            public static implicit operator ScalarOrVector<T>(Scalar<T> c) => new ScalarOrVector<T>(c);
            public static implicit operator ScalarOrVector<T>(Vector<T> c) => new ScalarOrVector<T>(c);
        }

        private interface IContainsColumn
        {
            PipelineColumn WrappedColumn { get; }
        }


        public class ScalarOrVectorOrVarVector<T> : IContainsColumn
        {
            private readonly PipelineColumn _wrappedColumn;
            PipelineColumn IContainsColumn.WrappedColumn => _wrappedColumn;

            private protected ScalarOrVectorOrVarVector(PipelineColumn col)
            {
                _wrappedColumn = col;
            }

            public static implicit operator ScalarOrVectorOrVarVector<T>(VarVector<T> c)
               => new ScalarOrVectorOrVarVector<T>(c);
        }

        private static PipelineColumn[] Helper<T>(PipelineColumn first, IList<ScalarOrVectorOrVarVector<T>> list)
        {
            PipelineColumn[] retval = new PipelineColumn[list.Count + 1];
            retval[0] = first;
            for (int i = 0; i < list.Count; ++i)
                retval[i + 1] = ((IContainsColumn)list[i]).WrappedColumn;
            return retval;
        }

        public static Vector<T> ConcatWith<T>(this Scalar<T> me, params ScalarOrVector<T>[] i)
            => _rec.Vector<T>(Helper(me, i));
        public static Vector<T> ConcatWith<T>(this Vector<T> me, params ScalarOrVector<T>[] i)
            => _rec.Vector<T>(Helper(me, i));

        public static VarVector<T> ConcatWith<T>(this Scalar<T> me, params ScalarOrVectorOrVarVector<T>[] i)
            => _rec.VarVector<T>(Helper(me, i));
        public static VarVector<T> ConcatWith<T>(this Vector<T> me, params ScalarOrVectorOrVarVector<T>[] i)
            => _rec.VarVector<T>(Helper(me, i));
        public static VarVector<T> ConcatWith<T>(this VarVector<T> me, params ScalarOrVectorOrVarVector<T>[] i)
            => _rec.VarVector<T>(Helper(me, i));
    }

    public static class NormalizeTransformExtensions
    {
        private static FakeTransformReconciler _rec = new FakeTransformReconciler("Normalize");

        public static Vector<float> Normalize(this Vector<float> me)
            => _rec.Vector<float>(me);

        public static Vector<double> Normalize(this Vector<double> me)
            => _rec.Vector<double>(me);
    }

    public static class WordTokenizeTransformExtensions
    {
        private static FakeTransformReconciler _rec = new FakeTransformReconciler("WordTokenize");

        public static VarVector<string> Tokenize(this Scalar<string> me)
            => _rec.VarVector<string>(me);
        public static VarVector<string> Tokenize(this Vector<string> me)
            => _rec.VarVector<string>(me);
        public static VarVector<string> Tokenize(this VarVector<string> me)
            => _rec.VarVector<string>(me);
    }

    public static class TermTransformExtensions
    {
        private static FakeTransformReconciler _rec = new FakeTransformReconciler("Term");

        public static Key<uint, T> Dictionarize<T>(this Scalar<T> me)
            => _rec.Key<uint, T>(me);
        public static Vector<Key<uint, T>> Dictionarize<T>(this Vector<T> me)
            => _rec.Vector<Key<uint, T>>(me);
        public static VarVector<Key<uint, T>> Dictionarize<T>(this VarVector<T> me)
            => _rec.VarVector<Key<uint, T>>(me);
    }

    public static class KeyToVectorTransformExtensions
    {
        private static FakeTransformReconciler _rec = new FakeTransformReconciler("KeyToVector");

        public static Vector<float> BagVectorize<T>(this VarVector<Key<T>> me)
            => _rec.Vector<float>(me);
        public static Vector<float> BagVectorize<T, TVal>(this VarVector<Key<T, TVal>> me)
            => _rec.Vector<float>(me);
    }

    public static class TextTransformExtensions
    {
        private static FakeTransformReconciler _rec = new FakeTransformReconciler("TextTransform");

        /// <summary>
        /// Performs text featurization on the input text. This will tokenize, do n-gram featurization,
        /// dictionary based term mapping, and finally produce a word-bag vector for the output.
        /// </summary>
        /// <param name="me">The text to featurize</param>
        /// <returns></returns>
        public static Vector<float> TextFeaturizer(this Scalar<string> me, bool keepDiacritics = true)
            => _rec.Vector<float>(me);
    }

    public static class TrainerTransformExtensions
    {
        private static FakeTransformReconciler _rec = new FakeTransformReconciler("LinearBinaryClassification");

        /// <summary>
        /// Trains a linear predictor using logistic regression.
        /// </summary>
        /// <param name="label">The target label for this binary classification task</param>
        /// <param name="features">The features to train on. Should be normalized.</param>
        /// <returns>A tuple of columns representing the score, the calibrated score as a probability, and the boolean predicted label</returns>
        public static (Scalar<float> score, Scalar<float> probability, Scalar<bool> predictedLabel) TrainLinearClassification(this Scalar<bool> label, Vector<float> features)
            => (_rec.Scalar<float>(label, features), _rec.Scalar<float>(label, features), _rec.Scalar<bool>(label, features));
    }

    public static class HashTransformExtensions
    {
        private static FakeTransformReconciler _rec = new FakeTransformReconciler("Hash");

        public static Key<uint> Hash<T>(this Scalar<T> me)
            => _rec.Key<uint>(me);
        public static Key<uint, string> Hash<T>(this Scalar<T> me, int invertHashTokens)
            => _rec.Key<uint, string>(me);
        public static Vector<Key<uint>> Hash<T>(this Vector<T> me)
            => _rec.Vector<Key<uint>>(me);
        public static Vector<Key<uint, string>> Hash<T>(this Vector<T> me, int invertHashTokens)
            => _rec.Vector<Key<uint, string>>(me);
        public static VarVector<Key<uint>> Hash<T>(this VarVector<T> me)
            => _rec.VarVector<Key<uint>>(me);
        public static VarVector<Key<uint, string>> Hash<T>(this VarVector<T> me, int invertHashTokens)
            => _rec.VarVector<Key<uint, string>>(me);
    }
}
