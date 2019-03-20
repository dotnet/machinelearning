// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Security;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Recommender.Internal
{
    /// <summary>
    /// Contains mirrors of unmanaged struct import extern functions from mf.h / mf.cpp, which implements Matrix Factorization in native C++.
    /// It also wraps/bridges the train, traintest and cv interfaces ready for ML.NET infra.
    /// </summary>
    internal sealed class SafeTrainingAndModelBuffer : IDisposable
    {
        [StructLayout(LayoutKind.Explicit)]
        private struct MFNode
        {
            [FieldOffset(0)]
            public int U;
            [FieldOffset(4)]
            public int V;
            [FieldOffset(8)]
            public float R;
        }

        [StructLayout(LayoutKind.Explicit)]
        private unsafe struct MFProblem
        {
            [FieldOffset(0)]
            public int M;
            [FieldOffset(4)]
            public int N;
            [FieldOffset(8)]
            public long Nnz;
            [FieldOffset(16)]
            public MFNode* R;
        }

        [StructLayout(LayoutKind.Explicit)]
        private struct MFParameter
        {
            /// <summary>
            /// Enum of loss functions which can be minimized.
            ///  0: square loss for regression.
            ///  1: absolute loss for regression.
            ///  2: KL-divergence for regression.
            ///  5: logistic loss for binary classification.
            ///  6: squared hinge loss for binary classification.
            ///  7: hinge loss for binary classification.
            ///  10: row-wise Bayesian personalized ranking.
            ///  11: column-wise Bayesian personalized ranking.
            ///  12: squared loss for implicit-feedback matrix factorization.
            /// Fun 12 is solved by a coordinate descent method while other functions invoke
            /// a stochastic gradient method.
            /// </summary>
            [FieldOffset(0)]
            public int Fun;

            /// <summary>
            /// Rank of factor matrices.
            /// </summary>
            [FieldOffset(4)]
            public int K;

            /// <summary>
            /// Number of threads which can be used for training.
            /// </summary>
            [FieldOffset(8)]
            public int NrThreads;

            /// <summary>
            /// Number of blocks that the training matrix is divided into. The parallel stochastic gradient
            /// method in LIBMF processes assigns each thread a block at one time. The ratings in one block
            /// would be sequentially accessed (not randomaly accessed like standard stochastic gradient methods).
            /// </summary>
            [FieldOffset(12)]
            public int NrBins;

            /// <summary>
            /// Number of training iteration. At one iteration, all values in the training matrix are roughly accessed once.
            /// </summary>
            [FieldOffset(16)]
            public int NrIters;

            /// <summary>
            /// L1-norm regularization coefficient of left factor matrix.
            /// </summary>
            [FieldOffset(20)]
            public float LambdaP1;

            /// <summary>
            /// L2-norm regularization coefficient of left factor matrix.
            /// </summary>
            [FieldOffset(24)]
            public float LambdaP2;

            /// <summary>
            /// L1-norm regularization coefficient of right factor matrix.
            /// </summary>
            [FieldOffset(28)]
            public float LambdaQ1;

            /// <summary>
            /// L2-norm regularization coefficient of right factor matrix.
            /// </summary>
            [FieldOffset(32)]
            public float LambdaQ2;

            /// <summary>
            /// Learning rate of LIBMF's stochastic gradient method.
            /// </summary>
            [FieldOffset(36)]
            public float Eta;

            /// <summary>
            /// Coefficient of loss function on unobserved entries in the training matrix. It's used only with fun=12.
            /// </summary>
            [FieldOffset(40)]
            public float Alpha;

            /// <summary>
            /// Desired value of unobserved entries in the training matrix. It's used only with fun=12.
            /// </summary>
            [FieldOffset(44)]
            public float C;

            /// <summary>
            /// Specify if the factor matrices should be non-negative.
            /// </summary>
            [FieldOffset(48)]
            public int DoNmf;

            /// <summary>
            /// Set to true so that LIBMF may produce less information to STDOUT.
            /// </summary>
            [FieldOffset(52)]
            public int Quiet;

            /// <summary>
            /// Set to false so that LIBMF may reuse and modifiy the data passed in.
            /// </summary>
            [FieldOffset(56)]
            public int CopyData;
        }

        [StructLayout(LayoutKind.Explicit)]
        private unsafe struct MFModel
        {
            [FieldOffset(0)]
            public int Fun;
            /// <summary>
            /// Number of rows in the training matrix.
            /// </summary>
            [FieldOffset(4)]
            public int M;
            /// <summary>
            /// Number of columns in the training matrix.
            /// </summary>
            [FieldOffset(8)]
            public int N;
            /// <summary>
            /// Rank of factor matrices.
            /// </summary>
            [FieldOffset(12)]
            public int K;
            /// <summary>
            /// Average value in the training matrix.
            /// </summary>
            [FieldOffset(16)]
            public float B;
            /// <summary>
            /// Left factor matrix. Its shape is M-by-K stored in row-major format.
            /// </summary>
            [FieldOffset(24)] // pointer is 8-byte on 64-bit machine.
            public float* P;
            /// <summary>
            /// Right factor matrix. Its shape is N-by-K stored in row-major format.
            /// </summary>
            [FieldOffset(32)] // pointer is 8-byte on 64-bit machine.
            public float* Q;
        }

        private const string NativePath = "MatrixFactorizationNative";

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        private static unsafe extern void MFDestroyModel(ref MFModel* model);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        private static unsafe extern MFModel* MFTrain(MFProblem* prob, MFParameter* param);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        private static unsafe extern MFModel* MFTrainWithValidation(MFProblem* tr, MFProblem* va, MFParameter* param);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        private static unsafe extern float MFCrossValidation(MFProblem* prob, int nrFolds, MFParameter* param);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        private static unsafe extern float MFPredict(MFModel* model, int pIdx, int qIdx);

        private MFParameter _mfParam;
        private unsafe MFModel* _pMFModel;
        private readonly IHost _host;

        public SafeTrainingAndModelBuffer(IHostEnvironment env, int fun, int k, int nrThreads,
            int nrBins, int nrIters, double lambda, double eta, double alpha, double c,
            bool doNmf, bool quiet, bool copyData)
        {
            _host = env.Register("SafeTrainingAndModelBuffer");
            _mfParam.Fun = fun;
            _mfParam.K = k;
            _mfParam.NrThreads = nrThreads;
            _mfParam.NrBins = nrBins;
            _mfParam.NrIters = nrIters;
            _mfParam.LambdaP1 = 0;
            _mfParam.LambdaP2 = (float)lambda;
            _mfParam.LambdaQ1 = 0;
            _mfParam.LambdaQ2 = (float)lambda;
            _mfParam.Eta = (float)eta;
            _mfParam.Alpha = (float)alpha;
            _mfParam.C = (float)c;
            _mfParam.DoNmf = doNmf ? 1 : 0;
            _mfParam.Quiet = quiet ? 1 : 0;
            _mfParam.CopyData = copyData ? 1 : 0;
        }

        ~SafeTrainingAndModelBuffer()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        private unsafe void Dispose(bool disposing)
        {
            // Free unmanaged resources.
            if (_pMFModel != null)
            {
                MFDestroyModel(ref _pMFModel);
                _host.Assert(_pMFModel == null);
            }
        }

        private MFNode[] ConstructLabeledNodesFrom(IChannel ch, DataViewRowCursor cursor, ValueGetter<float> labGetter,
            ValueGetter<uint> rowGetter, ValueGetter<uint> colGetter,
            int rowCount, int colCount)
        {
            long numSkipped = 0;
            uint row = 0;
            uint col = 0;
            float label = 0;

            List<MFNode> nodes = new List<MFNode>();
            long i = 0;
            using (var pch = _host.StartProgressChannel("Create matrix"))
            {
                pch.SetHeader(new ProgressHeader(new[] { "processed rows", "created nodes" }),
                    e => { e.SetProgress(0, i); e.SetProgress(1, nodes.Count); });
                while (cursor.MoveNext())
                {
                    i++;
                    labGetter(ref label);
                    if (!FloatUtils.IsFinite(label))
                    {
                        numSkipped++;
                        continue;
                    }
                    rowGetter(ref row);
                    // REVIEW: Instead of ignoring, should I throw in the row > rowCount case?
                    if (row == 0 || row > (uint)rowCount)
                    {
                        numSkipped++;
                        continue;
                    }
                    colGetter(ref col);
                    if (col == 0 || col > (uint)colCount)
                    {
                        numSkipped++;
                        continue;
                    }

                    MFNode node;
                    node.U = (int)(row - 1);
                    node.V = (int)(col - 1);
                    node.R = label;
                    nodes.Add(node);
                }
                pch.Checkpoint(i, nodes.Count);
            }
            if (numSkipped > 0)
                ch.Warning("Skipped {0} instances with missing/negative features during data loading", numSkipped);
            ch.Check(nodes.Count > 0, "No valid instances encountered during data loading");

            return nodes.ToArray();
        }

        public unsafe void Train(IChannel ch, int rowCount, int colCount,
            DataViewRowCursor cursor, ValueGetter<float> labGetter,
            ValueGetter<uint> rowGetter, ValueGetter<uint> colGetter)
        {
            if (_pMFModel != null)
            {
                MFDestroyModel(ref _pMFModel);
                _host.Assert(_pMFModel == null);
            }

            MFProblem prob = new MFProblem();
            MFNode[] nodes = ConstructLabeledNodesFrom(ch, cursor, labGetter, rowGetter, colGetter, rowCount, colCount);

            fixed (MFNode* nodesPtr = &nodes[0])
            {
                prob.R = nodesPtr;
                prob.M = rowCount;
                prob.N = colCount;
                prob.Nnz = nodes.Length;

                ch.Info("Training {0} by {1} problem on {2} examples",
                    prob.M, prob.N, prob.Nnz);

                fixed (MFParameter* pParam = &_mfParam)
                {
                    _pMFModel = MFTrain(&prob, pParam);
                }
            }
        }

        public unsafe void TrainWithValidation(IChannel ch, int rowCount, int colCount,
            DataViewRowCursor cursor, ValueGetter<float> labGetter,
            ValueGetter<uint> rowGetter, ValueGetter<uint> colGetter,
            DataViewRowCursor validCursor, ValueGetter<float> validLabGetter,
            ValueGetter<uint> validRowGetter, ValueGetter<uint> validColGetter)
        {
            if (_pMFModel != null)
            {
                MFDestroyModel(ref _pMFModel);
                _host.Assert(_pMFModel == null);
            }

            MFNode[] nodes = ConstructLabeledNodesFrom(ch, cursor, labGetter, rowGetter, colGetter, rowCount, colCount);
            MFNode[] validNodes = ConstructLabeledNodesFrom(ch, validCursor, validLabGetter, validRowGetter, validColGetter, rowCount, colCount);
            MFProblem prob = new MFProblem();
            MFProblem validProb = new MFProblem();
            fixed (MFNode* nodesPtr = &nodes[0])
            fixed (MFNode* validNodesPtrs = &validNodes[0])
            {
                prob.R = nodesPtr;
                prob.M = rowCount;
                prob.N = colCount;
                prob.Nnz = nodes.Length;

                validProb.R = validNodesPtrs;
                validProb.M = rowCount;
                validProb.N = colCount;
                validProb.Nnz = nodes.Length;

                ch.Info("Training {0} by {1} problem on {2} examples with a {3} by {4} validation set including {5} examples",
                    prob.M, prob.N, prob.Nnz, validProb.M, validProb.N, validProb.Nnz);

                fixed (MFParameter* pParam = &_mfParam)
                {
                    _pMFModel = MFTrainWithValidation(&prob, &validProb, pParam);
                }
            }
        }

        public unsafe void Get(out int m, out int n, out int k, out float[] p, out float[] q)
        {
            _host.Check(_pMFModel != null, "Attempted to get predictor before training");
            m = _pMFModel->M;
            _host.Check(m > 0, "Number of rows should have been positive but was not");
            n = _pMFModel->N;
            _host.Check(n > 0, "Number of columns should have been positive but was not");
            k = _pMFModel->K;
            _host.Check(k > 0, "Internal dimension should have been positive but was not");

            p = new float[m * k];
            q = new float[n * k];

            unsafe
            {
                Marshal.Copy((IntPtr)_pMFModel->P, p, 0, p.Length);
                Marshal.Copy((IntPtr)_pMFModel->Q, q, 0, q.Length);
            }
        }
    }
}
