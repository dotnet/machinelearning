// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Security;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Recommender.Internal
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
            [FieldOffset(0)]
            public int K;
            [FieldOffset(4)]
            public int NrThreads;
            [FieldOffset(8)]
            public int NrBins;
            [FieldOffset(12)]
            public int NrIters;
            [FieldOffset(16)]
            public float Lambda;
            [FieldOffset(20)]
            public float Eta;
            [FieldOffset(24)]
            public int DoNmf;
            [FieldOffset(28)]
            public int Quiet;
            [FieldOffset(32)]
            public int CopyData;
        }

        [StructLayout(LayoutKind.Explicit)]
        private unsafe struct MFModel
        {
            [FieldOffset(0)]
            public int M;
            [FieldOffset(4)]
            public int N;
            [FieldOffset(8)]
            public int K;
            [FieldOffset(16)]
            public float* P;
            [FieldOffset(24)]
            public float* Q;
        }

        private const string DllPath = "LibMFWrapper";

        [DllImport(DllPath), SuppressUnmanagedCodeSecurity]
        private static unsafe extern void MFDestroyModel(ref MFModel* model);

        [DllImport(DllPath), SuppressUnmanagedCodeSecurity]
        private static unsafe extern MFModel* MFTrain(MFProblem* prob, MFParameter* param);

        [DllImport(DllPath), SuppressUnmanagedCodeSecurity]
        private static unsafe extern MFModel* MFTrainWithValidation(MFProblem* tr, MFProblem* va, MFParameter* param);

        [DllImport(DllPath), SuppressUnmanagedCodeSecurity]
        private static unsafe extern float MFCrossValidation(MFProblem* prob, int nrFolds, MFParameter* param);

        [DllImport(DllPath), SuppressUnmanagedCodeSecurity]
        private static unsafe extern float MFPredict(MFModel* model, int pIdx, int qIdx);

        private MFParameter _mfParam;
        private unsafe MFModel* _pMFModel;
        private readonly IHost _host;

        public SafeTrainingAndModelBuffer(IHostEnvironment env, int k, int nrBins, int nrThreads, int nrIters, double lambda, double eta,
            bool doNmf, bool quiet, bool copyData)
        {
            _host = env.Register("SafeTrainingAndModelBuffer");
            _mfParam.K = k;
            _mfParam.NrBins = nrBins;
            _mfParam.NrThreads = nrThreads;
            _mfParam.NrIters = nrIters;
            _mfParam.Lambda = (float)lambda;
            _mfParam.Eta = (float)eta;
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

        private MFNode[] ConstructLabeledNodesFrom(IChannel ch, ICursor cursor, ValueGetter<float> labGetter,
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
                    // REVIEW tfinley: Instead of ignoring, should I throw in the row > rowCount case?
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
            ICursor cursor, ValueGetter<float> labGetter,
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
            ICursor cursor, ValueGetter<float> labGetter,
            ValueGetter<uint> rowGetter, ValueGetter<uint> colGetter,
            ICursor validCursor, ValueGetter<float> validLabGetter,
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
