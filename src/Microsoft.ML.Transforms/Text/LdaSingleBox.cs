﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security;
using Microsoft.ML.Runtime;
using Microsoft.Win32.SafeHandles;

namespace Microsoft.ML.TextAnalytics
{

    internal static class LdaInterface
    {
        public sealed class SafeLdaEngineHandle : SafeHandleZeroOrMinusOneIsInvalid
        {
            private SafeLdaEngineHandle()
                : base(true)
            {
            }

            protected override bool ReleaseHandle()
            {
                DestroyEngine(handle);
                return true;
            }
        }

        private const string NativePath = "LdaNative";
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern SafeLdaEngineHandle CreateEngine(int numTopic, int numVocab, float alphaSum, float beta, int numIter,
            int likelihoodInterval, int numThread, int mhstep, int maxDocToken);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern void AllocateModelMemory(SafeLdaEngineHandle engine, int numTopic, int numVocab, long tableSize, long aliasTableSize);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern void AllocateDataMemory(SafeLdaEngineHandle engine, int docNum, long corpusSize);

        [DllImport(NativePath, CharSet = CharSet.Ansi), SuppressUnmanagedCodeSecurity]
        internal static extern void Train(SafeLdaEngineHandle engine, string trainOutput);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern void GetModelStat(SafeLdaEngineHandle engine, out long memBlockSize, out long aliasMemBlockSize);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern void Test(SafeLdaEngineHandle engine, int numBurninIter, float[] pLogLikelihood);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern void CleanData(SafeLdaEngineHandle engine);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern void CleanModel(SafeLdaEngineHandle engine);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        private static extern void DestroyEngine(IntPtr engine);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern void GetWordTopic(SafeLdaEngineHandle engine, int wordId, int[] pTopic, int[] pProb, ref int length);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern void SetWordTopic(SafeLdaEngineHandle engine, int wordId, int[] pTopic, int[] pProb, int length);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern void SetAlphaSum(SafeLdaEngineHandle engine, float avgDocLength);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern int FeedInData(SafeLdaEngineHandle engine, int[] termId, int[] termFreq, int termNum, int numVocab);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern int FeedInDataDense(SafeLdaEngineHandle engine, int[] termFreq, int termNum, int numVocab);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern void GetDocTopic(SafeLdaEngineHandle engine, int docId, int[] pTopic, int[] pProb, ref int numTopicReturn);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern void GetTopicSummary(SafeLdaEngineHandle engine, int topicId, int[] pWords, float[] pProb, ref int numTopicReturn);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern void TestOneDoc(SafeLdaEngineHandle engine, int[] termId, int[] termFreq, int termNum, int[] pTopics, int[] pProbs, ref int numTopicsMax, int numBurnIter, bool reset);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern void TestOneDocDense(SafeLdaEngineHandle engine, int[] termFreq, int termNum, int[] pTopics, int[] pProbs, ref int numTopicsMax, int numBurninIter, bool reset);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern void InitializeBeforeTrain(SafeLdaEngineHandle engine);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern void InitializeBeforeTest(SafeLdaEngineHandle engine);
    }

    internal sealed class LdaSingleBox : IDisposable
    {
        private LdaInterface.SafeLdaEngineHandle _engine;
        private bool _isDisposed;
        private int[] _topics;
        private int[] _probabilities;
        private int[] _summaryTerm;
        private float[] _summaryTermProb;
        private readonly int _likelihoodInterval;
        private readonly float _alpha;
        private readonly float _beta;
        private readonly int _mhStep;
        private readonly int _numThread;
        private readonly int _numSummaryTerms;
        private readonly bool _denseOutput;

        public readonly int NumTopic;
        public readonly int NumVocab;
        public LdaSingleBox(int numTopic, int numVocab, float alpha,
                            float beta, int numIter, int likelihoodInterval, int numThread,
                            int mhstep, int numSummaryTerms, bool denseOutput, int maxDocToken)
        {
            NumTopic = numTopic;
            NumVocab = numVocab;
            _alpha = alpha;
            _beta = beta;
            _mhStep = mhstep;
            _numSummaryTerms = numSummaryTerms;
            _denseOutput = denseOutput;
            _likelihoodInterval = likelihoodInterval;
            _numThread = numThread;

            _topics = new int[numTopic];
            _probabilities = new int[numTopic];

            _summaryTerm = new int[_numSummaryTerms];
            _summaryTermProb = new float[_numSummaryTerms];

            _engine = LdaInterface.CreateEngine(numTopic, numVocab, alpha, beta, numIter, likelihoodInterval, numThread, mhstep, maxDocToken);
        }

        public void AllocateModelMemory(int numTopic, int numVocab, long tableSize, long aliasTableSize)
        {
            Contracts.Check(numTopic >= 0);
            Contracts.Check(numVocab >= 0);
            Contracts.Check(tableSize >= 0);
            Contracts.Check(aliasTableSize >= 0);
            LdaInterface.AllocateModelMemory(_engine, numVocab, numTopic, tableSize, aliasTableSize);
        }

        public void AllocateDataMemory(int docNum, long corpusSize)
        {
            Contracts.Check(docNum >= 0);
            Contracts.Check(corpusSize >= 0);
            LdaInterface.AllocateDataMemory(_engine, docNum, corpusSize);
        }

        public void Train(string trainOutput)
        {
            if (string.IsNullOrWhiteSpace(trainOutput))
                LdaInterface.Train(_engine, null);
            else
                LdaInterface.Train(_engine, trainOutput);
        }

        public void GetModelStat(out long memBlockSize, out long aliasMemBlockSize)
        {
            LdaInterface.GetModelStat(_engine, out memBlockSize, out aliasMemBlockSize);
        }

        public void Test(int numBurninIter, float[] logLikelihood)
        {
            Contracts.Check(numBurninIter >= 0);
            var pLogLikelihood = new float[numBurninIter];
            LdaInterface.Test(_engine, numBurninIter, pLogLikelihood);
            logLikelihood = pLogLikelihood.Select(item => (float)item).ToArray();
        }

        public void CleanData()
        {
            LdaInterface.CleanData(_engine);
        }

        public void CleanModel()
        {
            LdaInterface.CleanModel(_engine);
        }

        public void CopyModel(LdaSingleBox trainer, int wordId)
        {
            int length = NumTopic;
            LdaInterface.GetWordTopic(trainer._engine, wordId, _topics, _probabilities, ref length);
            LdaInterface.SetWordTopic(_engine, wordId, _topics, _probabilities, length);
        }

        public void SetAlphaSum(float averageDocLength)
        {
            LdaInterface.SetAlphaSum(_engine, averageDocLength);
        }

        public int LoadDoc(ReadOnlySpan<int> termID, ReadOnlySpan<double> termVal, int termNum, int numVocab)
        {
            Contracts.Check(numVocab == NumVocab);
            Contracts.Check(termNum > 0);
            Contracts.Check(termID.Length >= termNum);
            Contracts.Check(termVal.Length >= termNum);

            int[] pID = new int[termNum];
            int[] pVal = new int[termVal.Length];
            for (int i = 0; i < termVal.Length; i++)
                pVal[i] = (int)termVal[i];
            termID.Slice(0, termNum).CopyTo(pID);
            return LdaInterface.FeedInData(_engine, pID, pVal, termNum, NumVocab);
        }

        public int LoadDocDense(ReadOnlySpan<double> termVal, int termNum, int numVocab)
        {
            Contracts.Check(numVocab == NumVocab);
            Contracts.Check(termNum > 0);

            Contracts.Check(termVal.Length >= termNum);

            int[] pID = new int[termNum];
            int[] pVal = new int[termVal.Length];
            for (int i = 0; i < termVal.Length; i++)
                pVal[i] = (int)termVal[i];
            return LdaInterface.FeedInDataDense(_engine, pVal, termNum, NumVocab);
        }

        public List<KeyValuePair<int, float>> GetDocTopicVector(int docID)
        {
            int numTopicReturn = NumTopic;
            LdaInterface.GetDocTopic(_engine, docID, _topics, _probabilities, ref numTopicReturn);
            var topicRet = new List<KeyValuePair<int, float>>();
            int currentTopic = 0;
            for (int i = 0; i < numTopicReturn; i++)
            {
                if (_denseOutput)
                {
                    while (currentTopic < _topics[i])
                    {
                        //use a value to smooth the count so that we get dense output on each topic
                        //the smooth value is usually set to 0.1
                        topicRet.Add(new KeyValuePair<int, float>(currentTopic, (float)_alpha));
                        currentTopic++;
                    }
                    topicRet.Add(new KeyValuePair<int, float>(_topics[i], _probabilities[i] + (float)_alpha));
                    currentTopic++;
                }
                else
                {
                    topicRet.Add(new KeyValuePair<int, float>(_topics[i], (float)_probabilities[i]));
                }
            }

            if (_denseOutput)
            {
                while (currentTopic < NumTopic)
                {
                    topicRet.Add(new KeyValuePair<int, float>(currentTopic, (float)_alpha));
                    currentTopic++;
                }
            }
            return topicRet;
        }

        public List<KeyValuePair<int, float>> TestDoc(ReadOnlySpan<int> termID, ReadOnlySpan<double> termVal, int termNum, int numBurninIter, bool reset)
        {
            Contracts.Check(termNum > 0);
            Contracts.Check(termVal.Length >= termNum);
            Contracts.Check(termID.Length >= termNum);

            int[] pID = new int[termNum];
            int[] pVal = new int[termVal.Length];
            for (int i = 0; i < termVal.Length; i++)
                pVal[i] = (int)termVal[i];
            int[] pTopic = new int[NumTopic];
            int[] pProb = new int[NumTopic];
            termID.Slice(0, termNum).CopyTo(pID);

            int numTopicReturn = NumTopic;

            LdaInterface.TestOneDoc(_engine, pID, pVal, termNum, pTopic, pProb, ref numTopicReturn, numBurninIter, reset);

            // PREfast suspects that the value of numTopicReturn could be changed in _engine->TestOneDoc, which might result in read overrun in the following loop.
            if (numTopicReturn > NumTopic)
            {
                Contracts.Check(false);
                numTopicReturn = NumTopic;
            }

            var topicRet = new List<KeyValuePair<int, float>>();
            for (int i = 0; i < numTopicReturn; i++)
                topicRet.Add(new KeyValuePair<int, float>(pTopic[i], (float)pProb[i]));
            return topicRet;
        }

        public List<KeyValuePair<int, float>> TestDocDense(ReadOnlySpan<double> termVal, int termNum, int numBurninIter, bool reset)
        {
            Contracts.Check(termNum > 0);
            Contracts.Check(numBurninIter > 0);
            Contracts.Check(termVal.Length >= termNum);
            int[] pVal = new int[termVal.Length];
            for (int i = 0; i < termVal.Length; i++)
                pVal[i] = (int)termVal[i];
            int[] pTopic = new int[NumTopic];
            int[] pProb = new int[NumTopic];

            int numTopicReturn = NumTopic;

            // There are two versions of TestOneDoc interfaces
            // (1) TestOneDoc
            // (2) TestOneDocRestart
            // The second one is the same as the first one except that it will reset
            // the states of the internal random number generator, so that it yields reproducible results for the same input
            LdaInterface.TestOneDocDense(_engine, pVal, termNum, pTopic, pProb, ref numTopicReturn, numBurninIter, reset);

            // PREfast suspects that the value of numTopicReturn could be changed in _engine->TestOneDoc, which might result in read overrun in the following loop.
            if (numTopicReturn > NumTopic)
            {
                Contracts.Check(false);
                numTopicReturn = NumTopic;
            }

            var topicRet = new List<KeyValuePair<int, float>>();
            for (int i = 0; i < numTopicReturn; i++)
                topicRet.Add(new KeyValuePair<int, float>(pTopic[i], (float)pProb[i]));
            return topicRet;
        }

        public void InitializeBeforeTrain()
        {
            LdaInterface.InitializeBeforeTrain(_engine);
        }

        public void InitializeBeforeTest()
        {
            LdaInterface.InitializeBeforeTest(_engine);
        }

        public KeyValuePair<int, int>[] GetModel(int wordId)
        {
            int length = NumTopic;
            LdaInterface.GetWordTopic(_engine, wordId, _topics, _probabilities, ref length);
            var wordTopicVector = new KeyValuePair<int, int>[length];

            for (int i = 0; i < length; i++)
                wordTopicVector[i] = new KeyValuePair<int, int>(_topics[i], _probabilities[i]);
            return wordTopicVector;
        }

        public KeyValuePair<int, float>[] GetTopicSummary(int topicId)
        {
            int length = _numSummaryTerms;
            LdaInterface.GetTopicSummary(_engine, topicId, _summaryTerm, _summaryTermProb, ref length);
            var topicSummary = new KeyValuePair<int, float>[length];

            for (int i = 0; i < length; i++)
                topicSummary[i] = new KeyValuePair<int, float>(_summaryTerm[i], _summaryTermProb[i]);
            return topicSummary;
        }

        public void SetModel(int termID, int[] topicID, int[] topicProb, int topicNum)
        {
            Contracts.Check(termID >= 0);
            Contracts.Check(topicNum <= NumTopic);
            Array.Copy(topicID, _topics, topicNum);
            Array.Copy(topicProb, _probabilities, topicNum);
            LdaInterface.SetWordTopic(_engine, termID, _topics, _probabilities, topicNum);
        }

        public void Dispose()
        {
            if (_isDisposed)
                return;
            _isDisposed = true;
            _engine.Dispose();
        }
    }
}
