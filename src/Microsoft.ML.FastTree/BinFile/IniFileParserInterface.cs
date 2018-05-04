// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;

namespace Microsoft.ML.Runtime.FastTree.Internal
{
    internal sealed class IniFileParserInterface
    {
        private static class Native
        {
            private const string DllName = @"NeuralTreeEvaluator.dll";

            [DllImport(DllName, CharSet = CharSet.Ansi, EntryPoint = "InputExtractorCreateFromRanker")]
            public static extern IntPtr CreateFromRanker(string path);

            [DllImport(DllName, CharSet = CharSet.Ansi, EntryPoint = "InputExtractorCreateFromInputIni")]
            public static extern IntPtr CreateFromInputIni(string path);

            [DllImport(DllName, CharSet = CharSet.Ansi, EntryPoint = "InputExtractorCreateFromFreeform")]
            public static extern IntPtr CreateFromFreeform(string freeform);

            [DllImport(DllName, CharSet = CharSet.Ansi, EntryPoint = "InputExtractorCreateFromFreeformV2")]
            public static extern IntPtr CreateFromFreeform2(string freeform);

            [DllImport(DllName, EntryPoint = "InputExtractorDispose")]
            public static extern void DisposeInputExtractor(IntPtr pObject);

            [DllImport(DllName, EntryPoint = "FeatureMapGetFeatureCount")]
            public static extern UInt32 GetFeatureCount(IntPtr pObject);

            [DllImport(DllName, EntryPoint = "FeatureMapGetFeatureNameMaxLength")]
            public static extern UInt32 GetFeatureNameMaxLength(IntPtr pObject);

            [DllImport(DllName, CharSet = CharSet.Ansi, EntryPoint = "FeatureMapGetFeatureIndex")]
            [return: MarshalAs(UnmanagedType.U1)]
            public static extern bool GetFeatureIndex(IntPtr pObject, string featureName, out UInt32 featureIndex);

            [DllImport(DllName, CharSet = CharSet.Ansi, EntryPoint = "FeatureMapGetFeatureName")]
            [return: MarshalAs(UnmanagedType.U1)]
            public unsafe static extern bool GetFeatureName(IntPtr pObject, UInt32 featureIndex, byte[] buffer, UInt32 sizeOfBuffer, IntPtr resultLength);

            [DllImport(DllName, CharSet = CharSet.Ansi, EntryPoint = "InputExtractorGetInputName")]
            [return: MarshalAs(UnmanagedType.U1)]
            public unsafe static extern bool GetInputName(IntPtr pObject, UInt32 featureIndex, byte[] buffer, UInt32 sizeOfBuffer, IntPtr resultLength);

            [DllImport(DllName, CharSet = CharSet.Ansi)]
            [return: MarshalAs(UnmanagedType.U1)]
            public unsafe static extern bool GetSectionContent(IntPtr pObject, string sectionName, byte[] buffer, UInt32 sizeOfBuffer, IntPtr resultLength);

            [DllImport(DllName, EntryPoint = "InputExtractorGetInputCount")]
            public static extern UInt32 GetInputCount(IntPtr pObject);

            [DllImport(DllName, EntryPoint = "InputExtractorGetInput")]
            public static extern IntPtr GetInput(IntPtr pObject, UInt32 index);

            [DllImport(DllName, EntryPoint = "InputGetFeatures")]
            public static unsafe extern void GetInputFeatures(IntPtr pInput, UInt32[] features, UInt32 sizeOfFeatures, out UInt32 featureCount);

            [DllImport(DllName, EntryPoint = "InputIsCopy")]
            [return: MarshalAs(UnmanagedType.U1)]
            public unsafe static extern bool IsCopyInput(IntPtr pInput);

            [DllImport(DllName, EntryPoint = "InputEvaluate")]
            public static unsafe extern double EvaluateInput(IntPtr pInput, UInt32* input);

            [DllImport(DllName, EntryPoint = "InputEvaluateMany")]
            public static unsafe extern void EvaluateMany(IntPtr pInput, UInt32*[] inputs, double* outputs, UInt32 count);

            [DllImport(DllName, EntryPoint = "InputExtractorGetFeatureMap")]
            public static extern IntPtr GetFeatureMap(IntPtr pExtractor);
        }

        private unsafe string StringFetch<T>(Func<IntPtr, T, byte[], UInt32, IntPtr, bool> fetcher, T id, IntPtr objectPtr)
        {
            byte[] buffer = new byte[100];
            UInt32 len = 0;
            UInt32* pLen = &len;
            if (!fetcher(objectPtr, id, buffer, (UInt32)buffer.Length, (IntPtr)pLen))
            {
                if (len <= buffer.Length)
                    return null;
                Array.Resize(ref buffer, (int)len);
                // With sufficient size, we should now succeed.
                if (!fetcher(objectPtr, id, buffer, (UInt32)buffer.Length, (IntPtr)pLen))
                    return null;
            }
            return UTF8Encoding.UTF8.GetString(buffer, 0, (int)len);
        }

        public string GetInputContent(int inputId)
        {
            return StringFetch(Native.GetSectionContent, String.Format("Input:{0}", inputId + 1), InputExtractorPtr);
        }

        public string GetInputName(int inputId)
        {
            return StringFetch(Native.GetInputName, (UInt32)inputId, InputExtractorPtr);
        }

        public string GetFeatureName(int featureId)
        {
            return StringFetch(Native.GetFeatureName, (UInt32)featureId, FeatureMapPtr);
        }

        private IntPtr InputExtractorPtr { get; set; }
        private IntPtr FeatureMapPtr { get; set; }
        private readonly FeatureMap _featureMap;
        private readonly FeatureEvaluator[] _evaluators;

        /// <summary>
        /// Gets the map between feature names and indices
        /// </summary>
        public FeatureMap GetFeatureMap() { return _featureMap; }

        /// <summary>
        /// Gets the list of FeatureEvaluators
        /// </summary>
        /// <returns></returns>
        public FeatureEvaluator[] GetFeatureEvaluators() { return _evaluators; }

        public static IniFileParserInterface CreateFromFreeform(string freeform)
        {
            IntPtr ptr = Native.CreateFromFreeform(freeform);
            if (ptr == IntPtr.Zero)
                throw Contracts.ExceptDecode("Unable to load freeform {0}", freeform);
            return new IniFileParserInterface(ptr);
        }

        public static IniFileParserInterface CreateFromFreeform2(string freeform)
        {
            IntPtr ptr = Native.CreateFromFreeform2(freeform);
            if (ptr == IntPtr.Zero)
                throw Contracts.ExceptDecode("Unable to load freeform2 {0}", freeform);
            return new IniFileParserInterface(ptr);
        }

        public static IniFileParserInterface CreateFromInputIni(string path)
        {
            IntPtr ptr = Native.CreateFromInputIni(path);
            if (ptr == IntPtr.Zero)
                throw Contracts.ExceptDecode("Unable to load input ini at {0}", path);
            return new IniFileParserInterface(ptr);
        }

        public static IniFileParserInterface CreateFromRanker(string path)
        {
            IntPtr ptr = Native.CreateFromRanker(path);
            if (ptr == IntPtr.Zero)
                throw Contracts.ExceptDecode("Unable to load tree ensemble ranker at {0}", path);
            return new IniFileParserInterface(ptr);
        }

        /// <summary>
        /// Creates an InputExtractor wrapper for a given unmanaged pointer
        /// </summary>
        /// <param name="pInputExtractor"></param>
        private IniFileParserInterface(IntPtr pInputExtractor)
        {
            InputExtractorPtr = pInputExtractor;
            FeatureMapPtr = Native.GetFeatureMap(InputExtractorPtr);
            _featureMap = new FeatureMap(this);
            int numInputs = (int)Native.GetInputCount(InputExtractorPtr);
            _evaluators = Enumerable.Range(0, numInputs).Select(i => new FeatureEvaluator(this, i)).ToArray();
        }

        public int FeatureCount => (int)Native.GetFeatureCount(FeatureMapPtr);

        public int InputCount => (int)Native.GetInputCount(InputExtractorPtr);

        #region destructor and dispose

        public void Dispose()
        {
            Dispose(true);
        }

        private void Dispose(bool bDisposing)
        {
            if (InputExtractorPtr != IntPtr.Zero)
            {
                // Call the DLL Export to dispose this class
                Native.DisposeInputExtractor(InputExtractorPtr);
                InputExtractorPtr = IntPtr.Zero;
            }

            if (bDisposing)
            {
                // If true, it's called by user's code
                GC.SuppressFinalize(this);
            }
            // else it's called by the runtime from inside the finalizer
        }

        // This finalizer is called when Garbage collection occurs, but only if
        // the IDisposable.Dispose method wasn't already called.
        ~IniFileParserInterface()
        {
            Dispose(false);
        }

        #endregion

        /// <summary>
        /// Wraps the functions of InputExtractor that map between raw feature names and indices
        /// </summary>
        public class FeatureMap
        {
            private IniFileParserInterface _parserInterface;

            // Get the number of features required to evaluate all the inputs
            public int RawFeatureCount => (int)Native.GetFeatureCount(_parserInterface.FeatureMapPtr);

            public string this[int i] => GetRawFeatureName(i);
            public int this[string n] => GetRawFeatureIndex(n);

            public FeatureMap(IniFileParserInterface parserInterface)
            {
                _parserInterface = parserInterface;
            }

            // Get the index of a raw feature
            // wrapper to GetFeatureIndex(IntPtr p_object, char* featureName, UInt32* featureIndex);
            // throws ane exception if not found
            public int GetRawFeatureIndex(string name)
            {
                UInt32 featureIndex;
                bool ret = Native.GetFeatureIndex(_parserInterface.FeatureMapPtr, name, out featureIndex);
                if (!ret)
                    throw Contracts.ExceptParam(nameof(name), "GetRawFeatureIndex failed for feature {0}", name);
                return (int)featureIndex;
            }

            // Get the name of a raw feature from its index
            // wrapper to GetFeatureName(IntPtr p_object, UInt32 featureIndex, char* featureNameBuffer, UInt32 sizeOfBuffer);
            // returns null if not found
            public unsafe string GetRawFeatureName(int index)
            {
                return _parserInterface.StringFetch(Native.GetFeatureName, (UInt32)index, _parserInterface.FeatureMapPtr);
            }
        }

        /// <summary>
        /// Wraps the Evaluate method of InputExtractor (and associated methods)
        /// </summary>
        public sealed class FeatureEvaluator
        {
            private readonly IniFileParserInterface _parserInterface;
            private readonly IntPtr _inputPtr;

            public int Id { get; }

            public MD5Hash ContentMD5Hash => MD5Hasher.Hash(Content);

            // Return the name of the input
            public unsafe string Name => _parserInterface.GetInputName(Id);

            // Return the content of the input
            public unsafe string Content => _parserInterface.GetInputContent(Id);

            // Return true if this input evaluator (identified by inputID) just copies a raw feature
            public bool IsRawFeatureEvaluator => Native.IsCopyInput(_inputPtr);

            public FeatureEvaluator(IniFileParserInterface parserInterface, int inputId)
            {
                _parserInterface = parserInterface;
                Id = inputId;
                _inputPtr = Native.GetInput(parserInterface.InputExtractorPtr, (UInt32)Id);
            }

            // Get all the associated features (indices) for an evaluator (identified by evaluatorId)
            public unsafe int[] GetRequiredRawFeatureIndices()
            {
                UInt32[] associatedFeaturesList = new UInt32[_parserInterface.FeatureCount];
                UInt32 numAssociatedFeatures;

                Native.GetInputFeatures(_inputPtr, associatedFeaturesList, (UInt32)associatedFeaturesList.Length, out numAssociatedFeatures);
                return associatedFeaturesList.Take((int)numAssociatedFeatures).Select(x => (int)x).Distinct().OrderBy(x => x).ToArray();
            }

            // Evaluate an input evaluator (identified by evaluatorId)
            public unsafe double Evaluate(uint[] input)
            {
                fixed (uint* pInput = input)
                    return Native.EvaluateInput(_inputPtr, pInput);
            }
        }
    }
}
