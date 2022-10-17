// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.RegularExpressions;
using System.Text.Json;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.ML.Runtime;
//#if false
using Microsoft.ML.Trainers.FastTree;
//#else
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.CpuMath;
using Microsoft.ML.Internal.Utilities;
//using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Numeric;
using Microsoft.ML.Transforms;
//#endif

namespace Microsoft.ML.Trainers.XGBoost
{
    /// <summary>
    /// Wrapper of Booster object of XGBoost.
    /// </summary>
    internal class Booster : IDisposable
    {

        private bool disposed;
        private readonly IntPtr _handle;
        private const int normalPrediction = 0; // Value for the optionMask in prediction
#pragma warning disable CS0414
        private int numClass = 1;
#pragma warning restore CS0414

        public IntPtr Handle => _handle;

        public Booster(Dictionary<string, object> parameters, DMatrix trainDMatrix)
        {
            var dmats = new[] { trainDMatrix.Handle };
            var len = unchecked((ulong)dmats.Length);
            var errp = WrappedXGBoostInterface.XGBoosterCreate(dmats, len, out _handle);
            if (errp == -1)
            {
                string reason = WrappedXGBoostInterface.XGBGetLastError();
                throw new XGBoostDLLException(reason);
            }
            SetParameters(parameters);
        }

        public Booster(DMatrix trainDMatrix)
        {
            var dmats = new[] { trainDMatrix.Handle };
            var len = unchecked((ulong)dmats.Length);
            var errp = WrappedXGBoostInterface.XGBoosterCreate(dmats, len, out _handle);
            if (errp == -1)
            {
                string reason = WrappedXGBoostInterface.XGBGetLastError();
                throw new XGBoostDLLException(reason);
            }
        }

        public void Update(DMatrix train, int iter)
        {
            var errp = WrappedXGBoostInterface.XGBoosterUpdateOneIter(_handle, iter, train.Handle);
            if (errp == -1)
            {
                string reason = WrappedXGBoostInterface.XGBGetLastError();
                throw new XGBoostDLLException(reason);
            }
        }

        public float[] Predict(DMatrix test)
        {
            ulong predsLen;
            IntPtr predsPtr;
            /*
            allowed values of optionmask:
                     0:normal prediction
                     1:output margin instead of transformed value
                     2:output leaf index of trees instead of leaf value, note leaf index is unique per tree
                     4:output feature contributions to individual predictions

            // using `0` for ntreeLimit means use all the trees
            */

            var errp = WrappedXGBoostInterface.XGBoosterPredict(_handle, test.Handle, 0, 0, 0, out predsLen, out predsPtr);
            if (errp == -1)
            {
                string reason = WrappedXGBoostInterface.XGBGetLastError();
                throw new XGBoostDLLException(reason);
            }
            return XGBoostInterfaceUtils.GetPredictionsArray(predsPtr, predsLen);
        }

#if false
	// Should enable XGBoosterSaveModelToBuffer
        [BestFriend]
        internal unsafe string GetModelString()
        {
            int bufLen = 2 << 15;
            byte[] buffer = new byte[bufLen];
            int size = 0;
            fixed (byte* ptr = buffer)
                LightGbmInterfaceUtils.Check(WrappedLightGbmInterface.BoosterSaveModelToString(Handle, 0, BestIteration, bufLen, ref size, ptr));
            // If buffer size is not enough, reallocate buffer and get again.
            if (size > bufLen)
            {
                bufLen = size;
                buffer = new byte[bufLen];
                fixed (byte* ptr = buffer)
                    LightGbmInterfaceUtils.Check(WrappedLightGbmInterface.BoosterSaveModelToString(Handle, 0, BestIteration, bufLen, ref size, ptr));
            }
            byte[] content = new byte[size];
            Array.Copy(buffer, content, size);
            fixed (byte* ptr = content)
                return LightGbmInterfaceUtils.GetString((IntPtr)ptr);
        }
#endif

        public void SetParameters(Dictionary<string, object> parameters)
        {
            // support internationalisation i.e. support floats with commas (e.g. 0,5F)
            var nfi = new NumberFormatInfo { NumberDecimalSeparator = "." };

            SetParameter("max_depth", ((int)parameters["max_depth"]).ToString());
            SetParameter("learning_rate", ((float)parameters["learning_rate"]).ToString(nfi));
            SetParameter("n_estimators", ((int)parameters["n_estimators"]).ToString());
            SetParameter("silent", ((bool)parameters["silent"]).ToString());
            SetParameter("objective", (string)parameters["objective"]);
            SetParameter("booster", (string)parameters["booster"]);
            SetParameter("tree_method", (string)parameters["tree_method"]);

            SetParameter("nthread", ((int)parameters["nthread"]).ToString());
            SetParameter("gamma", ((float)parameters["gamma"]).ToString(nfi));
            SetParameter("min_child_weight", ((int)parameters["min_child_weight"]).ToString());
            SetParameter("max_delta_step", ((int)parameters["max_delta_step"]).ToString());
            SetParameter("subsample", ((float)parameters["subsample"]).ToString(nfi));
            SetParameter("colsample_bytree", ((float)parameters["colsample_bytree"]).ToString(nfi));
            SetParameter("colsample_bylevel", ((float)parameters["colsample_bylevel"]).ToString(nfi));
            SetParameter("reg_alpha", ((float)parameters["reg_alpha"]).ToString(nfi));
            SetParameter("reg_lambda", ((float)parameters["reg_lambda"]).ToString(nfi));
            SetParameter("scale_pos_weight", ((float)parameters["scale_pos_weight"]).ToString(nfi));

            SetParameter("base_score", ((float)parameters["base_score"]).ToString(nfi));
            SetParameter("seed", ((int)parameters["seed"]).ToString());
            SetParameter("missing", ((float)parameters["missing"]).ToString(nfi));

            SetParameter("sample_type", (string)parameters["sample_type"]);
            SetParameter("normalize_type ", (string)parameters["normalize_type"]);
            SetParameter("rate_drop", ((float)parameters["rate_drop"]).ToString(nfi));
            SetParameter("one_drop", ((int)parameters["one_drop"]).ToString());
            SetParameter("skip_drop", ((float)parameters["skip_drop"]).ToString(nfi));

            if (parameters.TryGetValue("num_class", out var value))
            {
                numClass = (int)value;
                SetParameter("num_class", numClass.ToString());
            }

        }

        public void SetParameter(string name, string val)
        {
            var errp = WrappedXGBoostInterface.XGBoosterSetParam(_handle, name, val);

            if (errp == -1)
            {
                string reason = WrappedXGBoostInterface.XGBGetLastError();
                throw new XGBoostDLLException(reason);
            }
        }

        #region Create Models

        public string[] DumpModelEx(string fmap, int with_stats, string format)
        {
            int length;
            IntPtr treePtr;
            var intptrSize = IntPtr.Size;

            WrappedXGBoostInterface.XGBoosterDumpModelEx(_handle, fmap, with_stats, format, out length, out treePtr);

            var trees = new string[length];
            int readSize = 0;
            var handle2 = GCHandle.Alloc(treePtr, GCHandleType.Pinned);

            //iterate through the length of the tree ensemble and pull the strings out from the returned pointer's array of pointers. prepend python's api convention of adding booster[i] to the beginning of the tree
            for (var i = 0; i < length; i++)
            {
                var ipt1 = Marshal.ReadIntPtr(Marshal.ReadIntPtr(handle2.AddrOfPinnedObject()), intptrSize * i);
                string s = Marshal.PtrToStringAnsi(ipt1);
                trees[i] = string.Format("booster[{0}]\n{1}", i, s);
                var bytesToRead = (s.Length * 2) + IntPtr.Size;
                readSize += bytesToRead;
            }
            handle2.Free();
            return trees;
        }

        public void GetModel()
        {
            var ss = DumpModelEx("", with_stats: 0, format: "json");
            var boosterPattern = @"^booster\[\d+\]";
#if false
            List<TreeNode> ensemble = new List<TreeNode>(); // should probably return this
#endif

            for (int i = 0; i < ss.Length; i++)
            {
                var m = Regex.Matches(ss[i], boosterPattern, RegexOptions.IgnoreCase);
                if ((m.Count >= 1) && (m[0].Groups.Count >= 1))
                {
                    // every booster representation should match
                    var structString = ss[i].Substring(m[0].Groups[0].Value.Length);
                    var doc = JsonDocument.Parse(structString);
#if false
                    TreeNode t = TreeNode.Create(doc.RootElement);
                    ensemble.Add(t);
#else
                    //var table = new TablePopulator(doc);
#endif
                }
            }
        }

        private class TablePopulator
        {
            public Dictionary<int, List<JsonElement>> dict = new();
            public Dictionary<string, int> nodes = new();
            public Dictionary<string, string> lte = new();
            public Dictionary<string, string> gt = new();

            public TablePopulator(JsonElement elm)
            {
                PopulateTable(elm, 0);
            }

            public void PopulateTable(JsonElement elm, int level)
            {
                string nodeId = "";
                if (elm.TryGetProperty("nodeid", out JsonElement nodeidElm))
                {
                    // If this test fails, should probably bail, as the syntax of the booster is incorrect
                    nodeId = nodeidElm.ToString();
                    nodes.Add(nodeId, level);
                }

                if (!dict.ContainsKey(level))
                {
                    dict.Add(level, new List<JsonElement>());
                }

                if (elm.TryGetProperty("leaf", out JsonElement leafJsonNode))
                {
                    dict[level].Add(elm);
                }
                else if (elm.TryGetProperty("children", out JsonElement internalJsonNode))
                {
                    dict[level].Add(elm);
                    if (elm.TryGetProperty("yes", out JsonElement yesNodeId))
                    {
                        lte.Add(nodeId, yesNodeId.ToString());
                    }

                    if (elm.TryGetProperty("no", out JsonElement noNodeId))
                    {
                        gt.Add(nodeId, noNodeId.ToString());
                    }

                    foreach (var e in internalJsonNode.EnumerateArray())
                    {
                        PopulateTable(e, level + 1);
                    }
                }
                else
                {
                    throw new Exception("Invalid booster content");
                }
            }
        }
        #endregion

        #region IDisposable Support
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposed)
            {
                return;
            }
            WrappedXGBoostInterface.XGBoosterFree(_handle);
            disposed = true;
        }
        #endregion
    }
}

