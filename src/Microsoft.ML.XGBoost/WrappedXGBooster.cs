// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.CpuMath;
using Microsoft.ML.Internal.Utilities;
//using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Numeric;
using Microsoft.ML.Runtime;

using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms;

namespace Microsoft.ML.Trainers.XGBoost
{
    /// <summary>
    /// Wrapper of Booster object of XGBoost.
    /// </summary>
    internal class Booster : IDisposable
    {
#pragma warning disable MSML_PrivateFieldName // Private field name not in: _camelCase format
        private bool disposed;
        private readonly IntPtr _handle;
        private const int normalPrediction = 0; // Value for the optionMask in prediction
        private int numClass = 1;
#pragma warning restore MSML_PrivateFieldName

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

        public unsafe void DumpAttributes()
        {
#pragma warning disable MSML_PrivateFieldName // Private field name not in: _camelCase format
            ulong attrsLen;
            byte** attrs;
#pragma warning restore MSML_PrivateFieldName

            var errp = WrappedXGBoostInterface.XGBoosterGetAttrNames(_handle, out attrsLen, out attrs);
            if (errp == -1)
            {
                string reason = WrappedXGBoostInterface.XGBGetLastError();
                throw new XGBoostDLLException(reason);
            }
            // TODO: Return Dictionary<string, object>

            var options = new JsonSerializerOptions
            {
                Converters = { new Utils.DictionaryStringObjectConverter() }
            };
            // return JsonSerializer.Deserialize<Dictionary<string, object>>(attrs, options);

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

        public class XgbNode
        {
#pragma warning disable MSML_GeneralName // Private field name not in: _camelCase format
            public int nodeid { get; set; }
#pragma warning restore MSML_GeneralName
        }

        public class XgbNodeLeaf : XgbNode
        {
#pragma warning disable MSML_GeneralName // Private field name not in: _camelCase format
            public float leaf { get; set; }
#pragma warning restore MSML_GeneralName // Private field name not in: _camelCase format
        }

        public class XgbNodeSplit : XgbNode
        {
#pragma warning disable MSML_GeneralName // Private field name not in: _camelCase format
            public int depth { get; set; }
            public int split { get; set; }
            public float split_condition { get; set; }
            public int yes { get; set; }
            public int no { get; set; }
            public float missing { get; set; }
#pragma warning restore MSML_GeneralName // Private field name not in: _camelCase format
        }

        class TablePopulator
        {
            public Dictionary<int, XgbNodeLeaf> Leaves = new();
            public Dictionary<int, XgbNodeSplit> Decisions = new();

            public TablePopulator(string jsonFragment)
            {
                PopulateTable(JsonDocument.Parse(jsonFragment).RootElement);
            }

            public void PopulateTable(JsonElement elm)
            {
                int nodeId = default;
                if (elm.TryGetProperty("nodeid", out JsonElement nodeidElm))
                {
                    // If this test fails, should probably bail, as the syntax of the booster is incorrect
                    nodeId = nodeidElm.GetInt32();
                }

                if (elm.TryGetProperty("leaf", out JsonElement leafJsonNode))
                {
                    Leaves.Add(nodeId, new XgbNodeLeaf { nodeid = nodeId, leaf = leafJsonNode.GetSingle() });
                }
                else if (elm.TryGetProperty("children", out JsonElement internalJsonNode))
                {
                    var node = new XgbNodeSplit { nodeid = nodeId };
                    Decisions.Add(nodeId, node);
                    if (elm.TryGetProperty("yes", out JsonElement yesNodeId))
                    {
                        node.yes = yesNodeId.GetInt32();
                    }

                    if (elm.TryGetProperty("no", out JsonElement noNodeId))
                    {
                        node.no = noNodeId.GetInt32();
                    }

                    // TODO: missing "missing"
                    if (elm.TryGetProperty("split", out JsonElement splitFeature))
                    {
                        var candidate = splitFeature.GetString();
                        if (Regex.IsMatch(candidate, "f[0-9]+"))
                        {
                            if (int.TryParse(candidate.Substring(1), out int splitFeatureIndex))
                            {
                                node.split = splitFeatureIndex;
                            }
                        }
                    }

                    if (elm.TryGetProperty("split_condition", out JsonElement splitThreshold))
                    {
                        node.split_condition = splitThreshold.GetSingle();
                    }


                    foreach (var e in internalJsonNode.EnumerateArray())
                    {
                        PopulateTable(e);
                    }
                }
                else
                {
                    throw new Exception("Invalid booster content");
                }
            }

#if false
            public (int[], int[]) Sequentialize()
#else
            public InternalRegressionTree Sequentialize()
#endif
            {
                int nextNode = 0;
                int nextLeaf = 1;
                Dictionary<int, int> mapNodes = new(); // internal nodes original id-to-seq id
                Dictionary<int, int> mapLeaves = new(); // leaves original id-to-seq-id map

                foreach (var n in Decisions)
                {
                    if (!mapNodes.ContainsKey(n.Key))
                    {
                        mapNodes.Add(n.Key, nextNode++);
                    }
                }
                foreach (var n in Leaves)
                {
                    if (!mapLeaves.ContainsKey(n.Key))
                    {
                        mapLeaves.Add(n.Key, nextLeaf++);
                    }
                }

                int[] lte = new int[mapNodes.Count];
                int[] gt = new int[mapNodes.Count];
                int[] splitFeatures = new int[mapNodes.Count];
                float[] rawThresholds = new float[mapNodes.Count];
                double[] leafValues = new double[mapLeaves.Count];

                // TODO: Can this be done with LINQ in a better way?
                foreach (var n in Decisions)
                {
                    if (Leaves.ContainsKey(n.Value.yes))
                    {
                        lte[mapNodes[n.Key]] = -mapLeaves[n.Value.yes];
                    }
                    else
                    {
                        lte[mapNodes[n.Key]] = mapNodes[n.Value.yes];
                    }

                    if (Leaves.ContainsKey(n.Value.no))
                    {
                        gt[mapNodes[n.Key]] = -mapLeaves[n.Value.no];
                    }
                    else
                    {
                        gt[mapNodes[n.Key]] = mapNodes[n.Value.no];
                    }
                    splitFeatures[mapNodes[n.Key]] = n.Value.split;
                    rawThresholds[mapNodes[n.Key]] = n.Value.split_condition;
                    // TODO: The rest
                }

                foreach (var l in Leaves)
                {
                    leafValues[mapLeaves[l.Key] - 1] = l.Value.leaf;
                }

                var tree = InternalRegressionTree.Create(Leaves.Count,
                splitFeatures,
                new double[Leaves.Count - 1], // double[] splitGain
                rawThresholds,
                new float[Leaves.Count - 1], // float[] defaultValueForMissing // FIXME
                lte,
                gt,
                leafValues,
                new int[Leaves.Count - 1][], // int[][] categoricalSplitFeatures
                new bool[Leaves.Count - 1] // bool[] categoricalSplit
                );

                return tree;
            }
        }

#if true
        public unsafe InternalTreeEnsemble DumpModel()
        {
            InternalTreeEnsemble ensemble = new InternalTreeEnsemble();
#pragma warning disable MSML_ParameterLocalVarName // Parameter or local variable name not standard
            ulong boosters_len;
            byte** booster_raw_arr;
#pragma warning restore MSML_ParameterLocalVarName // Parameter or local variable name not standard
            var errp = WrappedXGBoostInterface.XGBoosterDumpModelEx(_handle, "", 0, "json", out boosters_len, out booster_raw_arr);
            if (errp == -1)
            {
                string reason = WrappedXGBoostInterface.XGBGetLastError();
                throw new XGBoostDLLException(reason);
            }

            var result = new string[boosters_len];

            for (ulong i = 0; i < boosters_len; ++i)
            {
                result[i] = Marshal.PtrToStringUTF8((nint)booster_raw_arr[i]) ?? "";
                Console.WriteLine($"**** Trying to parse booster {i}, which is {result[i]}");
                Console.WriteLine($"**** Calling the TablePopulator on booster {i}..");
                var table = new TablePopulator(result[i]);
                var tree = table.Sequentialize();
                ensemble.AddTree(tree);
            }

            Console.WriteLine($"**** The length of the boosters are {result.Length}.");
            Console.WriteLine($"**** The number of trees in the ensemble are {ensemble.NumTrees}.");
            return ensemble;

        }

        public string DumpConfig()
        {
#pragma warning disable MSML_ParameterLocalVarName // Parameter or local variable name not standard
            ulong config_len;
#pragma warning restore MSML_ParameterLocalVarName // Parameter or local variable name not standard
            string result = default;
            unsafe
            {
#pragma warning disable MSML_ParameterLocalVarName // Parameter or local variable name not standard
                byte* config_result;
#pragma warning restore MSML_ParameterLocalVarName // Parameter or local variable name not standard
                var errp = WrappedXGBoostInterface.XGBoosterSaveJsonConfig(_handle, out config_len, &config_result);
                if (errp == -1)
                {
                    string reason = WrappedXGBoostInterface.XGBGetLastError();
                    throw new XGBoostDLLException(reason);
                }
                result = Marshal.PtrToStringUTF8((nint)config_result) ?? "";
            }
            return result;

        }
#else
#if false
#pragma warning disable MSML_ParameterLocalVarName
        public string[] DumpModelEx(string fmap, int with_stats, string format)
#pragma warning restore MSML_ParameterLocalVarName
        {
            int length;
            IntPtr treePtr;
            var intptrSize = IntPtr.Size;

            WrappedXGBoostInterface.XGBoosterDumpModelEx(_handle, fmap, with_stats, format, out (ulong)length, out (byte**)treePtr);

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
#else
            public unsafe void DumpModel()
        {
            ulong boostersLen;
            byte** boosterRawArr;
            var errp = WrappedXGBoostInterface.XGBoosterDumpModelEx(_handle, "", 0, "json", out boostersLen, out boosterRawArr);
            if (errp == -1)
            {
                string reason = WrappedXGBoostInterface.XGBGetLastError();
                throw new XGBoostDLLException(reason);
            }

            var result = new string[boostersLen];
#if false
            var boosterPattern = @"^booster\[\d+\]";
#endif

            for (ulong i = 0; i < boostersLen; ++i)
            {
                result[i] = Marshal.PtrToStringUTF8((nint)boosterRawArr[i]) ?? "";
                Console.WriteLine($"**** Trying to parse booster {i}, which is {result[i]}");
                var doc = JsonDocument.Parse(result[i]);
#if false
                    TreeNode t = TreeNode.Create(doc.RootElement);
                    ensemble.Add(t);
#else
                //var table = new TablePopulator(doc);
                Console.WriteLine($"**** Booster {i} has an element of type: {doc.RootElement.ValueKind}.");
#endif
            }

            //Console.WriteLine($"**** The length of the boosters are {result.Length}.");
            //Console.WriteLine($"**** The first booster is {result[0]}.");

        }
#endif

        public void GetModel()
        {
#if false
            var ss = DumpModelEx("", with_stats: 0, format: "json");
#else
            string ss = DumpModel();
#endif
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
#pragma warning disable MSML_GeneralName
            public Dictionary<int, List<JsonElement>> dict = new();
            public Dictionary<string, int> nodes = new();
            public Dictionary<string, string> lte = new();
            public Dictionary<string, string> gt = new();
#pragma warning restore MSML_GeneralName

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
#endif
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

