using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text.RegularExpressions;
using System.Text.Json;
using System.Runtime.InteropServices;

namespace Microsoft.ML.Trainers.XGBoost
{

public abstract class TreeNode
{
    private uint _nodeId;
    public uint NodeId { get => _nodeId; }
    public TreeNode(uint nodeId)
    {
        _nodeId = nodeId;
    }

    public static TreeNode Create(JsonElement jsonElement)
    {
        if (jsonElement.TryGetProperty("leaf", out JsonElement leafJsonNode)) {
            return TreeLeafNode.Create(jsonElement);
        } else if (jsonElement.TryGetProperty("children", out JsonElement internalJsonNode)) {
            return TreeInternalNode.Create(jsonElement);
        } else {
            throw new Exception("Invalid booster content");
        }
    }
}

public class TreeLeafNode : TreeNode
{
    private float _leafValue;
    public float LeafValue { get => _leafValue; }
    public TreeLeafNode(uint nodeId, float leafValue) : base(nodeId)
    {
        _leafValue = leafValue;
    }

    public static new TreeLeafNode Create(JsonElement jsonElement)
    {
        try {
            var nodeId = jsonElement.GetProperty("nodeid").GetUInt32();
            float leafValue = (float)(jsonElement.GetProperty("leaf").GetDouble());
            return new TreeLeafNode(nodeId, leafValue);
        } catch (Exception ex) {
            throw ex;
        }
    }
}

public class TreeInternalNode : TreeNode
{
    private uint _depth;
    private string _split;
    private float _split_condition;
    private uint _yes;
    private uint _no;
    private uint _missing;
    private Dictionary<uint, TreeNode> _children;

    public uint Depth { get => _depth; }
    public string Split { get => _split; }
    public float SplitCondition { get => _split_condition; }
    public uint Yes { get => _yes; }
    public uint No { get => _no; }
    public Dictionary<uint, TreeNode> Children { get => _children; }

    public TreeInternalNode(uint nodeId, uint depth, string split, float split_condition, uint missing, uint yes, uint no)
    : base (nodeId)
    {
        _depth = depth;
        _split = split;
        _split_condition = split_condition;
        _missing = missing;
        _yes = yes;
        _no = no;
        _children = new Dictionary<uint, TreeNode>();
    }

    public static new TreeInternalNode Create(JsonElement jsonElement)
    {
        try {
            var nodeId = jsonElement.GetProperty("nodeid").GetUInt32();
            var depth = jsonElement.GetProperty("depth").GetUInt32();
            var split = jsonElement.GetProperty("split").GetString();
            var split_condition = (float)(jsonElement.GetProperty("split_condition").GetDouble());
            var yes = jsonElement.GetProperty("yes").GetUInt32();
            var no = jsonElement.GetProperty("no").GetUInt32();
            var missing = jsonElement.GetProperty("missing").GetUInt32();
            var ret = new TreeInternalNode(nodeId, depth, split, split_condition, missing, yes, no);
            foreach(var e in jsonElement.GetProperty("children").EnumerateArray()) {
                var t = TreeNode.Create(e);
                ret.Children.Add(t.NodeId, t);
            }
            return ret;
        } catch (Exception ex) {
            throw ex;
        }
    }
}
}