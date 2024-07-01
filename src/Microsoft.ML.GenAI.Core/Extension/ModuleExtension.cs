// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TorchSharp;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Core.Extension;

public static class ModuleExtension
{
    public static long GetSizeInBytes(this nn.Module model)
    {
        var stateDict = model.state_dict();
        long size = 0;
        foreach (var (_, value) in stateDict)
        {
            size += value.numel() * value.element_size();
        }

        return size;
    }

    public static Dictionary<string, long> GetSizeForEachDynamicLayerInBytes(this nn.Module model)
    {
        var stateDict = model.named_children();
        if (stateDict.Count() == 0)
        {
            return new();
        }
        else
        {
            var dict = new Dictionary<string, long>();

            foreach (var (key, value) in stateDict)
            {
                if (value is IDynamicLoadModule)
                {
                    dict[key] = value.GetSizeInBytes();
                }
                else
                {
                    var subDict = value.GetSizeForEachDynamicLayerInBytes();
                    foreach (var (subKey, subValue) in subDict)
                    {
                        dict[key + "." + subKey] = subValue;
                    }
                }
            }

            return dict;
        }
    }

    /// <summary>
    /// Quantize the module using zero-point int8 quantization.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="model"></param>
    public static void ToInt8QuantizeModule<T>(
        this T model)
        where T : nn.Module
    {
        if (model is IQuantizeModule quantized)
        {
            quantized.Int8();

            return;
        }

        foreach (var (_, value) in model.named_children())
        {
            if (value is IQuantizeModule quantizeModule)
            {
                quantizeModule.Int8();
            }
            else
            {
                value.ToInt8QuantizeModule();
            }
        }
    }

    /// <summary>
    /// Quantize the module using zero-point int4 quantization.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="model"></param>
    public static void ToInt4QuantizeModule<T>(
        this T model)
        where T : nn.Module
    {
        if (model is IQuantizeModule quantized)
        {
            quantized.Int4();

            return;
        }

        foreach (var (_, value) in model.named_children())
        {
            if (value is IQuantizeModule quantizeModule)
            {
                quantizeModule.Int4();
            }
            else
            {
                value.ToInt4QuantizeModule();
            }
        }
    }

    public static T ToDynamicLoadingModel<T>(
        this T model,
        Dictionary<string, string> deviceMap,
        string targetDevice)
        where T : nn.Module
    {
        if (deviceMap.Count == 0)
        {
            model.to(new Device(targetDevice));

            return model;
        }

        // for each module in the model, update device if it is IDynamicLoadModule
        foreach (var (key, value) in model.named_children())
        {
            if (value is IDynamicLoadModule dynamicModule)
            {
                var device = deviceMap[key];
                if (device != targetDevice)
                {
                    dynamicModule.LoadToDeviceFunc = (nn.Module module) =>
                    {
                        module.to(new Device(targetDevice));
                    };
                    dynamicModule.UnloadFromDeviceFunc = (nn.Module module) =>
                    {
                        module.to(new Device(device));
                    };
                }

                value.to(new Device(device));
            }
            else
            {
                var childrenDeviceMap = deviceMap.Where(x => x.Key.StartsWith($"{key}.")).ToDictionary(x => x.Key.Substring($"{key}.".Length), x => x.Value);
                value.ToDynamicLoadingModel(childrenDeviceMap, targetDevice);
            }
        }

        return model;
    }

    /// <summary>
    /// Infer the device map for each layer in the model.
    /// The device map is a dictionary where the key is the device id (e.g. "cuda:0") and the value is the memory size in bytes of the device.
    /// When inferring the device map, each layer in the model will be placed on the device in the order of the devices list.
    /// </summary>
    /// <param name="model"></param>
    /// <param name="devices">a list of device ids (e.g. ["cuda:0", "cpu", "disk"])</param>
    /// <param name="deviceSizeMapInByte">a map where the key is the device id (e.g. "cuda:0") and the value is the memory size in bytes of the device</param>
    /// <returns></returns>
    public static Dictionary<string, string> InferDeviceMapForEachLayer(
        this nn.Module model,
        string[] devices,
        Dictionary<string, long> deviceSizeMapInByte)
    {
        var layerSizeMap = model.GetSizeForEachDynamicLayerInBytes();
        var sizeToRemainOnEachDevice = 2 * layerSizeMap.Max(x => x.Value);
        var deviceMap = new Dictionary<string, string>();
        foreach (var device in devices)
        {
            long size = deviceSizeMapInByte[device];
            var remainingLayerSizeMap = layerSizeMap.Where(x => !deviceMap.ContainsKey(x.Key)).ToDictionary(x => x.Key, x => x.Value);
            // larger layer fit first
            foreach (var (key, value) in remainingLayerSizeMap.OrderByDescending(x => x.Value))
            {
                if (size >= value)
                {
                    deviceMap[key] = device;
                    size -= value;
                }

                if (size < sizeToRemainOnEachDevice)
                {
                    break;
                }
            }
        }

        return deviceMap;
    }

    internal static string Peek(this nn.Module model)
    {
        var sb = new StringBuilder();
        var stateDict = model.state_dict();
        // preview state_dict
        int i = 0;
        foreach (var (key, value) in stateDict.OrderBy(x => x.Key, StringComparer.OrdinalIgnoreCase))
        {
            var str = value.Peek(key);
            sb.AppendLine($"{i}: {str}");
            i++;
        }

        var res = sb.ToString();

        return res;
    }

    internal static string PeekShape(this nn.Module model)
    {
        var sb = new StringBuilder();
        var stateDict = model.state_dict();
        // preview state_dict
        int i = 0;
        foreach (var (key, value) in stateDict.OrderBy(x => x.Key, StringComparer.OrdinalIgnoreCase))
        {
            // shape str: [x, y, z]
            var shapeStr = string.Join(", ", value.shape);
            sb.AppendLine($"{i}: {key} shape: [{shapeStr}]");
            i++;
        }

        var res = sb.ToString();

        return res;
    }
}
