// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.Onnx;

namespace Microsoft.ML.Transforms.Onnx
{
    public static class OnnxSessionOptionsExtensions
    {
        private const string OnnxSessionOptionsName = "OnnxSessionOptions";

        public static OnnxSessionOptions GetOnnxSessionOption(this IHostEnvironment env)
        {
            if (env is IHostEnvironmentInternal localEnvironment)
            {
                return localEnvironment.GetOptionOrDefault<OnnxSessionOptions>(OnnxSessionOptionsName);
            }

            throw new ArgumentException("No Onnx Session Options");
        }

        public static void SetOnnxSessionOption(this IHostEnvironment env, OnnxSessionOptions onnxSessionOptions)
        {
            if (env is IHostEnvironmentInternal localEnvironment)
            {
                localEnvironment.AddOrOverwriteOption(OnnxSessionOptionsName, onnxSessionOptions);
            }

            throw new ArgumentException("No Onnx Session Options");
        }
    }

    public class OnnxSessionOptions
    {
        /// <summary>
        /// Enables the use of the memory allocation patterns in the first Run() call for subsequent runs. Default = true.
        /// </summary>
        /// <value>returns enableMemoryPattern flag value</value>
        public bool EnableMemoryPattern = true;

        /// <summary>
        /// Path prefix to use for output of profiling data
        /// </summary>
        public string ProfileOutputPathPrefix = "onnxruntime_profile_";   // this is the same default in C++ implementation

        /// <summary>
        /// Enables profiling of InferenceSession.Run() calls. Default is false
        /// </summary>
        /// <value>returns _enableProfiling flag value</value>
        public bool EnableProfiling = false;

        /// <summary>
        ///  Set filepath to save optimized model after graph level transformations. Default is empty, which implies saving is disabled.
        /// </summary>
        /// <value>returns _optimizedModelFilePath flag value</value>
        public string OptimizedModelFilePath = "";

        /// <summary>
        /// Enables Arena allocator for the CPU memory allocations. Default is true.
        /// </summary>
        /// <value>returns _enableCpuMemArena flag value</value>
        public bool EnableCpuMemArena = true;

        /// <summary>
        /// Disables the per session threads. Default is true.
        /// This makes all sessions in the process use a global TP.
        /// </summary>
        public bool DisablePerSessionThreads = true;

        /// <summary>
        /// Log Id to be used for the session. Default is empty string.
        /// </summary>
        /// <value>returns _logId value</value>
        public string LogId = string.Empty;

        /// <summary>
        /// Log Severity Level for the session logs. Default = ORT_LOGGING_LEVEL_WARNING
        /// </summary>
        /// <value>returns _logSeverityLevel value</value>
        public OrtLoggingLevel LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;

        /// <summary>
        /// Log Verbosity Level for the session logs. Default = 0. Valid values are >=0.
        /// This takes into effect only when the LogSeverityLevel is set to ORT_LOGGING_LEVEL_VERBOSE.
        /// </summary>
        /// <value>returns _logVerbosityLevel value</value>
        public int LogVerbosityLevel = 0;

        /// <summary>
        /// Sets the number of threads used to parallelize the execution within nodes
        /// A value of 0 means ORT will pick a default
        /// </summary>
        /// <value>returns _intraOpNumThreads value</value>
        public int IntraOpNumThreads = 0;

        /// <summary>
        /// Sets the number of threads used to parallelize the execution of the graph (across nodes)
        /// If sequential execution is enabled this value is ignored
        /// A value of 0 means ORT will pick a default
        /// </summary>
        /// <value>returns _interOpNumThreads value</value>
        public int InterOpNumThreads = 0;

        /// <summary>
        /// Sets the graph optimization level for the session. Default is set to ORT_ENABLE_ALL.
        /// </summary>
        /// <value>returns _graphOptimizationLevel value</value>
        public GraphOptimizationLevel GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

        /// <summary>
        /// Sets the execution mode for the session. Default is set to ORT_SEQUENTIAL.
        /// See [ONNX_Runtime_Perf_Tuning.md] for more details.
        /// </summary>
        /// <value>returns _executionMode value</value>
        public ExecutionMode ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;

        public delegate SessionOptions CreateOnnxSessionOptions();

        public CreateOnnxSessionOptions CreateSessionOptions;
    }
}
