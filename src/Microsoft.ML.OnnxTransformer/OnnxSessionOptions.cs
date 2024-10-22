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
                localEnvironment.SetOption(OnnxSessionOptionsName, onnxSessionOptions);
            }
            else
                throw new ArgumentException("No Onnx Session Options");
        }
    }

    public sealed class OnnxSessionOptions
    {
        internal void CopyTo(SessionOptions sessionOptions)
        {
            sessionOptions.EnableMemoryPattern = EnableMemoryPattern;
            sessionOptions.ProfileOutputPathPrefix = ProfileOutputPathPrefix;
            sessionOptions.EnableProfiling = EnableProfiling;
            sessionOptions.OptimizedModelFilePath = OptimizedModelFilePath;
            sessionOptions.EnableCpuMemArena = EnableCpuMemArena;
            if (!PerSessionThreads)
                sessionOptions.DisablePerSessionThreads();
            sessionOptions.LogId = LogId;
            sessionOptions.LogSeverityLevel = LogSeverityLevel;
            sessionOptions.LogVerbosityLevel = LogVerbosityLevel;
            sessionOptions.InterOpNumThreads = InterOpNumThreads;
            sessionOptions.IntraOpNumThreads = IntraOpNumThreads;
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel;
            sessionOptions.ExecutionMode = ExecutionMode;
        }

        /// <summary>
        /// Enables the use of the memory allocation patterns in the first Run() call for subsequent runs. Default = true.
        /// </summary>
#pragma warning disable MSML_NoInstanceInitializers // No initializers on instance fields or properties
        public bool EnableMemoryPattern { get; set; } = true;

        /// <summary>
        /// Path prefix to use for output of profiling data
        /// </summary>
        public string ProfileOutputPathPrefix { get; set; } = "onnxruntime_profile_";   // this is the same default in C++ implementation

        /// <summary>
        /// Enables profiling of InferenceSession.Run() calls. Default is false
        /// </summary>
        public bool EnableProfiling { get; set; } = false;

        /// <summary>
        ///  Set filepath to save optimized model after graph level transformations. Default is empty, which implies saving is disabled.
        /// </summary>
        public string OptimizedModelFilePath { get; set; } = string.Empty;

        /// <summary>
        /// Enables Arena allocator for the CPU memory allocations. Default is true.
        /// </summary>
        public bool EnableCpuMemArena { get; set; } = true;

        /// <summary>
        /// Per session threads. Default is true.
        /// If false this makes all sessions in the process use a global TP.
        /// </summary>
        public bool PerSessionThreads { get; set; } = true;

        /// <summary>
        /// Sets the number of threads used to parallelize the execution within nodes
        /// A value of 0 means ORT will pick a default. Only used when <see cref="PerSessionThreads"/> is false.
        /// </summary>
        public int GlobalIntraOpNumThreads { get; set; } = 0;

        /// <summary>
        /// Sets the number of threads used to parallelize the execution of the graph (across nodes)
        /// If sequential execution is enabled this value is ignored
        /// A value of 0 means ORT will pick a default. Only used when <see cref="PerSessionThreads"/> is false.
        /// </summary>
        public int GlobalInterOpNumThreads { get; set; } = 0;

        /// <summary>
        /// Log Id to be used for the session. Default is empty string.
        /// </summary>
        public string LogId { get; set; } = string.Empty;

        /// <summary>
        /// Log Severity Level for the session logs. Default = ORT_LOGGING_LEVEL_WARNING
        /// </summary>
        public OrtLoggingLevel LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;

        /// <summary>
        /// Log Verbosity Level for the session logs. Default = 0. Valid values are >=0.
        /// This takes into effect only when the LogSeverityLevel is set to ORT_LOGGING_LEVEL_VERBOSE.
        /// </summary>
        public int LogVerbosityLevel { get; set; } = 0;

        /// <summary>
        /// Sets the number of threads used to parallelize the execution within nodes
        /// A value of 0 means ORT will pick a default
        /// </summary>
        public int IntraOpNumThreads { get; set; } = 0;

        /// <summary>
        /// Sets the number of threads used to parallelize the execution of the graph (across nodes)
        /// If sequential execution is enabled this value is ignored
        /// A value of 0 means ORT will pick a default
        /// </summary>
        public int InterOpNumThreads { get; set; } = 0;

        /// <summary>
        /// Sets the graph optimization level for the session. Default is set to ORT_ENABLE_ALL.
        /// </summary>
        public GraphOptimizationLevel GraphOptimizationLevel { get; set; } = GraphOptimizationLevel.ORT_ENABLE_ALL;

        /// <summary>
        /// Sets the execution mode for the session. Default is set to ORT_SEQUENTIAL.
        /// See [ONNX_Runtime_Perf_Tuning.md] for more details.
        /// </summary>
        public ExecutionMode ExecutionMode { get; set; } = ExecutionMode.ORT_SEQUENTIAL;
#pragma warning restore MSML_NoInstanceInitializers // No initializers on instance fields or properties

        public delegate SessionOptions CreateOnnxSessionOptions();

        public CreateOnnxSessionOptions CreateSessionOptions { get; set; }
    }
}
