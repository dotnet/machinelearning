// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.ImageAnalytics.EntryPoints;
using System.Security.Principal;
using System.Security.AccessControl;

namespace Microsoft.ML.Transforms.TensorFlow
{
    public static class TensorFlowUtils
    {
        // This method is needed for the Pipeline API, since ModuleCatalog does not load entry points that are located
        // in assemblies that aren't directly used in the code. Users who want to use TensorFlow components will have to call
        // TensorFlowUtils.Initialize() before creating the pipeline.
        /// <summary>
        /// Initialize the TensorFlow environment. Call this method before adding TensorFlow components to a learning pipeline.
        /// </summary>
        public static void Initialize()
        {
            ImageAnalytics.Initialize();
        }

        internal static PrimitiveType Tf2MlNetType(TFDataType type)
        {
            switch (type)
            {
                case TFDataType.Float:
                    return NumberType.R4;
                case TFDataType.Double:
                    return NumberType.R8;
                case TFDataType.UInt16:
                    return NumberType.U2;
                case TFDataType.UInt8:
                    return NumberType.U1;
                case TFDataType.UInt32:
                    return NumberType.U4;
                case TFDataType.UInt64:
                    return NumberType.U8;
                default:
                    throw new NotSupportedException("TensorFlow type not supported.");
            }
        }

        internal static unsafe void FetchData<T>(IntPtr data, T[] result)
        {
            var size = result.Length;

            GCHandle handle = GCHandle.Alloc(result, GCHandleType.Pinned);
            IntPtr target = handle.AddrOfPinnedObject();

            Int64 sizeInBytes = size * Marshal.SizeOf((typeof(T)));
            Buffer.MemoryCopy(data.ToPointer(), target.ToPointer(), sizeInBytes, sizeInBytes);
            handle.Free();
        }

        internal static bool IsTypeSupported(TFDataType tfoutput)
        {
            switch (tfoutput)
            {
                case TFDataType.Float:
                case TFDataType.Double:
                case TFDataType.UInt8:
                case TFDataType.UInt16:
                case TFDataType.UInt32:
                case TFDataType.UInt64:
                    return true;
                default:
                    return false;
            }
        }

        // A TensorFlow frozen model is a single file. An un-frozen (SavedModel) on the other hand has a well-defined folder structure.
        // Given a modelPath, this utility method determines if we should treat it as a frozen model or not
        internal static bool IsFrozenTensorFlowModel(string modelPath)
        {
            FileAttributes attr = File.GetAttributes(modelPath);

            if (attr.HasFlag(FileAttributes.Directory))
                return false;
            else
                return true;
        }
        internal static void CreateTempDirectory(string tempDirPath)
        {
            //if directory exists, do nothing.
            if (Directory.Exists(tempDirPath))
                return;

            WindowsIdentity currentIdentity = null;
            try
            {
                currentIdentity = WindowsIdentity.GetCurrent();
            }
            catch (PlatformNotSupportedException)
            { }

            if (currentIdentity != null && new WindowsPrincipal(currentIdentity).IsInRole(WindowsBuiltInRole.Administrator))
            {
                // Create high integrity dir and set no delete policy for all files under the directory.
                // In case of failure, throw exception.
                CreateTempDirectoryWithAcl(tempDirPath, currentIdentity.User.ToString());
            }
            else
                Directory.CreateDirectory(tempDirPath);
        }

        private static void CreateTempDirectoryWithAcl(string dirPath, string identity)
        {
            // Dacl Sddl string:
            // D: Dacl type
            // D; Deny access
            // OI; Object inherit ace
            // SD; Standard delete function
            // wIdentity.User Sid of the given user.
            // A; Allow access
            // OICI; Object inherit, container inherit
            // FA File access
            // BA Built-in administrators
            // S: Sacl type
            // ML;; Mandatory Label
            // NW;;; No write policy
            // HI High integrity processes only
            string sddl = "D:(D;OI;SD;;;" + identity + ")(A;OICI;FA;;;BA)S:(ML;OI;NW;;;HI)";

            var dir = Directory.CreateDirectory(dirPath);
            DirectorySecurity dirSec = new DirectorySecurity();
            dirSec.SetSecurityDescriptorSddlForm(sddl);
            dirSec.SetAccessRuleProtection(true, false);  // disable inheritance
            dir.SetAccessControl(dirSec);

            // Cleaning out the directory, in case someone managed to sneak in between creation and setting ACL.
            DirectoryInfo dirInfo = new DirectoryInfo(dirPath);
            foreach (FileInfo file in dirInfo.GetFiles())
            {
                file.Delete();
            }
            foreach (DirectoryInfo subDirInfo in dirInfo.GetDirectories())
            {
                subDirInfo.Delete(true);
            }
        }
    }
}
