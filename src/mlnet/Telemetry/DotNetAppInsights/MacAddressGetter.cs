// Copyright (c) .NET Foundation and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using System.Diagnostics;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using System.Net.NetworkInformation;
using System.ComponentModel;
using Microsoft.DotNet.Cli.Utils;

namespace Microsoft.DotNet.Cli.Telemetry
{
    internal static class MacAddressGetter
    {
        private const string MacRegex = @"(?:[a-z0-9]{2}[:\-]){5}[a-z0-9]{2}";
        private const string ZeroRegex = @"(?:00[:\-]){5}00";
        private const int ErrorFileNotFound = 0x2;
        public static string GetMacAddress()
        {
            try
            {
                var shelloutput = GetShellOutMacAddressOutput();
                if (shelloutput == null)
                {
                    return null;
                }

                return ParseMACAddress(shelloutput);
            }
            catch (Win32Exception e)
            {
                if (e.NativeErrorCode == ErrorFileNotFound)
                {
                    return GetMacAddressByNetworkInterface();
                }
                else
                {
                    throw;
                }
            }
        }

        private static string ParseMACAddress(string shelloutput)
        {
            string macAddress = null;
            foreach (Match match in Regex.Matches(shelloutput, MacRegex, RegexOptions.IgnoreCase))
            {
                if (!Regex.IsMatch(match.Value, ZeroRegex))
                {
                    macAddress = match.Value;
                    break;
                }
            }

            if (macAddress != null)
            {
                return macAddress;
            }
            return null;
        }

        private static string GetIpCommandOutput()
        {
            var ipResult = new ProcessStartInfo
            {
                FileName = "ip",
                Arguments = "link",
                UseShellExecute = false
            }.ExecuteAndCaptureOutput(out string ipStdOut, out string ipStdErr);

            if (ipResult == 0)
            {
                return ipStdOut;
            }
            else
            {
                return null;
            }
        }

        private static string GetShellOutMacAddressOutput()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                var result = new ProcessStartInfo
                {
                    FileName = "getmac.exe",
                    UseShellExecute = false
                }.ExecuteAndCaptureOutput(out string stdOut, out string stdErr);

                if (result == 0)
                {
                    return stdOut;
                }
                else
                {
                    return null;
                }
            }
            else
            {
                try
                {
                    var ifconfigResult = new ProcessStartInfo
                    {
                        FileName = "ifconfig",
                        Arguments = "-a",
                        UseShellExecute = false
                    }.ExecuteAndCaptureOutput(out string ifconfigStdOut, out string ifconfigStdErr);

                    if (ifconfigResult == 0)
                    {
                        return ifconfigStdOut;
                    }
                    else
                    {
                        return GetIpCommandOutput();
                    }
                }
                catch (Win32Exception e)
                {
                    if (e.NativeErrorCode == ErrorFileNotFound)
                    {
                        return GetIpCommandOutput();
                    }
                    else
                    {
                        throw;
                    }
                }
            }
        }

        private static string GetMacAddressByNetworkInterface()
        {
            return GetMacAddressesByNetworkInterface().FirstOrDefault();
        }

        private static List<string> GetMacAddressesByNetworkInterface()
        {
            NetworkInterface[] nics = NetworkInterface.GetAllNetworkInterfaces();
            var macs = new List<string>();

            if (nics == null || nics.Length < 1)
            {
                macs.Add(string.Empty);
                return macs;
            }

            foreach (NetworkInterface adapter in nics)
            {
                IPInterfaceProperties properties = adapter.GetIPProperties();

                PhysicalAddress address = adapter.GetPhysicalAddress();
                byte[] bytes = address.GetAddressBytes();
                macs.Add(string.Join("-", bytes.Select(x => x.ToString("X2"))));
                if (macs.Count >= 10)
                {
                    break;
                }
            }
            return macs;
        }
    }
}
