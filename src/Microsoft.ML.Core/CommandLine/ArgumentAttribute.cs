// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;

namespace Microsoft.ML.Runtime.CommandLine
{
    /// <summary>
    /// Allows control of command line parsing.
    /// Attach this attribute to instance fields of types used
    /// as the destination of command line argument parsing.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field)]
    [BestFriend]
    internal class ArgumentAttribute : Attribute
    {
        public enum VisibilityType
        {
            Everywhere,
            CmdLineOnly,
            EntryPointsOnly
        }

        private string _shortName;
        private string _name;

        /// <summary>
        /// Allows control of command line parsing.
        /// </summary>
        /// <param name="type"> Specifies the error checking to be done on the argument. </param>
        public ArgumentAttribute(ArgumentType type)
        {
            Type = type;
            SortOrder = 150;
        }

        /// <summary>
        /// The error checking to be done on the argument.
        /// </summary>
        public ArgumentType Type { get; }

        /// <summary>
        /// The short name(s) of the argument.
        /// Set to null means use the default short name if it does not
        /// conflict with any other parameter name.
        /// Set to String.Empty for no short name.
        /// More than one short name can be separated by commas or spaces.
        /// This property should not be set for DefaultArgumentAttributes.
        /// </summary>
        public string ShortName
        {
            get => _shortName;
            set
            {
                Contracts.Check(value == null || !(this is DefaultArgumentAttribute));
                _shortName = value;
            }
        }

        /// <summary>
        /// The help text for the argument.
        /// </summary>
        public string HelpText { get; set; }

        public bool Hide { get; set; }

        public double SortOrder { get; set; }

        public string NullName { get; set; }

        public bool IsInputFileName { get; set; }

        /// <summary>
        /// Allows the GUI or other tools to inspect the intended purpose of the argument and pick a correct custom control.
        /// </summary>
        public string Purpose { get; set; }

        public VisibilityType Visibility { get; set; }

        public string Name
        {
            get => _name;
            set { _name = string.IsNullOrWhiteSpace(value) ? null : value; }
        }

        public string[] Aliases
        {
            get
            {
                if (string.IsNullOrWhiteSpace(_shortName))
                    return null;
                return _shortName.Split(',').Select(name => name.Trim()).ToArray();
            }
        }

        public bool IsRequired => ArgumentType.Required == (Type & ArgumentType.Required);

        public Type SignatureType { get; set; }
    }
}