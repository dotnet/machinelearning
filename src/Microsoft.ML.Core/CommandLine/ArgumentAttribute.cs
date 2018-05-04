// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// This is separated from CmdParser.cs

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
    public class ArgumentAttribute : Attribute
    {
        public enum VisibilityType
        {
            Everywhere,
            CmdLineOnly,
            EntryPointsOnly
        }

        private ArgumentType _type;
        private string _shortName;
        private string _helpText;
        private bool _hide;
        private double _sortOrder;
        private string _nullName;
        private bool _isInputFileName;
        private string _specialPurpose;
        private VisibilityType _visibility;
        private string _name;

        /// <summary>
        /// Allows control of command line parsing.
        /// </summary>
        /// <param name="type"> Specifies the error checking to be done on the argument. </param>
        public ArgumentAttribute(ArgumentType type)
        {
            _type = type;
            _sortOrder = 150;
        }

        /// <summary>
        /// The error checking to be done on the argument.
        /// </summary>
        public ArgumentType Type
        {
            get { return _type; }
        }

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
            get { return _shortName; }
            set
            {
                Contracts.Check(value == null || !(this is DefaultArgumentAttribute));
                _shortName = value;
            }
        }

        /// <summary>
        /// The help text for the argument.
        /// </summary>
        public string HelpText
        {
            get { return _helpText; }
            set { _helpText = value; }
        }

        public bool Hide
        {
            get { return _hide; }
            set { _hide = value; }
        }

        public double SortOrder
        {
            get { return _sortOrder; }
            set { _sortOrder = value; }
        }

        public string NullName
        {
            get { return _nullName; }
            set { _nullName = value; }
        }

        public bool IsInputFileName
        {
            get { return _isInputFileName; }
            set { _isInputFileName = value; }
        }

        /// <summary>
        /// Allows the GUI or other tools to inspect the intended purpose of the argument and pick a correct custom control.
        /// </summary>
        public string Purpose
        {
            get { return _specialPurpose; }
            set { _specialPurpose = value; }
        }

        public VisibilityType Visibility
        {
            get { return _visibility; }
            set { _visibility = value; }
        }

        public string Name
        {
            get { return _name; }
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

        public bool IsRequired
        {
            get { return ArgumentType.Required == (_type & ArgumentType.Required); }
        }
    }
}