//////////////////////////////////////////////////////////////////////////////
//    Command Line Argument Parser
//    ----------------------------
//    Usage
//    -----
//
//    Parsing command line arguments to a console application is a common problem.
//    This library handles the common task of reading arguments from a command line
//    and filling in the values in a type.
//
//    To use this library, define a class whose fields represent the data that your
//    application wants to receive from arguments on the command line. Then call
//    CommandLine.ParseArguments() to fill the object with the data
//    from the command line. Each field in the class defines a command line argument.
//    The type of the field is used to validate the data read from the command line.
//    The name of the field defines the name of the command line option.
//
//    The parser can handle fields of the following types:
//
//    - string
//    - int
//    - uint
//    - bool
//    - enum
//    - array of the above type
//
//    For example, suppose you want to read in the argument list for wc (word count).
//    wc takes three optional boolean arguments: -l, -w, and -c and a list of files.
//
//    You could parse these arguments using the following code:
//
//    class WCArguments
//    {
//        public bool lines;
//        public bool words;
//        public bool chars;
//        public string[] files;
//    }
//
//    class WC
//    {
//        static void Main(string[] args)
//        {
//            if (CommandLine.ParseArgumentsWithUsage(args, parsedArgs))
//            {
//            //     insert application code here
//            }
//        }
//    }
//
//    So you could call this aplication with the following command line to count
//    lines in the foo and bar files:
//
//        wc.exe /lines /files:foo /files:bar
//
//    The program will display the following usage message when bad command line
//    arguments are used:
//
//        wc.exe -x
//
//    Unrecognized command line argument '-x'
//        /lines[+|-]                         short form /l
//        /words[+|-]                         short form /w
//        /chars[+|-]                         short form /c
//        /files=<string>                     short form /f
//        @<file>                             Read response file for more options
//
//    That was pretty easy. However, you realy want to omit the "/files:" for the
//    list of files. The details of field parsing can be controled using custom
//    attributes. The attributes which control parsing behaviour are:
//
//    ArgumentAttribute
//        - controls short name, long name, required, allow duplicates, default value
//        and help text
//    DefaultArgumentAttribute
//        - allows omition of the "/name".
//        - This attribute is allowed on only one field in the argument class.
//
//    So for the wc.exe program we want this:
//
//    using System;
//    using Utilities;
//
//    class WCArguments
//    {
//        [Argument(ArgumentType.AtMostOnce, HelpText="Count number of lines in the input text.")]
//        public bool lines;
//        [Argument(ArgumentType.AtMostOnce, HelpText="Count number of words in the input text.")]
//        public bool words;
//        [Argument(ArgumentType.AtMostOnce, HelpText="Count number of chars in the input text.")]
//        public bool chars;
//        [DefaultArgument(ArgumentType.MultipleUnique, HelpText="Input files to count.")]
//        public string[] files;
//    }
//
//    class WC
//    {
//        static void Main(string[] args)
//        {
//            WCArguments parsedArgs = new WCArguments();
//            if (CommandLine.ParseArgumentsWithUsage(args, parsedArgs))
//            {
//            //     insert application code here
//            }
//        }
//    }
//
//
//
//    So now we have the command line we want:
//
//        wc.exe /lines foo bar
//
//    This will set lines to true and will set files to an array containing the
//    strings "foo" and "bar".
//
//    The new usage message becomes:
//
//        wc.exe -x
//
//    Unrecognized command line argument '-x'
//    /lines[+|-]  Count number of lines in the input text. (short form /l)
//    /words[+|-]  Count number of words in the input text. (short form /w)
//    /chars[+|-]  Count number of chars in the input text. (short form /c)
//    @<file>      Read response file for more options
//    <files>      Input files to count. (short form /f)
//
//    If you want more control over how error messages are reported, how /help is
//    dealt with, etc you can instantiate the CommandLine.Parser class.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.CommandLine
{
    /// <summary>
    /// Used to control parsing of command line arguments.
    /// </summary>
    [Flags]
    public enum ArgumentType
    {
        /// <summary>
        /// Indicates that this field is required. An error will be displayed
        /// if it is not present when parsing arguments.
        /// </summary>
        Required = 0x01,

        /// <summary>
        /// Only valid in conjunction with Multiple.
        /// Duplicate values will result in an error.
        /// </summary>
        Unique = 0x02,

        /// <summary>
        /// Inidicates that the argument may be specified more than once.
        /// Only valid if the argument is a collection
        /// </summary>
        Multiple = 0x04,

        /// <summary>
        /// The default type for non-collection arguments.
        /// The argument is not required, but an error will be reported if it is specified more than once.
        /// </summary>
        AtMostOnce = 0x00,

        /// <summary>
        /// For non-collection arguments, when the argument is specified more than
        /// once no error is reported and the value of the argument is the last
        /// value which occurs in the argument list.
        /// </summary>
        LastOccurenceWins = Multiple,

        /// <summary>
        /// The default type for collection arguments.
        /// The argument is permitted to occur multiple times, but duplicate
        /// values will cause an error to be reported.
        /// </summary>
        MultipleUnique = Multiple | Unique,
    }

    /// <summary>
    /// Indicates that this argument is the default argument.
    /// '/' or '-' prefix only the argument value is specified.
    /// The ShortName property should not be set for DefaultArgumentAttribute
    /// instances. The LongName property is used for usage text only and
    /// does not affect the usage of the argument.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field)]
    public class DefaultArgumentAttribute : ArgumentAttribute
    {
        /// <summary>
        /// Indicates that this argument is the default argument.
        /// </summary>
        /// <param name="type"> Specifies the error checking to be done on the argument. </param>
        public DefaultArgumentAttribute(ArgumentType type)
            : base(type)
        {
        }
    }

    /// <summary>
    /// On an enum value - indicates that the value should not be shown in help or UI.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field)]
    public class HideEnumValueAttribute : Attribute
    {
        public HideEnumValueAttribute()
        {
        }
    }

    /// <summary>
    /// On an enum value - specifies the display name.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field)]
    public class EnumValueDisplayAttribute : Attribute
    {
        public readonly string Name;

        public EnumValueDisplayAttribute(string name)
        {
            Name = name;
        }
    }

    /// <summary>
    /// A delegate used in error reporting.
    /// </summary>
    public delegate void ErrorReporter(string message);

    [Flags]
    public enum SettingsFlags
    {
        None = 0x00,

        ShortNames = 0x01,
        NoSlashes = 0x02,
        NoUnparse = 0x04,

        Default = ShortNames | NoSlashes
    }

    /// <summary>
    /// Parser for command line arguments.
    ///
    /// The parser specification is infered from the instance fields of the object
    /// specified as the destination of the parse.
    /// Valid argument types are: int, uint, string, bool, enums
    /// Also argument types of Array of the above types are also valid.
    ///
    /// Error checking options can be controlled by adding a ArgumentAttribute
    /// to the instance fields of the destination object.
    ///
    /// At most one field may be marked with the DefaultArgumentAttribute
    /// indicating that arguments without a '-' or '/' prefix will be parsed as that argument.
    ///
    /// If not specified then the parser will infer default options for parsing each
    /// instance field. The default long name of the argument is the field name. The
    /// default short name is the first character of the long name. Long names and explicitly
    /// specified short names must be unique. Default short names will be used provided that
    /// the default short name does not conflict with a long name or an explicitly
    /// specified short name.
    ///
    /// Arguments which are array types are collection arguments. Collection
    /// arguments can be specified multiple times.
    /// </summary>
    public sealed class CmdParser
    {
        private const int SpaceBeforeParam = 2;
        private readonly ErrorReporter _reporter;
        private readonly IHost _host;
        // REVIEW: _catalog should be part of environment and can be get through _host.
        private Lazy<ModuleCatalog> _catalog;

        /// <summary>
        /// Parses a command line. This assumes that the exe name has been stripped off.
        /// Errors are output on Console.Error.
        /// Use ArgumentAttributes to control parsing behaviour.
        /// </summary>
        /// <param name="env"> The host environment</param>
        /// <param name="settings">The command line</param>
        /// <param name="destination">The object to receive the options</param>
        /// <returns>true if no errors were detected</returns>
        public static bool ParseArguments(IHostEnvironment env, string settings, object destination)
        {
            return ParseArguments(env, settings, destination, Console.Error.WriteLine);
        }

        /// <summary>
        /// Parses a command line. This assumes that the exe name has been stripped off.
        /// Use ArgumentAttributes to control parsing behaviour.
        /// </summary>
        /// <param name="env"> The host environment</param>
        /// <param name="settings">The command line</param>
        /// <param name="destination">The object to receive the options</param>
        /// <param name="destinationType">The type of 'destination'</param>
        /// <param name="reporter"> The destination for parse errors. </param>
        /// <returns>true if no errors were detected</returns>
        public static bool ParseArguments(IHostEnvironment env, string settings, object destination, Type destinationType, ErrorReporter reporter)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(destination, nameof(destination));
            env.CheckValue(destinationType, nameof(destinationType));
            env.Check(destinationType.IsInstanceOfType(destination));
            env.CheckValue(reporter, nameof(reporter));

            string[] strs;
            if (!LexString(settings, out strs))
            {
                reporter("Error: Unbalanced quoting in command line arguments");
                return false;
            }

            var info = GetArgumentInfo(destinationType, null);
            CmdParser parser = new CmdParser(env, reporter);
            return parser.Parse(info, strs, destination);
        }

        private static bool ParseArguments(string settings, object destination, CmdParser parser)
        {
            string[] strs;
            var destinationType = destination.GetType();
            if (!LexString(settings, out strs))
            {
                parser.Report("Error: Unbalanced quoting in command line arguments");
                return false;
            }

            var info = GetArgumentInfo(destinationType, null);
            return parser.Parse(info, strs, destination);
        }

        /// <summary>
        /// Parses a command line. This assumes that the exe name has been stripped off.
        /// Use ArgumentAttributes to control parsing behaviour.
        /// </summary>
        /// <param name="env"> The host environment</param>
        /// <param name="settings">The command line</param>
        /// <param name="destination">The object to receive the options</param>
        /// <param name="reporter"> The destination for parse errors. </param>
        /// <returns>true if no errors were detected</returns>
        public static bool ParseArguments(IHostEnvironment env, string settings, object destination, ErrorReporter reporter)
        {
            Contracts.CheckValue(destination, nameof(destination));
            return ParseArguments(env, settings, destination, destination.GetType(), reporter);
        }

        public static bool ParseArguments(IHostEnvironment env, string settings, object destination, out string helpText)
        {
            return ParseArguments(env, settings, destination, Console.Error.WriteLine, out helpText);
        }

        public static bool ParseArguments(IHostEnvironment env, string settings, object destination, ErrorReporter reporter, out string helpText)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(reporter, nameof(reporter));

            var info = GetArgumentInfo(destination.GetType(), destination);
            CmdParser parser = new CmdParser(env, reporter);

            string[] strs;
            if (!LexString(settings, out strs))
                reporter("Error: Unbalanced quoting in command line arguments");
            else if (parser.Parse(info, strs, destination))
            {
                helpText = null;
                return true;
            }

            helpText = parser.GetUsageString(env, info);
            return false;
        }

        public static string CombineSettings(string[] settings)
        {
            if (settings == null || settings.Length == 0)
                return null;

            StringBuilder sb = new StringBuilder();
            foreach (var s in settings)
            {
                string inner = CmdLexer.UnquoteValue(s);
                inner = inner.Trim();
                if (inner.Length > 0)
                {
                    if (sb.Length > 0)
                        sb.Append(' ');
                    sb.Append(inner);
                }
            }

            return sb.ToString();
        }

        // REVIEW: Add a method for cloning arguments, instead of going to text and back.
        public static string GetSettings(IExceptionContext ectx, object values, object defaults, SettingsFlags flags = SettingsFlags.Default)
        {
            Type t1 = values.GetType();
            Type t2 = defaults.GetType();
            Type t;

            if (t1 == t2)
                t = t1;
            else if (t1.IsAssignableFrom(t2))
                t = t1;
            else if (t2.IsAssignableFrom(t1))
                t = t2;
            else
                return null;

            var info = GetArgumentInfo(t, defaults);
            return GetSettingsCore(ectx, info, values, flags);
        }

        public static IEnumerable<KeyValuePair<string, string>> GetSettingPairs(IHostEnvironment env, object values, object defaults, SettingsFlags flags = SettingsFlags.None)
        {
            Type t1 = values.GetType();
            Type t2 = defaults.GetType();
            Type t;

            if (t1 == t2)
                t = t1;
            else if (t1.IsAssignableFrom(t2))
                t = t1;
            else if (t2.IsAssignableFrom(t1))
                t = t2;
            else
                return null;

            var info = GetArgumentInfo(t, defaults);
            var parser = new CmdParser(env);
            return parser.GetSettingPairsCore(env, info, values, flags);
        }

        public static IEnumerable<KeyValuePair<string, string>> GetSettingPairs(IHostEnvironment env, object values, SettingsFlags flags = SettingsFlags.None)
        {
            var info = GetArgumentInfo(values.GetType(), null);
            var parser = new CmdParser(env);
            return parser.GetSettingPairsCore(env, info, values, flags);
        }

        /// <summary>
        /// Check whether a certain type is numeric.
        /// </summary>
        public static bool IsNumericType(Type type)
        {
            return
                type == typeof(double) ||
                type == typeof(float) ||
                type == typeof(int) ||
                type == typeof(long) ||
                type == typeof(short) ||
                type == typeof(uint) ||
                type == typeof(ulong) ||
                type == typeof(ushort);
        }

        private void Report(string str)
        {
            _reporter?.Invoke(str);
        }

        private void Report(string fmt, params object[] args)
        {
            _reporter?.Invoke(string.Format(fmt, args));
        }

        /// <summary>
        /// Returns a Usage string for command line argument parsing.
        /// Use ArgumentAttributes to control parsing behaviour.
        /// </summary>
        /// <param name="env"> The host environment. </param>
        /// <param name="type"> The type of the arguments to display usage for. </param>
        /// <param name="defaults"> The default values. </param>
        /// <param name="showRsp"> Whether to show the @file item. </param>
        /// <param name="columns"> The number of columns to format the output to. </param>
        /// <returns> Printable string containing a user friendly description of command line arguments. </returns>
        public static string ArgumentsUsage(IHostEnvironment env, Type type, object defaults, bool showRsp = false, int? columns = null)
        {
            var info = GetArgumentInfo(type, defaults);
            var parser = new CmdParser(env);
            return parser.GetUsageString(env, info, showRsp, columns);
        }

#if CORECLR
        /// <summary>
        /// Fix the window width for the Core build to remove the kernel32.dll dependency.
        /// </summary>
        /// <returns></returns>
        public static int GetConsoleWindowWidth()
        {
            return 120;
        }
#else
        private const int StdOutputHandle = -11;

        private struct Coord
        {
            internal Int16 X;
            internal Int16 Y;
        }

        private struct SmallRect
        {
            internal Int16 Left;
            internal Int16 Top;
            internal Int16 Right;
            internal Int16 Bottom;
        }

        private struct ConsoleScreenBufferInfo
        {
            internal Coord DwSize;
            internal Coord DwCursorPosition;
            internal Int16 WAttributes;
            internal SmallRect SrWindow;
            internal Coord DwMaximumWindowSize;
        }

        [DllImport("kernel32.dll", EntryPoint = "GetStdHandle", SetLastError = true, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.StdCall)]
        private static extern int GetStdHandle(int nStdHandle);

        [DllImport("kernel32.dll", EntryPoint = "GetConsoleScreenBufferInfo", SetLastError = true, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.StdCall)]
        private static extern int GetConsoleScreenBufferInfo(int hConsoleOutput, ref ConsoleScreenBufferInfo lpConsoleScreenBufferInfo);

        /// <summary>
        /// Returns the number of columns in the current console window
        /// </summary>
        /// <returns>Returns the number of columns in the current console window</returns>
        public static int GetConsoleWindowWidth()
        {
            int screenWidth;
            ConsoleScreenBufferInfo csbi = new ConsoleScreenBufferInfo();
            // Just to remove the warning messages...
            csbi.DwCursorPosition.X = 0;
            csbi.DwCursorPosition.Y = 0;
            csbi.SrWindow.Bottom = 0;
            csbi.SrWindow.Top = 0;
            csbi.SrWindow.Left = 0;
            csbi.SrWindow.Right = 0;

            int rc;
            rc = GetConsoleScreenBufferInfo(GetStdHandle(StdOutputHandle), ref csbi);
            screenWidth = csbi.DwSize.X;
            return screenWidth;
        }
#endif

        private CmdParser(IHostEnvironment env)
        {
            _host = env.Register("CmdParser");
            _catalog = new Lazy<ModuleCatalog>(() => ModuleCatalog.CreateInstance(_host));
            _reporter = Console.Error.WriteLine;
        }

        private CmdParser(IHostEnvironment env, ErrorReporter reporter)
        {
            Contracts.AssertValueOrNull(reporter);
            _host = env.Register("CmdParser");
            _catalog = new Lazy<ModuleCatalog>(() => ModuleCatalog.CreateInstance(_host));
            _reporter = reporter;
        }

        public static ArgInfo GetArgInfo(Type type, object defaults)
        {
            return ArgInfo.Arg.GetInfo(type, defaults);
        }

        private static string ArgCase(string name)
        {
            if (string.IsNullOrEmpty(name))
                return name;
            if (!char.IsUpper(name[0]))
                return name;

            if (name.Length == 1)
                return name.ToLowerInvariant();
            if (!char.IsUpper(name[1]))
                return name.Substring(0, 1).ToLowerInvariant() + name.Substring(1);

            int firstNonUpper;
            for (firstNonUpper = 0; firstNonUpper < name.Length && char.IsUpper(name[firstNonUpper]); ++firstNonUpper)
                ;
            Contracts.Assert(1 < firstNonUpper && firstNonUpper <= name.Length);
            if (firstNonUpper == name.Length)
                return name.ToLowerInvariant();
            --firstNonUpper;
            return name.Substring(0, firstNonUpper).ToLowerInvariant() + name.Substring(firstNonUpper);
        }

        private static ArgumentInfo GetArgumentInfo(Type type, object defaults)
        {
            Contracts.CheckValue(type, nameof(type));
            Contracts.CheckParam(type.IsClass, nameof(type));

            var args = new List<Argument>();
            var map = new Dictionary<string, Argument>();
            Argument def = null;

            foreach (FieldInfo field in type.GetFields())
            {
                ArgumentAttribute attr = GetAttribute(field);
                if (attr != null)
                {
                    Contracts.Check(!field.IsStatic && !field.IsInitOnly && !field.IsLiteral);
                    bool isDefault = attr is DefaultArgumentAttribute;
                    if (isDefault && def != null)
                        throw Contracts.Except("Duplicate default argument '{0}' vs '{1}'", def.LongName, field.Name);

                    string name = ArgCase(field.Name);
                    string[] nicks;
                    // Semantics of ShortName:
                    //    The string provided represents an array of names separated by commas and spaces, once empty entries are removed.
                    //    'null' or a singleton array with containing only the long field name means "use the default short name",
                    //    and is represented by the null 'nicks' array.
                    //    'String.Empty' or a string containing only spaces and commas means "no short name", and is represented by an empty 'nicks' array.
                    if (attr.ShortName == null)
                        nicks = null;
                    else
                    {
                        nicks = attr.ShortName.Split(new char[] { ',', ' ' }, StringSplitOptions.RemoveEmptyEntries);
                        if (nicks.Length == 0 || nicks.Length == 1 && nicks[0].ToLowerInvariant() == name.ToLowerInvariant())
                            nicks = null;
                    }
                    Contracts.Assert(!isDefault || nicks == null);

                    if (map.ContainsKey(name.ToLowerInvariant()))
                        throw Contracts.Except("Duplicate name '{0}' in argument type '{1}'", name, type.Name);
                    if (nicks != null)
                    {
                        foreach (var nick in nicks)
                        {
                            if (map.ContainsKey(nick.ToLowerInvariant()))
                                throw Contracts.Except("Duplicate name '{0}' in argument type '{1}'", nick, type.Name);
                        }
                    }

                    var arg = new Argument(isDefault ? -1 : args.Count, name, nicks, defaults, attr, field);

                    // Note that we put the default arg in the map to ensure that no other args use the same name.
                    map.Add(name.ToLowerInvariant(), arg);
                    if (nicks != null)
                    {
                        foreach (var nick in nicks)
                            map.Add(nick.ToLowerInvariant(), arg);
                    }

                    if (isDefault)
                        def = arg;
                    else
                        args.Add(arg);
                }
            }

            // If there is a default argument, remove it from the _map.
            if (def != null)
            {
                Contracts.Assert(def.ShortNames == null);
                string name = def.LongName.ToLowerInvariant();
                Contracts.Assert(map.ContainsKey(name) && map[name] == def);
                map.Remove(name);
            }

            return new ArgumentInfo(type, def, args.ToArray(), map);
        }

        private static ArgumentAttribute GetAttribute(FieldInfo field)
        {
            var attrs = field.GetCustomAttributes(typeof(ArgumentAttribute), false).ToArray();
            if (attrs.Length == 1)
            {
                var argumentAttribute = (ArgumentAttribute)attrs[0];
                if (argumentAttribute.Visibility == ArgumentAttribute.VisibilityType.EntryPointsOnly)
                    return null;
                return argumentAttribute;
            }
            Contracts.Assert(attrs.Length == 0);
            return null;
        }

        private void ReportUnrecognizedArgument(string argument)
        {
            Report("Unrecognized command line argument '{0}'", argument);
        }

        /// <summary>
        /// Parses an argument list into an object
        /// </summary>
        /// <param name="info"></param>
        /// <param name="strs"></param>
        /// <param name="destination"></param>
        /// <param name="values"></param>
        /// <returns> true if successful </returns>
        private bool ParseArgumentList(ArgumentInfo info, string[] strs, object destination, ArgValue[] values)
        {
            if (strs == null)
                return true;

            bool hadError = false;
            for (int i = 0; i < strs.Length; i++)
            {
                string str = strs[i];
                Contracts.Assert(!string.IsNullOrEmpty(str));

                Argument arg;
                string option;
                string value;
                string tag;
                switch (str[0])
                {
                    case '@':
                        string[] nested;
                        hadError |= !LexFileArguments(str.Substring(1), out nested);
                        hadError |= !ParseArgumentList(info, nested, destination, values);
                        continue;

                    case '-':
                    case '/':
                        if (!TryGetOptionValue(info, str.Substring(1), out arg, out option, out tag, out value))
                        {
                            ReportUnrecognizedArgument(option);
                            hadError = true;
                            if (value == null && i + 2 < strs.Length && strs[i + 1] == "=")
                                i += 2;
                            continue;
                        }
                        break;

                    default:
                        // No switch. See if it looks like a switch or a default value.
                        if (TryGetOptionValue(info, str, out arg, out option, out tag, out value))
                        {
                            if (info.ArgDef == null)
                                break;

                            // There is a default argument, so if the arg looks like a filename, assume it is
                            // a default argument.
                            if (option.Length > 1 || str.Length <= 1 || str[1] != ':')
                                break;
                        }

                        if (info.ArgDef != null)
                        {
                            str = CmdLexer.UnquoteValue(str);
                            hadError |= !info.ArgDef.SetValue(this, ref values[info.Args.Length], str, "", destination);
                        }
                        else
                        {
                            ReportUnrecognizedArgument(option);
                            hadError = true;
                            if (value == null && i + 2 < strs.Length && strs[i + 1] == "=")
                                i += 2;
                        }
                        continue;
                }
                Contracts.AssertValue(arg);
                Contracts.Assert(arg != info.ArgDef);
                Contracts.Assert(0 <= arg.Index & arg.Index < info.Args.Length);
                if (tag != null && !arg.IsTaggedCollection)
                {
                    hadError = true;
                    Report("Error: Tag not allowed for option '{0}'", arg.LongName);
                    tag = null;
                }
                tag = tag ?? "";

                if (value == null)
                {
                    if (i + 2 < strs.Length && strs[i + 1] == "=")
                        value = strs[i += 2];
                    else if (arg.ItemValueType == typeof(bool))
                    {
                        // The option is a boolean without embedded value, so its value is true.
                        value = "+";
                    }
                    else if (i + 1 < strs.Length)
                        value = strs[++i];
                    else
                    {
                        hadError = true;
                        Report("Error: Need a value for option '{0}'", arg.LongName);
                        continue;
                    }
                }

                if (arg.IsComponentFactory)
                {
                    ModuleCatalog.ComponentInfo component;
                    if (IsCurlyGroup(value) && value.Length == 2)
                        arg.Field.SetValue(destination, null);
                    else if (_catalog.Value.TryFindComponentCaseInsensitive(arg.Field.FieldType, value, out component))
                    {
                        var activator = Activator.CreateInstance(component.ArgumentType);
                        if (!IsCurlyGroup(value) && i + 1 < strs.Length && IsCurlyGroup(strs[i + 1]))
                        {
                            hadError |= !ParseArguments(CmdLexer.UnquoteValue(strs[i + 1]), activator, this);
                            i++;
                        }
                        if (!hadError)
                            arg.Field.SetValue(destination, activator);
                    }
                    else
                    {
                        Report("Error: Failed to find component with name '{0}' for option '{1}'", value, arg.LongName);
                        hadError |= true;
                    }
                    continue;
                }

                if (arg.IsSubComponentItemType)
                {
                    hadError |= !arg.SetValue(this, ref values[arg.Index], value, tag, destination);
                    if (!IsCurlyGroup(value) && i + 1 < strs.Length && IsCurlyGroup(strs[i + 1]))
                        hadError |= !arg.SetValue(this, ref values[arg.Index], strs[++i], "", destination);
                    continue;
                }

                if (arg.IsCustomItemType)
                {
                    hadError |= !arg.SetValue(this, ref values[arg.Index], value, tag, destination);
                    continue;
                }

                if (IsCurlyGroup(value))
                {
                    value = CmdLexer.UnquoteValue(value);
                    hadError |= !arg.SetValue(this, ref values[arg.Index], value, tag, destination);
                    continue;
                }

                // Collections of value types (enum or number) can be specified as a colon separated list.
                if (!arg.IsCollection || value == null || !arg.ItemValueType.IsSubclassOf(typeof(System.ValueType)) ||
                    arg.ItemValueType == typeof(char) || !value.Contains(":"))
                {
                    hadError |= !arg.SetValue(this, ref values[arg.Index], value, tag, destination);
                    continue;
                }

                foreach (string val in value.Split(':'))
                    hadError |= !arg.SetValue(this, ref values[arg.Index], val, "", destination);
            }

            return !hadError;
        }

        // Decompose str into the option, tag (if present) and value (if an embedded value). Get the Argument from the option,
        // or return false if there isn't a corresponding Argument.
        private bool TryGetOptionValue(ArgumentInfo info, string str, out Argument arg, out string option, out string tag, out string value)
        {
            // See if it contains a value.
            int ichLim = str.IndexOfAny(new char[] { ':', '+', '-' });
            if (ichLim > 0 && str[ichLim] == ':')
            {
                option = str.Substring(0, ichLim);
                value = str.Substring(ichLim + 1);
            }
            else if (ichLim > 0 && ichLim == str.Length - 1)
            {
                option = str.Substring(0, ichLim);
                value = str.Substring(ichLim);
            }
            else
            {
                option = str;
                value = null;
            }

            // See if option contains a tag
            tag = null;
            var open = option.IndexOf('[');
            if (open > 0)
            {
                var close = option.IndexOf(']');
                if (close == option.Length - 1)
                {
                    tag = option.Substring(open, close - open + 1);
                    option = option.Substring(0, open);
                }
            }

            return info.Map.TryGetValue(option.ToLowerInvariant(), out arg);
        }

        private static bool IsCurlyGroup(string str)
        {
            return str != null && str.StartsWith("{") && str.EndsWith("}");
        }

        /// <summary>
        /// Parses an argument list.
        /// </summary>
        /// <param name="info"></param>
        /// <param name="strs"> The arguments to parse. </param>
        /// <param name="destination"> The destination of the parsed arguments. </param>
        /// <returns> true if no parse errors were encountered. </returns>
        private bool Parse(ArgumentInfo info, string[] strs, object destination)
        {
            var values = new ArgValue[info.Args.Length + 1];
            bool hadError = !ParseArgumentList(info, strs, destination, values);

            // check for missing required arguments
            for (int i = 0; i < info.Args.Length; i++)
            {
                var arg = info.Args[i];
                hadError |= arg.Finish(this, values[arg.Index], destination);
            }

            if (info.ArgDef != null)
                hadError |= info.ArgDef.Finish(this, values[info.Args.Length], destination);

            return !hadError;
        }

        private static string GetSettingsCore(IExceptionContext ectx, ArgumentInfo info, object values, SettingsFlags flags)
        {
            StringBuilder sb = new StringBuilder();

            if (info.ArgDef != null)
            {
                var val = info.ArgDef.GetValue(values);
                info.ArgDef.AppendSetting(ectx, sb, val, flags);
            }

            foreach (Argument arg in info.Args)
            {
                var val = arg.GetValue(values);
                arg.AppendSetting(ectx, sb, val, flags);
            }

            return sb.ToString();
        }

        /// <summary>
        /// GetSettingsCore handles the top-level case. This handles the nested custom record case.
        /// It deals with custom "unparse" functionality, as well as quoting. It also appends to a StringBuilder
        /// instead of returning a string.
        /// </summary>
        private static void AppendCustomItem(IExceptionContext ectx, ArgumentInfo info, object values, SettingsFlags flags, StringBuilder sb)
        {
            int ich = sb.Length;
            // We always call unparse, even when NoUnparse is specified, since Unparse can "cleanse", which
            // we also want done in the full case.
            if (info.TryUnparse(values, sb))
            {
                // Make sure it doesn't need quoted.
                if ((flags & SettingsFlags.NoUnparse) != 0 || !CmdQuoter.NeedsQuoting(sb, ich))
                    return;
            }
            sb.Length = ich;

            if (info.ArgDef != null)
            {
                var val = info.ArgDef.GetValue(values);
                info.ArgDef.AppendSetting(ectx, sb, val, flags);
            }

            foreach (Argument arg in info.Args)
            {
                var val = arg.GetValue(values);
                arg.AppendSetting(ectx, sb, val, flags);
            }

            string str = sb.ToString(ich, sb.Length - ich);
            sb.Length = ich;
            CmdQuoter.QuoteValue(str, sb, force: true);
        }

        private IEnumerable<KeyValuePair<string, string>> GetSettingPairsCore(IExceptionContext ectx, ArgumentInfo info, object values, SettingsFlags flags)
        {
            StringBuilder buffer = new StringBuilder();
            foreach (Argument arg in info.Args)
            {
                string key = arg.GetKey(flags);
                object value = arg.GetValue(values);
                foreach (string val in arg.GetSettingStrings(ectx, value, buffer))
                    yield return new KeyValuePair<string, string>(key, val);
            }
        }

        private struct ArgumentHelpStrings
        {
            public readonly string Syntax;
            public readonly string Help;

            public ArgumentHelpStrings(string syntax, string help)
            {
                Syntax = syntax;
                Help = help;
            }
        }

        /// <summary>
        /// A user friendly usage string describing the command line argument syntax.
        /// </summary>
        private string GetUsageString(IExceptionContext ectx, ArgumentInfo info, bool showRsp = true, int? columns = null)
        {
            int screenWidth = columns ?? GetConsoleWindowWidth();
            if (screenWidth <= 0)
                screenWidth = 80;

            ArgumentHelpStrings[] strings = GetAllHelpStrings(ectx, info, showRsp);

            int maxParamLen = 0;
            foreach (ArgumentHelpStrings helpString in strings)
            {
                maxParamLen = Math.Max(maxParamLen, helpString.Syntax.Length);
            }

            const int minimumNumberOfCharsForHelpText = 10;
            const int minimumHelpTextColumn = 5;
            const int minimumScreenWidth = minimumHelpTextColumn + minimumNumberOfCharsForHelpText;

            int helpTextColumn;
            int idealMinimumHelpTextColumn = maxParamLen + SpaceBeforeParam;
            screenWidth = Math.Max(screenWidth, minimumScreenWidth);
            if (screenWidth < (idealMinimumHelpTextColumn + minimumNumberOfCharsForHelpText))
                helpTextColumn = minimumHelpTextColumn;
            else
                helpTextColumn = idealMinimumHelpTextColumn;

            const string newLine = "\r\n";
            StringBuilder builder = new StringBuilder();
            foreach (ArgumentHelpStrings helpStrings in strings)
            {
                // add syntax string
                int syntaxLength = helpStrings.Syntax.Length;
                builder.Append(helpStrings.Syntax);

                // start help text on new line if syntax string is too long
                int currentColumn = syntaxLength;
                if (syntaxLength >= helpTextColumn)
                {
                    builder.Append(newLine);
                    currentColumn = 0;
                }

                // add help text broken on spaces
                int charsPerLine = screenWidth - helpTextColumn;
                int index = 0;
                while (index < helpStrings.Help.Length)
                {
                    // tab to start column
                    builder.Append(' ', helpTextColumn - currentColumn);
                    currentColumn = helpTextColumn;

                    // find number of chars to display on this line
                    int endIndex = index + charsPerLine;
                    if (endIndex >= helpStrings.Help.Length)
                    {
                        // rest of text fits on this line
                        endIndex = helpStrings.Help.Length;
                    }
                    else
                    {
                        endIndex = helpStrings.Help.LastIndexOf(' ', endIndex - 1, Math.Min(endIndex - index, charsPerLine));
                        if (endIndex <= index)
                        {
                            // no spaces on this line, append full set of chars
                            endIndex = index + charsPerLine;
                        }
                    }

                    // add chars
                    builder.Append(helpStrings.Help, index, endIndex - index);
                    index = endIndex;

                    // do new line
                    AddNewLine(newLine, builder, ref currentColumn);

                    // don't start a new line with spaces
                    while (index < helpStrings.Help.Length && helpStrings.Help[index] == ' ')
                        index++;
                }

                // add newline if there's no help text
                if (helpStrings.Help.Length == 0)
                {
                    builder.Append(newLine);
                }
            }

            return builder.ToString();
        }

        private static void AddNewLine(string newLine, StringBuilder builder, ref int currentColumn)
        {
            builder.Append(newLine);
            currentColumn = 0;
        }

        private static ArgumentHelpStrings[] GetAllHelpStrings(IExceptionContext ectx, ArgumentInfo info, bool showRsp)
        {
            List<ArgumentHelpStrings> strings = new List<ArgumentHelpStrings>();

            if (info.ArgDef != null)
                strings.Add(GetHelpStrings(ectx, info.ArgDef));

            foreach (Argument arg in info.Args)
            {
                if (!arg.IsHidden)
                    strings.Add(GetHelpStrings(ectx, arg));
            }

            if (showRsp)
                strings.Add(new ArgumentHelpStrings("@<file>", "Read response file for more options"));

            return strings.ToArray();
        }

        private static ArgumentHelpStrings GetHelpStrings(IExceptionContext ectx, Argument arg)
        {
            return new ArgumentHelpStrings(arg.GetSyntaxHelp(), arg.GetFullHelpText(ectx));
        }

        private bool LexFileArguments(string fileName, out string[] arguments)
        {
            string args;
            try
            {
                using (FileStream file = new FileStream(fileName, FileMode.Open, FileAccess.Read))
                {
                    args = (new StreamReader(file)).ReadToEnd();
                }
            }
            catch (Exception e)
            {
                Report("Error: Can't open command line argument file '{0}' : '{1}'", fileName, e.Message);
                arguments = null;
                return false;
            }

            if (!LexString(args, out arguments))
            {
                Report("Error: Unbalanced '\"' in command line argument file '{0}'", fileName);
                return false;
            }

            return true;
        }

        public static string TrimExePath(string args, out string exe)
        {
            if (string.IsNullOrEmpty(args))
            {
                exe = "";
                return args;
            }

            CharCursor curs = new CharCursor(args);

            // The exe part shouldn't consider \ as an escaping character.
            CmdLexer lex = new CmdLexer(curs, false);

            // REVIEW: Should this care about the error flag?
            StringBuilder sb = new StringBuilder();
            lex.GetToken(sb);
            lex.SkipWhiteSpace();

            exe = sb.ToString();
            return curs.GetRest();
        }

        public static bool TryGetFirstToken(string text, out string token, out string rest)
        {
            if (string.IsNullOrWhiteSpace(text))
            {
                token = null;
                rest = null;
                return false;
            }

            StringBuilder bldr = new StringBuilder();
            CharCursor curs = new CharCursor(text);
            CmdLexer lex = new CmdLexer(curs);

            int ich = curs.IchCur;
            lex.GetToken(bldr);
            Contracts.Assert(ich < curs.IchCur || curs.Eof);

            token = bldr.ToString();
            rest = text.Substring(curs.IchCur);

            if (lex.Error || token.Length < 2 || token[0] != '@')
                return !lex.Error;

            // The first token is an rsp file, so need to drill into it.
            string path = token.Substring(1);
            string nested;
            try
            {
                using (FileStream file = new FileStream(path, FileMode.Open, FileAccess.Read))
                {
                    nested = (new StreamReader(file)).ReadToEnd();
                }
            }
            catch (Exception e)
            {
                Console.WriteLine("Error: Can't open command line argument file '{0}' : '{1}'", path, e.Message);
                token = null;
                rest = null;
                return false;
            }

            // Rsp files can contain other rsp files, so we simply recurse....
            if (!TryGetFirstToken(nested, out token, out nested))
                return false;

            // Concatenate the rest of the rsp with the rest of the original string.
            rest = nested + "\r\n" + rest;
            return true;
        }

        public static bool LexString(string text, out string[] arguments)
        {
            if (string.IsNullOrEmpty(text))
            {
                arguments = new string[0];
                return true;
            }

            List<string> args = new List<string>();
            StringBuilder bldr = new StringBuilder();
            CharCursor curs = new CharCursor(text);
            CmdLexer lex = new CmdLexer(curs);

            while (!curs.Eof)
            {
                bldr.Clear();
                int ich = curs.IchCur;
                lex.GetToken(bldr);
                Contracts.Assert(ich < curs.IchCur || curs.Eof);

                if (bldr.Length > 0)
                    args.Add(bldr.ToString());
            }

            arguments = args.ToArray();
            return !lex.Error;
        }

        private static bool IsNullableType(Type type)
        {
            return type.IsConstructedGenericType && type.GetGenericTypeDefinition() == typeof(Nullable<>);
        }

        private static bool IsValidItemType(Type type)
        {
            if (type == null)
                return false;

            if (type == typeof(string))
                return true;

            Type typeBase = type;
            if (IsNullableType(type))
                typeBase = type.GetGenericArguments()[0];

            return
                typeBase == typeof(int) ||
                typeBase == typeof(uint) ||
                typeBase == typeof(short) ||
                typeBase == typeof(ushort) ||
                typeBase == typeof(long) ||
                typeBase == typeof(ulong) ||
                typeBase == typeof(byte) ||
                typeBase == typeof(sbyte) ||
                typeBase == typeof(float) ||
                typeBase == typeof(double) ||
                typeBase == typeof(Guid) ||
                typeBase == typeof(DateTime) ||
                typeBase == typeof(char) ||
                typeBase == typeof(decimal) ||
                typeBase == typeof(bool) ||
                typeBase.IsEnum;
        }

        private sealed class ArgValue
        {
            public readonly string FirstValue;
            public List<KeyValuePair<string, object>> Values; // Key=tag.

            public ArgValue(string first)
            {
                Contracts.AssertValueOrNull(first);
                FirstValue = first;
            }
        }

        public sealed class ArgInfo
        {
            public readonly Arg[] Args;
            private readonly ArgumentInfo _info;

            private ArgInfo(ArgumentInfo info, Arg[] args)
            {
                Contracts.AssertValue(info);
                Contracts.AssertValue(args);
                Args = args;
                _info = info;
            }

            public bool TryGetArg(string name, out Arg arg)
            {
                Argument argument;
                if (string.IsNullOrWhiteSpace(name) || !_info.Map.TryGetValue(name.ToLowerInvariant(), out argument))
                {
                    arg = null;
                    return false;
                }
                // Given that we find it in the map, there should be precisely one match.
                Contracts.Assert(Args.Where(a => a.Index == argument.Index).Count() == 1);
                arg = Args.FirstOrDefault(a => a.Index == argument.Index);
                Contracts.AssertValue(arg);
                return true;
            }

            public sealed class Arg
            {
                /// <summary>
                /// This class exposes those parts of this wrapped <see cref="Argument"/> appropriate
                /// for public consumption.
                /// </summary>
                private readonly Argument _arg;

                public int Index { get { return _arg.Index; } }
                public FieldInfo Field { get { return _arg.Field; } }
                public string LongName { get { return _arg.LongName; } }
                public string[] ShortNames { get { return _arg.ShortNames; } }
                public string HelpText { get { return _arg.HelpText; } }
                public Type ItemType { get { return _arg.ItemType; } }
                public double SortOrder { get { return _arg.SortOrder; } }
                public string NullName { get { return _arg.NullName; } }

                // If tagged collection, this is the 'value' of the KeyValuePair. Otherwise, equal to ItemType.
                public Type ItemValueType { get { return _arg.ItemValueType; } }
                public ArgumentType Kind { get { return _arg.Kind; } }
                public bool IsDefault { get { return _arg.IsDefault; } }
                public bool IsHidden { get { return _arg.IsHidden; } }
                public bool IsCollection { get { return _arg.IsCollection; } }
                public bool IsSubComponentItemType { get { return _arg.IsSubComponentItemType; } }
                public bool IsTaggedCollection { get { return _arg.IsTaggedCollection; } }
                // Used for help and composing settings strings.
                public object DefaultValue { get { return _arg.DefaultValue; } }

                public bool IsRequired {
                    get { return ArgumentType.Required == (Kind & ArgumentType.Required); }
                }

                public string ItemTypeString { get { return TypeToString(ItemValueType); } }

                private Arg(Argument arg)
                {
                    Contracts.AssertValue(arg);
                    _arg = arg;
                }

                internal static ArgInfo GetInfo(Type type, object defaults = null)
                {
                    Contracts.CheckValue(type, nameof(type));
                    Contracts.CheckValueOrNull(defaults);
                    if (defaults == null)
                        defaults = Activator.CreateInstance(type);
                    ArgumentInfo argumentInfo = GetArgumentInfo(type, defaults);
                    Arg[] args = Utils.BuildArray(argumentInfo.Args.Length, i => new Arg(argumentInfo.Args[i]));
                    Array.Sort(args, 0, args.Length, Comparer<Arg>.Create((x, y) => x.SortOrder.CompareTo(y.SortOrder)));
                    return new ArgInfo(argumentInfo, args);
                }

                private static string TypeToString(Type type)
                {
                    Contracts.AssertValue(type);
                    bool isGeneric = type.IsGenericType;
                    if (type.IsGenericEx(typeof(Nullable<>)))
                    {
                        Contracts.Assert(isGeneric);
                        var genArgs = type.GetGenericTypeArgumentsEx();
                        Contracts.Assert(Utils.Size(genArgs) == 1);
                        return TypeToString(genArgs[0]) + "?";
                    }

                    if (type == typeof(int))
                        return "int";
                    else if (type == typeof(float))
                        return "float";
                    else if (type == typeof(double))
                        return "double";
                    else if (type == typeof(string))
                        return "string";
                    else if (type == typeof(uint))
                        return "uint";
                    else if (type == typeof(byte))
                        return "byte";
                    else if (type == typeof(sbyte))
                        return "sbyte";
                    else if (type == typeof(short))
                        return "short";
                    else if (type == typeof(ushort))
                        return "ushort";
                    else if (type == typeof(long))
                        return "long";
                    else if (type == typeof(ulong))
                        return "ulong";
                    else if (type == typeof(bool))
                        return "bool";
                    else if (type == typeof(decimal))
                        return "decimal";
                    else if (type == typeof(char))
                        return "char";
                    else if (type == typeof(System.Guid))
                        return "guid";
                    else if (type == typeof(System.DateTime))
                        return "datetime";
                    else if (type.IsEnum)
                    {
                        var bldr = new StringBuilder();
                        bldr.Append("[");
                        string sep = "";
                        foreach (FieldInfo field in type.GetFields())
                        {
                            if (!field.IsStatic || field.FieldType != type)
                                continue;
                            if (field.GetCustomAttribute<HideEnumValueAttribute>() != null)
                                continue;

                            bldr.Append(sep).Append(field.Name);
                            sep = " | ";
                        }
                        bldr.Append(']');
                        return bldr.ToString();
                    }
                    else if (type.IsGenericEx(typeof(SubComponent<,>)))
                    {
                        var genArgs = type.GetGenericTypeArgumentsEx();
                        Contracts.Assert(Utils.Size(genArgs) == 2);
                        return $"{ComponentCatalog.SignatureToString(genArgs[1])} ⇒ {genArgs[0].Name}";
                    }
                    else
                        return type.Name;
                }
            }
        }

        private sealed class ArgumentInfo
        {
            public readonly Argument ArgDef;
            public readonly Argument[] Args;
            public readonly Dictionary<string, Argument> Map;
            public readonly MethodInfo ParseCustom;
            public readonly MethodInfo TryUnparseCustom;

            public ArgumentInfo(Type type, Argument argDef, Argument[] args, Dictionary<string, Argument> map)
            {
                Contracts.Assert(argDef == null || argDef.Index == -1);
                Contracts.AssertValue(args);
                Contracts.AssertValue(map);
                Contracts.Assert(map.Count >= args.Length);
                Contracts.Assert(args.Select((arg, index) => arg.Index == index).All(b => b));

                ArgDef = argDef;
                Args = args;
                Map = map;

                // See if there is a custom parse method.
                // REVIEW: Is there a better way to handle parsing and unparsing of custom forms?
                // For unparse, we could use an interface or base class. That doesn't work for parsing though,
                // since we don't have an instance at parse time (Parse is a static method). Perhaps we could
                // first construct an instance, then call an instance method to try the parsing. Unfortunately,
                // if parsing fails, we'd have to toss that instance and create another into which to do the
                // field-by-field parsing.
                ParseCustom = GetParseMethod(type);
                TryUnparseCustom = GetUnparseMethod(type);
            }

            private static MethodInfo GetParseMethod(Type type)
            {
                Contracts.AssertValue(type);
                var meth = type.GetMethod("Parse", new[] { typeof(string) });
                if (meth != null && meth.IsStatic && meth.IsPublic && meth.ReturnType == type)
                    return meth;
                return null;
            }

            private static MethodInfo GetUnparseMethod(Type type)
            {
                Contracts.AssertValue(type);
                var meth = type.GetMethod("TryUnparse", new[] { typeof(StringBuilder) });
                if (meth != null && !meth.IsStatic && meth.IsPublic && meth.ReturnType == typeof(bool))
                    return meth;
                return null;
            }

            public bool TryUnparse(object obj, StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (TryUnparseCustom == null)
                    return false;
                return (bool)TryUnparseCustom.Invoke(obj, new object[] { sb });
            }
        }

        private sealed class Argument
        {
            public readonly int Index;
            public readonly FieldInfo Field;
            public readonly string LongName;
            public readonly string[] ShortNames;
            public readonly string HelpText;
            public readonly Type ItemType;
            public readonly double SortOrder;
            public readonly string NullName;

            // If tagged collection, this is the 'value' of the KeyValuePair. Otherwise, equal to ItemType.
            public readonly Type ItemValueType;

            public readonly ArgumentType Kind;
            public readonly bool IsDefault;
            public readonly bool IsHidden;
            public readonly bool IsCollection;
            public readonly bool IsSubComponentItemType;
            public readonly bool IsTaggedCollection;
            public readonly bool IsComponentFactory;

            // Used for help and composing settings strings.
            public readonly object DefaultValue;

            // For custom types.
            private readonly ArgumentInfo _infoCustom;
            private readonly ConstructorInfo _ctorCustom;

            public Argument(int index, string name, string[] nicks, object defaults, ArgumentAttribute attr, FieldInfo field)
            {
                Contracts.Assert(index >= -1);
                Contracts.Assert(!string.IsNullOrWhiteSpace(name));
                Contracts.Check(nicks == null || nicks.All(nick => !string.IsNullOrWhiteSpace(nick)));
                Contracts.AssertValueOrNull(defaults);
                Contracts.AssertValue(attr);
                Contracts.AssertValue(field);

                Index = index;
                Field = field;
                LongName = name;
                ShortNames = nicks;
                HelpText = attr.HelpText;
                SortOrder = attr.SortOrder;
                NullName = attr.NullName;
                if (string.IsNullOrWhiteSpace(HelpText))
                    HelpText = null;

                Kind = attr.Type;
                IsDefault = attr is DefaultArgumentAttribute;
                Contracts.Assert(!IsDefault || Utils.Size(ShortNames) == 0);
                IsHidden = attr.Hide;

                if (field.FieldType.IsArray)
                {
                    IsCollection = true;
                    ItemType = field.FieldType.GetElementType();

                    IsTaggedCollection = IsKeyValuePair(ItemType);
                    ItemValueType = IsTaggedCollection ? ItemType.GenericTypeArguments[1] : ItemType;
                }
                else
                {
                    IsCollection = IsTaggedCollection = false;
                    ItemValueType = ItemType = field.FieldType;
                }

                if (defaults != null && !IsRequired)
                    DefaultValue = field.GetValue(defaults);

                if (typeof(SubComponent).IsAssignableFrom(ItemValueType))
                    IsSubComponentItemType = true;
                else if (typeof(IComponentFactory).IsAssignableFrom(ItemValueType))
                    IsComponentFactory = true;
                else if (!IsValidItemType(ItemValueType))
                {
                    object def;
                    try
                    {
                        _ctorCustom = ItemValueType.GetConstructor(Type.EmptyTypes);
                        def = _ctorCustom != null ? _ctorCustom.Invoke(null) : null;
                    }
                    catch
                    {
                        def = null;
                    }

                    if (def != null)
                    {
                        _infoCustom = GetArgumentInfo(ItemValueType, def);
                        if (_infoCustom.ArgDef == null && _infoCustom.Args.Length == 0)
                            _infoCustom = null;
                    }

                    if (_infoCustom == null)
                        throw Contracts.Except("Invalid argument type: '{0}'", ItemValueType.Name);
                }

                Contracts.Check(!IsCollection || AllowMultiple, "Collection arguments must allow multiple");
                Contracts.Check(!IsSingleSubComponent || AllowMultiple, "SubComponent arguments must allow multiple");
                Contracts.Check(!Unique || IsCollection, "Unique only applicable to collection arguments");
            }

            private bool IsKeyValuePair(Type type)
            {
                Contracts.AssertValue(type);
                if (!type.IsGenericEx(typeof(KeyValuePair<,>)))
                    return false;

                Contracts.AssertValue(type.GenericTypeArguments);
                if (type.GenericTypeArguments.Length != 2)
                    return false;

                return type.GenericTypeArguments[0] == typeof(string);
            }

            public bool Finish(CmdParser owner, ArgValue val, object destination)
            {
                if (val == null)
                {
                    // Make sure all collections have a non-null.
                    // REVIEW: Should we really do this? Or should all code be able to handle null arrays?
                    // Should we also set null strings to ""? SubComponent?
                    if (IsCollection && Field.GetValue(destination) == null)
                        Field.SetValue(destination, Array.CreateInstance(ItemType, 0));
                    return ReportMissingRequiredArgument(owner, val);
                }

                var values = val.Values;
                bool error = false;
                if (IsSingleSubComponent)
                {
                    bool haveKind = false;
                    var com = SubComponent.Create(ItemType);
                    for (int i = 0; i < Utils.Size(values);)
                    {
                        string str = (string)values[i].Value;
                        if (str.StartsWith("{"))
                        {
                            i++;
                            continue;
                        }
                        if (haveKind)
                        {
                            owner.Report("Duplicate component kind for argument {0}", LongName);
                            error = true;
                        }
                        com.Kind = str;
                        haveKind = true;
                        values.RemoveAt(i);
                    }

                    if (Utils.Size(values) > 0)
                        com.Settings = values.Select(x => (string)x.Value).ToArray();

                    Field.SetValue(destination, com);
                }
                else if (IsMultiSubComponent)
                {
                    // REVIEW: the kind should not be separated from settings: everything related
                    // to one item should go into one value, not multiple values
                    if (IsTaggedCollection)
                    {
                        // Tagged collection of SubComponents
                        var comList = new List<KeyValuePair<string, SubComponent>>();

                        for (int i = 0; i < Utils.Size(values);)
                        {
                            var com = SubComponent.Create(ItemValueType);
                            string tag = values[i].Key;
                            com.Kind = (string)values[i++].Value;
                            if (i < values.Count && IsCurlyGroup((string)values[i].Value) && string.IsNullOrEmpty(values[i].Key))
                                com.Settings = new string[] { (string)values[i++].Value };
                            comList.Add(new KeyValuePair<string, SubComponent>(tag, com));
                        }

                        var arr = Array.CreateInstance(ItemType, comList.Count);
                        for (int i = 0; i < arr.Length; i++)
                        {
                            var kvp = Activator.CreateInstance(ItemType, comList[i].Key, comList[i].Value);
                            arr.SetValue(kvp, i);
                        }

                        Field.SetValue(destination, arr);
                    }
                    else
                    {
                        // Collection of SubComponents
                        var comList = new List<SubComponent>();
                        for (int i = 0; i < Utils.Size(values);)
                        {
                            var com = SubComponent.Create(ItemValueType);
                            com.Kind = (string)values[i++].Value;
                            if (i < values.Count && IsCurlyGroup((string)values[i].Value))
                                com.Settings = new string[] { (string)values[i++].Value };
                            comList.Add(com);
                        }

                        var arr = Array.CreateInstance(ItemValueType, comList.Count);
                        for (int i = 0; i < arr.Length; i++)
                            arr.SetValue(comList[i], i);
                        Field.SetValue(destination, arr);
                    }
                }
                else if (IsTaggedCollection)
                {
                    var res = Array.CreateInstance(ItemType, Utils.Size(values));
                    for (int i = 0; i < res.Length; i++)
                    {
                        var kvp = Activator.CreateInstance(ItemType, values[i].Key, values[i].Value);
                        res.SetValue(kvp, i);
                    }
                    Field.SetValue(destination, res);
                }
                else if (IsCollection)
                {
                    var res = Array.CreateInstance(ItemType, Utils.Size(values));
                    for (int i = 0; i < res.Length; i++)
                        res.SetValue(values[i].Value, i);
                    Field.SetValue(destination, res);
                }

                return error;
            }

            private bool ReportMissingRequiredArgument(CmdParser owner, ArgValue val)
            {
                if (!IsRequired || val != null)
                    return false;

                if (IsDefault)
                    owner.Report("Missing required argument '<{0}>'.", LongName);
                else
                    owner.Report("Missing required argument '{0}'.", LongName);
                return true;
            }

            private void ReportDuplicateArgumentValue(CmdParser owner, string value)
            {
                owner.Report("Duplicate '{0}' argument '{1}'", LongName, value);
            }

            public bool SetValue(CmdParser owner, ref ArgValue val, string value, string tag, object destination)
            {
                if (val == null)
                    val = new ArgValue(value);
                else if (!AllowMultiple)
                {
                    owner.Report("Duplicate '{0}' argument: '{1}' then '{2}'", LongName, val.FirstValue, value);
                    return false;
                }

                Contracts.Assert(string.IsNullOrEmpty(tag) || tag.StartsWith("[") && tag.EndsWith("]"));
                bool hasTag = tag != null && tag.Length > 2;

                tag = hasTag ? tag.Substring(1, tag.Length - 2) : "";
                if (hasTag && !IsTaggedCollection)
                {
                    owner.Report("Tags aren't allowed for '{0}' argument", LongName);
                    return false;
                }

                object newValue;
                if (!ParseValue(owner, value, out newValue))
                    return false;

                if (IsCollection)
                {
                    if (val.Values == null)
                        val.Values = new List<KeyValuePair<string, object>>();
                    else if (Unique && val.Values.Any(x => x.Value.Equals(newValue)))
                    {
                        ReportDuplicateArgumentValue(owner, value);
                        return false;
                    }
                    val.Values.Add(new KeyValuePair<string, object>(tag, newValue));
                }
                else if (IsSingleSubComponent)
                {
                    Contracts.Assert(newValue is string || newValue == null);
                    Contracts.Assert((string)newValue != "");
                    if (newValue != null && (string)newValue != "{}")
                        Utils.Add(ref val.Values, new KeyValuePair<string, object>("", newValue));
                }
                else
                    Field.SetValue(destination, newValue);

                return true;
            }

            public object GetValue(object source)
            {
                return Field.GetValue(source);
            }

            public string GetKey(SettingsFlags flags)
            {
                if ((flags & SettingsFlags.ShortNames) != 0)
                    return Utils.Size(ShortNames) != 0 ? ShortNames[0] : LongName;
                return LongName;
            }

            private void ReportBadArgumentValue(CmdParser owner, string value)
            {
                owner.Report("'{0}' is not a valid value for the '{1}' command line option", value, LongName);
            }

            private bool ParseValue(CmdParser owner, string data, out object value)
            {
                Type type = ItemValueType;

                // Treat empty string the same as null. Null is only valid for nullable types, strings,
                // and sub components.
                if (string.IsNullOrEmpty(data))
                {
                    value = null;
                    if (IsNullableType(type))
                        return true;
                    if (type == typeof(string))
                        return true;
                    if (IsSubComponentItemType)
                        return true;

                    ReportBadArgumentValue(owner, data);
                    return false;
                }

                if (IsSubComponentItemType)
                {
                    value = data;
                    return true;
                }

                if (IsCustomItemType)
                {
                    object res;
                    if (!IsCurlyGroup(data))
                    {
                        if (_infoCustom.ParseCustom != null &&
                            (res = _infoCustom.ParseCustom.Invoke(null, new string[] { data })) != null)
                        {
                            value = res;
                            return true;
                        }

                        owner.Report("Expected arguments in curly braces for '{0}'", LongName);
                        value = null;
                        return false;
                    }

                    // Need to unquote.
                    data = CmdLexer.UnquoteValue(data);

                    res = _ctorCustom.Invoke(null);
                    string[] strs;
                    if (!LexString(data, out strs))
                    {
                        value = null;
                        return false;
                    }
                    if (!owner.Parse(_infoCustom, strs, res))
                    {
                        value = null;
                        return false;
                    }
                    value = res;
                    return true;
                }

                if (IsNullableType(type))
                    type = type.GetGenericArguments()[0];

                try
                {
                    if (type == typeof(string))
                    {
                        value = data;
                        return true;
                    }
                    else if (type == typeof(bool))
                    {
                        if (data == "+")
                        {
                            value = true;
                            return true;
                        }
                        else if (data == "-")
                        {
                            value = false;
                            return true;
                        }
                    }
                    else if (type == typeof(int))
                    {
                        value = int.Parse(data);
                        return true;
                    }
                    else if (type == typeof(uint))
                    {
                        value = uint.Parse(data);
                        return true;
                    }
                    else if (type == typeof(short))
                    {
                        value = short.Parse(data);
                        return true;
                    }
                    else if (type == typeof(ushort))
                    {
                        value = ushort.Parse(data);
                        return true;
                    }
                    else if (type == typeof(byte))
                    {
                        value = byte.Parse(data);
                        return true;
                    }
                    else if (type == typeof(sbyte))
                    {
                        value = sbyte.Parse(data);
                        return true;
                    }
                    else if (type == typeof(long))
                    {
                        value = long.Parse(data);
                        return true;
                    }
                    else if (type == typeof(ulong))
                    {
                        value = ulong.Parse(data);
                        return true;
                    }
                    else if (type == typeof(float))
                    {
                        value = float.Parse(data, CultureInfo.InvariantCulture);
                        return true;
                    }
                    else if (type == typeof(double))
                    {
                        value = double.Parse(data, CultureInfo.InvariantCulture);
                        return true;
                    }
                    else if (type == typeof(decimal))
                    {
                        value = decimal.Parse(data);
                        return true;
                    }
                    else if (type == typeof(char))
                    {
                        value = char.Parse(data);
                        return true;
                    }
                    else if (type == typeof(System.Guid))
                    {
                        value = new System.Guid(data);
                        return true;
                    }
                    else if (type == typeof(System.DateTime))
                    {
                        value = System.DateTime.Parse(data);
                        return true;
                    }
                    else
                    {
                        Contracts.Assert(type.IsEnum);
                        value = Enum.Parse(type, data, true);
                        return true;
                    }
                }
                catch
                {
                    // catch parse errors
                }

                ReportBadArgumentValue(owner, data);
                value = null;
                return false;
            }

            private void AppendHelpValue(IExceptionContext ectx, StringBuilder builder, object value)
            {
                if (value == null)
                    builder.Append("{}");
                else if (value is bool)
                    builder.Append((bool)value ? "+" : "-");
                else if (value is Array)
                {
                    string pre = "";
                    foreach (object o in (System.Array)value)
                    {
                        builder.Append(pre);
                        AppendHelpValue(ectx, builder, o);
                        pre = ", ";
                    }
                }
                else if (value is IComponentFactory)
                {
                    string name;
                    var catalog = ModuleCatalog.CreateInstance(ectx);
                    var type = value.GetType();
                    bool success = catalog.TryGetComponentShortName(type, out name);
                    Contracts.Assert(success);

                    var settings = GetSettings(ectx, value, Activator.CreateInstance(type));
                    builder.Append(name);
                    if (!string.IsNullOrWhiteSpace(settings))
                    {
                        StringBuilder sb = new StringBuilder();
                        CmdQuoter.QuoteValue(settings, sb, true);
                        builder.Append(sb);
                    }
                }
                else
                {
                    // REVIEW: This isn't necessarily correct for string - may need to quote!
                    builder.Append(value.ToString());
                }
            }

            // If value differs from the default, appends the setting to sb.
            public void AppendSetting(IExceptionContext ectx, StringBuilder sb, object value, SettingsFlags flags)
            {
                object def = DefaultValue;
                if (!IsCollection)
                {
                    if (value == null)
                    {
                        if (def != null || IsRequired)
                            AppendSettingCore(ectx, sb, "", flags);
                    }
                    else if (def == null || !value.Equals(def))
                    {
                        var buffer = new StringBuilder();
                        if (!(value is IComponentFactory) || (GetString(ectx, value, buffer) != GetString(ectx, def, buffer)))
                            AppendSettingCore(ectx, sb, value, flags);
                    }
                    return;
                }
                Contracts.Assert(value == null || value is Array);

                IList vals = (Array)value;
                if (vals == null || vals.Count == 0)
                {
                    // REVIEW: Is there any way to represent an empty array?
                    return;
                }

                // See if vals matches defs.
                IList defs = (Array)def;
                if (defs != null && vals.Count == defs.Count)
                {
                    for (int i = 0; ; i++)
                    {
                        if (i >= vals.Count)
                            return;
                        if (!vals[i].Equals(defs[i]))
                            break;
                    }
                }

                foreach (object x in vals)
                    AppendSettingCore(ectx, sb, x, flags);
            }

            private void AppendSettingCore(IExceptionContext ectx, StringBuilder sb, object value, SettingsFlags flags)
            {
                if (sb.Length > 0)
                    sb.Append(" ");

                if (!IsDefault)
                {
                    if ((flags & SettingsFlags.NoSlashes) == 0)
                        sb.Append('/');
                    sb.Append(GetKey(flags));
                    string tag;
                    ExtractTag(value, out tag, out value);
                    sb.Append(tag);
                    sb.Append('=');
                }

                if (value is string)
                    CmdQuoter.QuoteValue(value.ToString(), sb);
                else if (value is bool)
                    sb.Append((bool)value ? "+" : "-");
                else if (IsCustomItemType)
                    AppendCustomItem(ectx, _infoCustom, value, flags, sb);
                else if (IsComponentFactory)
                {
                    var buffer = new StringBuilder();
                    sb.Append(GetString(ectx, value, buffer));
                }
                else
                    sb.Append(value.ToString());
            }

            private void ExtractTag(object value, out string tag, out object newValue)
            {
                if (!IsTaggedCollection)
                {
                    newValue = value;
                    tag = "";
                    return;
                }
                var type = value.GetType();

                Contracts.Assert(IsKeyValuePair(type));
                tag = (string)type.GetProperty("Key").GetValue(value);
                if (!string.IsNullOrEmpty(tag))
                    tag = string.Format("[{0}]", tag);
                newValue = type.GetProperty("Value").GetValue(value);
            }

            // If value differs from the default, return the string representation of 'value',
            // or an IEnumerable of string representations if 'value' is an array.
            public IEnumerable<string> GetSettingStrings(IExceptionContext ectx, object value, StringBuilder buffer)
            {
                object def = DefaultValue;

                if (!IsCollection)
                {
                    if (value == null)
                    {
                        if (def != null || IsRequired)
                            yield return GetString(ectx, value, buffer);
                    }
                    else if (def == null || !value.Equals(def))
                    {
                        if (!(value is IComponentFactory) || (GetString(ectx, value, buffer) != GetString(ectx, def, buffer)))
                            yield return GetString(ectx, value, buffer);
                    }
                    yield break;
                }

                Contracts.Assert(value == null || value is Array);

                IList vals = (Array)value;
                if (vals == null || vals.Count == 0)
                {
                    // REVIEW: Is there any way to represent an empty array?
                    yield break;
                }

                // See if vals matches defs.
                IList defs = (Array)def;
                if (defs != null && vals.Count == defs.Count)
                {
                    for (int i = 0; ; i++)
                    {
                        if (i >= vals.Count)
                            yield break;
                        if (!vals[i].Equals(defs[i]))
                            break;
                    }
                }

                foreach (object x in vals)
                    yield return GetString(ectx, x, buffer);
            }

            private string GetString(IExceptionContext ectx, object value, StringBuilder buffer)
            {
                if (value == null)
                    return "{}";

                if (value is string)
                {
                    buffer.Clear();
                    CmdQuoter.QuoteValue(value.ToString(), buffer);
                    return buffer.ToString();
                }

                if (value is bool)
                    return (bool)value ? "+" : "-";

                if (value is IComponentFactory)
                {
                    string name;
                    var catalog = ModuleCatalog.CreateInstance(ectx);
                    var type = value.GetType();
                    bool success = catalog.TryGetComponentShortName(type, out name);
                    Contracts.Assert(success);

                    var settings = GetSettings(ectx, value, Activator.CreateInstance(type));
                    buffer.Clear();
                    buffer.Append(name);
                    if (!string.IsNullOrWhiteSpace(settings))
                    {
                        StringBuilder sb = new StringBuilder();
                        CmdQuoter.QuoteValue(settings, sb, true);
                        buffer.Append(sb);
                    }
                    return buffer.ToString();
                }

                return value.ToString();
            }

            public string GetFullHelpText(IExceptionContext ectx)
            {
                if (IsHidden)
                    return null;

                StringBuilder builder = new StringBuilder();
                if (HelpText != null)
                {
                    builder.Append(HelpText);

                    // REVIEW: Add this code.
                    switch (HelpText[HelpText.Length - 1])
                    {
                        case '.':
                        case '?':
                        case '!':
                            break;
                        default:
                            //builder.Append(".");
                            break;
                    }
                }
                if (DefaultValue != null && !"".Equals(DefaultValue))
                {
                    if (builder.Length > 0)
                        builder.Append(" ");
                    builder.Append("Default value:'");
                    AppendHelpValue(ectx, builder, DefaultValue);
                    builder.Append('\'');
                }
                if (Utils.Size(ShortNames) != 0)
                {
                    if (builder.Length > 0)
                        builder.Append(" ");
                    // REVIEW: This hides all short names except the first, which may cause user confusion.
                    builder.Append("(short form ").Append(ShortNames[0]).Append(")");
                }
                return builder.ToString();
            }

            public string GetSyntaxHelp()
            {
                if (IsHidden)
                    return null;

                StringBuilder bldr = new StringBuilder();
                if (IsDefault)
                    bldr.Append("<").Append(LongName).Append(">");
                else
                {
                    bldr.Append(LongName);
                    Type type = ItemValueType;

                    if (IsNullableType(type))
                        type = type.GetGenericArguments()[0];

                    if (IsTaggedCollection)
                        bldr.Append("[<tag>]");

                    if (type == typeof(int))
                        bldr.Append("=<int>");
                    else if (type == typeof(uint))
                        bldr.Append("=<uint>");
                    else if (type == typeof(byte))
                        bldr.Append("=<byte>");
                    else if (type == typeof(sbyte))
                        bldr.Append("=<sbyte>");
                    else if (type == typeof(short))
                        bldr.Append("=<short>");
                    else if (type == typeof(ushort))
                        bldr.Append("=<ushort>");
                    else if (type == typeof(long))
                        bldr.Append("=<long>");
                    else if (type == typeof(ulong))
                        bldr.Append("=<ulong>");
                    else if (type == typeof(bool))
                        bldr.Append("=[+|-]");
                    else if (type == typeof(string))
                        bldr.Append("=<string>");
                    else if (type == typeof(float))
                        bldr.Append("=<float>");
                    else if (type == typeof(double))
                        bldr.Append("=<double>");
                    else if (type == typeof(decimal))
                        bldr.Append("=<decimal>");
                    else if (type == typeof(char))
                        bldr.Append("=<char>");
                    else if (type == typeof(System.Guid))
                        bldr.Append("=<guid>");
                    else if (type == typeof(System.DateTime))
                        bldr.Append("=<datetime>");
                    else if (IsSubComponentItemType)
                        bldr.Append("=<name>{<options>}");
                    else if (IsComponentFactory)
                        bldr.Append("=<name>{<options>}");
                    else if (IsCustomItemType)
                        bldr.Append("={<options>}");
                    else if (type.IsEnum)
                    {
                        bldr.Append("=[");
                        string sep = "";
                        foreach (FieldInfo field in type.GetFields())
                        {
                            if (!field.IsStatic || field.FieldType != type)
                                continue;
                            if (field.GetCustomAttribute<HideEnumValueAttribute>() != null)
                                continue;

                            bldr.Append(sep).Append(field.Name);
                            sep = "|";
                        }
                        bldr.Append(']');
                    }
                    else
                        Contracts.Assert(false);
                }

                return bldr.ToString();
            }

            public bool IsRequired {
                get { return 0 != (Kind & ArgumentType.Required); }
            }

            public bool AllowMultiple {
                get { return 0 != (Kind & ArgumentType.Multiple); }
            }

            public bool Unique {
                get { return 0 != (Kind & ArgumentType.Unique); }
            }

            public bool IsSingleSubComponent {
                get { return IsSubComponentItemType && !Field.FieldType.IsArray; }
            }

            public bool IsMultiSubComponent {
                get { return IsSubComponentItemType && Field.FieldType.IsArray; }
            }

            public bool IsCustomItemType {
                get { return _infoCustom != null; }
            }
        }
    }
}