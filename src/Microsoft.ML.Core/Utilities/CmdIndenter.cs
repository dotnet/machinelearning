// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.CodeDom.Compiler;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Runtime.CommandLine;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    public static class CmdIndenter
    {
        /// <summary>
        /// Get indented version of command line or same string if we unable to produce it.
        /// </summary>
        /// <param name="commandLine">command line</param>
        /// <returns>indented version of command line(if possible)</returns>
        public static string GetIndentedCommandLine(string commandLine)
        {
            using (var sw = new System.IO.StringWriter())
            {
                var itw = new IndentedTextWriter(sw, "  ");
                if (TryProduceIndentString(commandLine, itw))
                    return sw.ToString().Trim();
                return commandLine;
            }
        }

        /// <summary>
        /// Try produce indented string for command line.
        /// </summary>
        /// <param name="text">command line</param>
        /// <param name="itw">indenting text writer</param>
        /// <returns>true if we was able to produce indented string without any problem</returns>
        private static bool TryProduceIndentString(string text, IndentedTextWriter itw)
        {
            string[] tokens;
            if (!CmdParser.LexString(text, out tokens))
                return false;
            for (var i = 0; i < tokens.Length; )
            {
                //We in last token, or next token don't equal to '='.
                if (i + 1 == tokens.Length || tokens[i + 1] != "=")
                {
                    itw.WriteLine(tokens[i++]);
                }
                else
                {
                    itw.Write(tokens[i++]);
                    itw.Write(tokens[i++]);
                    // We have something like "name =" which is invalid.
                    if (i >= tokens.Length)
                        return false;
                    //We have something like "name = value {options}".
                    if (i + 1 < tokens.Length && tokens[i + 1].StartsWith("{") && tokens[i + 1].EndsWith("}"))
                    {
                        itw.Write(tokens[i++]);
                        itw.WriteLine("{");
                        using (itw.Nest())
                        {
                            var str = CmdLexer.UnquoteValue(tokens[i++]);
                            // REVIEW: Probably we shouldn't give up if we have problem within one of the token
                            //and we need return partially indented string.
                            bool success = TryProduceIndentString(str, itw);
                            if (!success)
                                return false;
                        }
                        itw.WriteLine("}");

                    }
                    //We have something like "name = value".
                    else
                    {
                        itw.WriteLine(tokens[i++]);
                    }
                }
            }
            return true;
        }
    }
}
