using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Tensorflow.IO
{
    public class GFile
    {
        /// <summary>
        /// Recursive directory tree generator for directories.
        /// </summary>
        /// <param name="top">a Directory name</param>
        /// <param name="in_order">Traverse in order if True, post order if False.</param>
        public IEnumerable<(string, string[], string[])> Walk(string top, bool in_order = true)
        {
            return walk_v2(top, in_order);
        }

        private IEnumerable<(string, string[], string[])> walk_v2(string top, bool topdown)
        {
            var subdirs = Directory.GetDirectories(top);
            var files = Directory.GetFiles(top);

            var here = (top, subdirs, files);

            if (subdirs.Length == 0)
                yield return here;
            else
                foreach (var dir in subdirs)
                    foreach (var f in walk_v2(dir, topdown))
                        yield return f;
        }
    }
}
