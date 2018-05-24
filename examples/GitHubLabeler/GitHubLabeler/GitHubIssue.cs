using Microsoft.ML.Runtime.Api;

#pragma warning disable 649 // We don't care about unsused fields here, because they are mapped with the input file.

namespace GitHubLabeler
{
    internal class GitHubIssue
    {
        [Column(ordinal: "0")]
        public string ID;

        [Column(ordinal: "1")]
        public string Area; // This is an issue label, for example "area-System.Threading"

        [Column(ordinal: "2")]
        public string Title;

        [Column(ordinal: "3")]
        public string Description;
    }
}
