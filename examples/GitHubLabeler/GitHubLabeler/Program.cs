using System;
using System.Configuration;
using System.Threading.Tasks;

namespace GitHubLabeler
{
    internal static class Program
    {
        private static async Task Main(string[] args)
        {
            if (args.Length != 1)
            {
                Console.Error.WriteLine("error: wrong number of arguments.");
                PrintUsage();
                return;
            }

            var command = args[0];
            if (command == "--help" || command == "-h" || command == "-?")
            {
                PrintUsage();
            }
            else if (command == "train")
            {
                await Train();
            }
            else if (command == "label")
            {
                await Label();
            }
            else
            {
                Console.Error.WriteLine($"error: '{command}' is not a valid command.");
                PrintUsage();
            }
        }

        private static void PrintUsage()
        {
            Console.Error.WriteLine("usage: GitHubLabeler <command>");
            Console.Error.WriteLine("commands: ");
            Console.Error.WriteLine();
            Console.Error.WriteLine("       label   Uses the trained model to assign labels to");
            Console.Error.WriteLine("               issues that don't have any lables yet.");
            Console.Error.WriteLine();
            Console.Error.WriteLine("       train   Trains the model.");
        }

        private static async Task Train()
        {
            await Predictor.TrainAsync();
        }

        private static async Task Label()
        {
            var token = ConfigurationManager.AppSettings["GitHubToken"];
            var userName = ConfigurationManager.AppSettings["GitHubUserName"];
            var repoName = ConfigurationManager.AppSettings["GitHubRepositoryName"];

            if (string.IsNullOrEmpty(token) ||
                string.IsNullOrEmpty(userName) ||
                string.IsNullOrEmpty(repoName))
            {
                Console.Error.WriteLine("error: please configure the credentials in the app.config");
                return;
            }

            var labeler = new Labeler(userName, repoName, token);

            await labeler.LabelAllNewIssues();

            Console.WriteLine("Labeling completed");
        }
    }
}
