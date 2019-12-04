using System;
using System.Data.SqlClient;

namespace Microsoft.ML.TestFrameworkCommon
{
    public static class Centrallizedlogger
    {
        static string connetionString = "Server=tcp:mldotnet.database.windows.net,1433;Initial Catalog=flakytests;Persist Security Info=False;User ID=mldotnet;Password=Password2019^^;MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Connection Timeout=30;";
        
        public static void LogFlakyTests(string displayName, string os, string architecture, string framework,
            string configuration, int failCount, string messages, DateTime failTime)
        {
            string insStmt = "insert into flakytests ([displayname], [os], [architecture], [framework], [configuration], [failcount], [messages], [failtime]) values " +
                " (@displayname, @os, @architecture, @framework, @configuration, @failcount, @messages, @failtime)";

            try
            {
                using (SqlConnection cnn = new SqlConnection(connetionString))
                {
                    cnn.Open();
                    SqlCommand insCmd = new SqlCommand(insStmt, cnn);

                    insCmd.Parameters.AddWithValue("@displayname", displayName);
                    insCmd.Parameters.AddWithValue("@os", os);
                    insCmd.Parameters.AddWithValue("@architecture", architecture);
                    insCmd.Parameters.AddWithValue("@framework", framework);
                    insCmd.Parameters.AddWithValue("@configuration", configuration);
                    insCmd.Parameters.AddWithValue("@failcount", failCount);
                    insCmd.Parameters.AddWithValue("@messages", messages);
                    insCmd.Parameters.AddWithValue("@failtime", failTime);
                    insCmd.ExecuteNonQuery();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Fail to write message into db with exception {ex.Message}");
            }
        }

        public static void LogUnhandleExceptions(string displayName, string os, string architecture, string framework,
            string configuration, string exceptionMessage, string callStack, string messages, DateTime failTime)
        {
            string insStmt = "insert into unhandleexceptions ([displayname], [os], [architecture], [framework], [configuration], [exceptionmessage], [callstack], [messages], [failtime]) values " +
                " (@displayname, @os, @architecture, @framework, @configuration, @exceptionmessage, @callstack, @messages, @failtime)";

            try
            {
                using (SqlConnection cnn = new SqlConnection(connetionString))
                {
                    cnn.Open();
                    SqlCommand insCmd = new SqlCommand(insStmt, cnn);

                    insCmd.Parameters.AddWithValue("@displayname", displayName);
                    insCmd.Parameters.AddWithValue("@os", os);
                    insCmd.Parameters.AddWithValue("@architecture", architecture);
                    insCmd.Parameters.AddWithValue("@framework", framework);
                    insCmd.Parameters.AddWithValue("@configuration", configuration);
                    insCmd.Parameters.AddWithValue("@exceptionmessage", exceptionMessage);
                    insCmd.Parameters.AddWithValue("@callstack", callStack);
                    insCmd.Parameters.AddWithValue("@messages", messages);
                    insCmd.Parameters.AddWithValue("@failtime", failTime);
                    insCmd.ExecuteNonQuery();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Fail to write message into db with exception {ex.Message}");
            }
        }
    }
}
