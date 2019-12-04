using System;
using System.Data.SqlClient;

namespace Microsoft.ML.TestFrameworkCommon
{
    public static class Flakytestlogger
    {
        public static void WriteFailedTestToDB(string displayName, string os, string architecture, string framework,
            string configuration, int failCount, DateTime failTime)
        {
            string connetionString = "Server=tcp:mldotnet.database.windows.net,1433;Initial Catalog=flakytests;Persist Security Info=False;User ID=mldotnet;Password=Password2019^^;MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Connection Timeout=30;";
            string insStmt = "insert into flakytests ([displayname], [os], [architecture], [framework], [configuration], [failcount], [failtime]) values " +
                " (@displayname, @os, @architecture, @framework, @configuration, @failcount, @failtime)";

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
