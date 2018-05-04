// Import the utility functionality.

import jobs.generation.ArchivalSettings;
import jobs.generation.Utilities;
import jobs.generation.InternalUtilities;

def project = GithubProject
def branch = GithubBranchName

['Windows_NT', 'Linux', 'OSX10.13'].each { os ->
    ['Debug', 'Release'].each { config ->
        [true, false].each { isPR ->
            // Calculate job name
            def jobName = os.toLowerCase() + '_' + config.toLowerCase()
            def buildFile = '';

            def machineAffinity = 'latest-or-auto'

            // Calculate the build command
            if (os == 'Windows_NT') {
                buildFile = ".\\build.cmd"
            } else {
                buildFile = "./build.sh"
            }

            def buildCommand = buildFile + " -$config -runtests"
            def packCommand = buildFile + " -buildPackages"

            def newJob = job(InternalUtilities.getFullJobName(project, jobName, isPR)) {
                steps {
                    if (os == 'Windows_NT') {
                        batchFile(buildCommand)
                        batchFile(packCommand)
                    }
                    else {
                        // Shell
                        shell(buildCommand)
                        shell(packCommand)
                    }
                }
            }

            def osImageName = os
            if (os == 'Linux') {
                // Trigger a portable Linux build that runs on RHEL7.2
                osImageName = "RHEL7.2"
            }

            Utilities.setMachineAffinity(newJob, osImageName, machineAffinity)
            InternalUtilities.standardJobSetup(newJob, project, isPR, "*/${branch}")

            if (isPR) {
                Utilities.addGithubPRTriggerForBranch(newJob, branch, "$os $config")
            }

            Utilities.addMSTestResults(newJob, 'bin/**/*.trx')

            def archiveSettings = new ArchivalSettings()
            archiveSettings.addFiles('bin/**/*')
            archiveSettings.excludeFiles('bin/obj/**')
            archiveSettings.setFailIfNothingArchived()
            archiveSettings.setArchiveOnFailure()
            Utilities.addArchival(newJob, archiveSettings)
        }
    }
}
