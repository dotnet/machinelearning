// Import the utility functionality.

import jobs.generation.ArchivalSettings;
import jobs.generation.Utilities;

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

            def newJob = job(Utilities.getFullJobName(project, jobName, isPR)) {
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
            Utilities.standardJobSetup(newJob, project, isPR, "*/${branch}")

            if (isPR) {
                Utilities.addGithubPRTriggerForBranch(newJob, branch, "$os $config")
            }
            else {
                Utilities.addGithubPushTrigger(newJob)
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
