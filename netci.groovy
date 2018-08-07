// Import the utility functionality.

import jobs.generation.ArchivalSettings;
import jobs.generation.Utilities;

def project = GithubProject
def branch = GithubBranchName

['OSX10.13'].each { os ->
    ['Debug', 'Release'].each { config ->
        [true, false].each { isPR ->
            // Calculate job name
            def jobName = os.toLowerCase() + '_' + config.toLowerCase()

            def machineAffinity = 'latest-or-auto'

            def newJob = job(Utilities.getFullJobName(project, jobName, isPR)) {
                steps {
                    shell("./build.sh -$config")
                    shell("./build.sh -$config -runtests")
                    shell("./build.sh -buildPackages")
                }
            }

            Utilities.setMachineAffinity(newJob, os, machineAffinity)
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
