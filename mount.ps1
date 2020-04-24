$connectTestResult = Test-NetConnection -ComputerName nugetpackage.file.core.windows.net -Port 445
if ($connectTestResult.TcpTestSucceeded) {
    # Save the password so the drive will persist on reboot
    cmd.exe /C "cmdkey /add:`"nugetpackage.file.core.windows.net`" /user:`"Azure\nugetpackage`" /pass:`"bALT8KhVcCefQFvd2Is97b5fHZpSoBL7Au0I1nAdj0hiIjikpjNoMJLjtVxVooY+INazcHDcXX4pMdF9qMBsqw==`""
    # Mount the drive
    New-PSDrive -Name Z -PSProvider FileSystem -Root "\\nugetpackage.file.core.windows.net\mlnetcinuget"-Persist
} else {
    Write-Error -Message "Unable to reach the Azure storage account via port 445. Check to make sure your organization or ISP is not blocking port 445, or use Azure P2S VPN, Azure S2S VPN, or Express Route to tunnel SMB traffic over a different port."
}