$pid_to_wait = 29520
Write-Host "Waiting for PID $pid_to_wait to finish..."
Wait-Process -Id $pid_to_wait -ErrorAction SilentlyContinue
Write-Host "PID $pid_to_wait finished. Starting Model_123 LOSO..."
Set-Location "d:\classification-main\classification-main"
python IMU_main_LOSO.py
Write-Host "Model_123 LOSO done."
