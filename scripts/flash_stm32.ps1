param(
    [ValidateRange(0,127)][int]$Mask = 34,
    [switch]$Program
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$openocd = Join-Path $taskRoot 'third_party/mcu_tools/xpack-openocd-0.12.0-7/bin/openocd.exe'
if (-not (Test-Path -LiteralPath $openocd)) { throw 'Portable OpenOCD is missing; see firmware/stm32f103/README_中文.md.' }
$modelDir = Join-Path $taskRoot "results/stm32f103/seed42_mask$Mask"
$manifest = Get-Content -LiteralPath (Join-Path $modelDir 'manifest.json') -Raw | ConvertFrom-Json
$image = (Resolve-Path -LiteralPath (Join-Path $modelDir 'firmware.bin')).Path.Replace('\','/')
$actualHash = (Get-FileHash -LiteralPath $image -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualHash -ne $manifest.build.firmware_sha256) { throw 'Firmware binary hash mismatch.' }
$arguments = @('-f','interface/cmsis-dap.cfg','-c','transport select swd',
    '-f','target/stm32f1x.cfg','-c','adapter speed 1000')
if ($Program) {
    Write-Host "Writing verified mask $Mask image to STM32 Flash at 0x08000000."
    $arguments += @('-c',"program {$image} 0x08000000 verify reset exit")
} else {
    Write-Host 'Probe only: read device/Flash information, then resume the target.'
    $arguments += @('-c','init','-c','reset halt','-c','flash info 0',
        '-c','mdw 0xE0042000 1','-c','mdh 0x1FFFF7E0 1','-c','reset run','-c','shutdown')
}
& $openocd @arguments
if ($LASTEXITCODE -ne 0) { throw "OpenOCD exited with code $LASTEXITCODE. Save the complete output for diagnosis." }
