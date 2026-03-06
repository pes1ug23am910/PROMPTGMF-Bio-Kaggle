# Pre-Push Verification Script for PromptGFM-Bio
# Author: Yash Verma (PES1UG23AM910)
# Run this before pushing to GitHub to ensure everything is correct

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "  PromptGFM-Bio - GitHub Push Verification" -ForegroundColor Cyan
Write-Host "  Author: Yash Verma (PES1UG23AM910)" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

# Check 1: Verify we're in the right directory
Write-Host "[1/8] Checking current directory..." -ForegroundColor Yellow
$expectedPath = "E:\Lab\DLG\PromptGMF-Bio"
if ($PWD.Path -ne $expectedPath) {
    Write-Host "  ⚠️  WARNING: Not in project directory!" -ForegroundColor Red
    Write-Host "  Current: $($PWD.Path)" -ForegroundColor Red
    Write-Host "  Expected: $expectedPath" -ForegroundColor Red
    Write-Host "  Run: cd `"$expectedPath`"" -ForegroundColor Yellow
} else {
    Write-Host "  ✅ In correct directory" -ForegroundColor Green
}
Write-Host ""

# Check 2: Verify .env file exists and is ignored
Write-Host "[2/8] Checking .env file security..." -ForegroundColor Yellow
if (Test-Path ".env") {
    Write-Host "  ✅ .env file exists (contains GitHub token)" -ForegroundColor Green
    
    # Check if .env is in .gitignore
    $gitignoreContent = Get-Content ".gitignore" -Raw
    if ($gitignoreContent -match "\.env") {
        Write-Host "  ✅ .env is in .gitignore (will NOT be committed)" -ForegroundColor Green
    } else {
        Write-Host "  ❌ CRITICAL: .env NOT in .gitignore!" -ForegroundColor Red
        Write-Host "  Your token will be exposed!" -ForegroundColor Red
    }
} else {
    Write-Host "  ⚠️  .env file not found" -ForegroundColor Yellow
}
Write-Host ""

# Check 3: Verify no large model files will be committed
Write-Host "[3/8] Checking for large files..." -ForegroundColor Yellow
$largeExtensions = @("*.pt", "*.pth", "*.ckpt", "*.h5", "*.bin")
$foundLargeFiles = $false
foreach ($ext in $largeExtensions) {
    $files = Get-ChildItem -Path . -Recurse -Filter $ext -ErrorAction SilentlyContinue | Where-Object { $_.Length -gt 10MB }
    if ($files) {
        $foundLargeFiles = $true
        Write-Host "  ⚠️  Found large $ext files:" -ForegroundColor Red
        $files | ForEach-Object {
            $sizeMB = [math]::Round($_.Length / 1MB, 2)
            Write-Host "    - $($_.FullName) ($sizeMB MB)" -ForegroundColor Red
        }
    }
}
if (-not $foundLargeFiles) {
    Write-Host "  ✅ No large model files found (or properly ignored)" -ForegroundColor Green
}
Write-Host ""

# Check 4: Verify personal info is updated
Write-Host "[4/8] Checking personal information..." -ForegroundColor Yellow
$readmeContent = Get-Content "README.md" -Raw
$setupContent = Get-Content "setup.py" -Raw
$licenseContent = Get-Content "LICENSE" -Raw

$checks = @{
    "GitHub username" = ($readmeContent -match "pes1ug23am910" -and $setupContent -match "pes1ug23am910")
    "Author name" = ($readmeContent -match "Yash Verma" -and $setupContent -match "Yash Verma" -and $licenseContent -match "Yash Verma")
    "Email" = ($readmeContent -match "yashverma.pes@gmail.com" -and $setupContent -match "yashverma.pes@gmail.com")
}

foreach ($check in $checks.GetEnumerator()) {
    if ($check.Value) {
        Write-Host "  ✅ $($check.Key) updated" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $($check.Key) NOT updated!" -ForegroundColor Red
    }
}
Write-Host ""

# Check 5: Verify no placeholders remain
Write-Host "[5/8] Checking for placeholder text..." -ForegroundColor Yellow
$placeholders = @("yourusername", "your.email@example.com", "[Your Name]", "YOUR_USERNAME")
$foundPlaceholders = $false
foreach ($placeholder in $placeholders) {
    $files = Get-ChildItem -Path . -Include "*.md", "*.py", "*.yaml" -Recurse -ErrorAction SilentlyContinue | 
             Select-String -Pattern $placeholder -ErrorAction SilentlyContinue
    if ($files) {
        $foundPlaceholders = $true
        Write-Host "  ⚠️  Found placeholder '$placeholder' in:" -ForegroundColor Yellow
        $files | ForEach-Object { Write-Host "    - $($_.Path):$($_.LineNumber)" -ForegroundColor Yellow }
    }
}
if (-not $foundPlaceholders) {
    Write-Host "  ✅ No placeholders found" -ForegroundColor Green
}
Write-Host ""

# Check 6: Estimate repository size
Write-Host "[6/8] Estimating repository size..." -ForegroundColor Yellow
$srcSize = (Get-ChildItem -Path "src" -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
$scriptsSize = (Get-ChildItem -Path "scripts" -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
$docsSize = (Get-ChildItem -Path "docs" -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
$configsSize = (Get-ChildItem -Path "configs" -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
$totalSize = $srcSize + $scriptsSize + $docsSize + $configsSize

$totalSizeMB = [math]::Round($totalSize / 1MB, 2)
Write-Host "  Estimated size: $totalSizeMB MB" -ForegroundColor Cyan

if ($totalSizeMB -lt 100) {
    Write-Host "  ✅ Size is reasonable for GitHub (<100MB)" -ForegroundColor Green
} elseif ($totalSizeMB -lt 500) {
    Write-Host "  ⚠️  Size is large but acceptable" -ForegroundColor Yellow
} else {
    Write-Host "  ❌ Size is too large! Check for accidentally included files" -ForegroundColor Red
}
Write-Host ""

# Check 7: Verify Git configuration
Write-Host "[7/8] Checking Git configuration..." -ForegroundColor Yellow
try {
    $gitUser = git config user.name
    $gitEmail = git config user.email
    
    if ($gitUser -eq "Yash Verma" -and $gitEmail -eq "yashverma.pes@gmail.com") {
        Write-Host "  ✅ Git user configured: $gitUser <$gitEmail>" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  Git user not configured correctly" -ForegroundColor Yellow
        Write-Host "  Current: $gitUser <$gitEmail>" -ForegroundColor Yellow
        Write-Host "  Run:" -ForegroundColor Yellow
        Write-Host "    git config user.name 'Yash Verma'" -ForegroundColor Cyan
        Write-Host "    git config user.email 'yashverma.pes@gmail.com'" -ForegroundColor Cyan
    }
} catch {
    Write-Host "  ⚠️  Git not initialized yet" -ForegroundColor Yellow
    Write-Host "  Run: git init" -ForegroundColor Cyan
}
Write-Host ""

# Check 8: Required files checklist
Write-Host "[8/8] Checking required files..." -ForegroundColor Yellow
$requiredFiles = @{
    "README.md" = "Project overview"
    "LICENSE" = "MIT License"
    ".gitignore" = "Ignore rules"
    "requirements.txt" = "Dependencies"
    "setup.py" = "Package setup"
    "CONTRIBUTING.md" = "Contribution guide"
    "CHANGELOG.md" = "Version history"
    "CODE_OF_CONDUCT.md" = "Community guidelines"
}

foreach ($file in $requiredFiles.GetEnumerator()) {
    if (Test-Path $file.Key) {
        Write-Host "  ✅ $($file.Key) - $($file.Value)" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $($file.Key) missing!" -ForegroundColor Red
    }
}
Write-Host ""

# Summary
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "  VERIFICATION COMPLETE" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

Write-Host "📋 NEXT STEPS:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Review any warnings above" -ForegroundColor White
Write-Host "2. If all checks passed, proceed with:" -ForegroundColor White
Write-Host ""
Write-Host "   git init" -ForegroundColor Cyan
Write-Host "   git add ." -ForegroundColor Cyan
Write-Host "   git commit -m 'Initial commit: PromptGFM-Bio v1.0.0'" -ForegroundColor Cyan
Write-Host "   git remote add origin https://github.com/pes1ug23am910/PromptGFM-Bio.git" -ForegroundColor Cyan
Write-Host "   git branch -M main" -ForegroundColor Cyan
Write-Host "   git push -u origin main" -ForegroundColor Cyan
Write-Host ""
Write-Host "3. See PUSH_TO_GITHUB.md for detailed instructions" -ForegroundColor White
Write-Host ""

Write-Host "🔒 SECURITY REMINDER:" -ForegroundColor Red
Write-Host "   Your GitHub token is in .env which is excluded from git" -ForegroundColor Red
Write-Host "   NEVER commit or share your token!" -ForegroundColor Red
Write-Host ""

# Repository Info
Write-Host "📊 REPOSITORY INFO:" -ForegroundColor Yellow
Write-Host "   Author: Yash Verma" -ForegroundColor White
Write-Host "   SRN: PES1UG23AM910" -ForegroundColor White
Write-Host "   GitHub: pes1ug23am910" -ForegroundColor White
Write-Host "   URL: https://github.com/pes1ug23am910/PromptGFM-Bio" -ForegroundColor White
Write-Host ""
