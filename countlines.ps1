# ====== CONFIGURATION ======
# Root directory to scan
$RootPath = "."

# File extensions considered "source code"
$SourceExtensions = @(
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".java", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".go", ".rs", ".php", ".rb",
    ".swift", ".kt", ".scala",
    ".html", ".css", ".scss",
    ".sql", ".sh", ".ps1", ".yaml", ".yml",
    ".json", ".xml", ".md"
)

# Folder names to exclude
$ExcludeDirs = @(
    "node_modules", "dist", "build", "out",
    ".git", ".svn", ".idea", ".vscode",
    "__pycache__", ".next", ".cache",
    "venv", ".venv", "env"
)

# ====== SCRIPT ======

$totalLines = 0
$fileStats = @()

Get-ChildItem -Path $RootPath -Recurse -File | Where-Object {

    # Exclude unwanted directories
    foreach ($dir in $ExcludeDirs) {
        if ($_.FullName -match "\\$dir\\") { return $false }
    }

    # Include only chosen source extensions
    return $SourceExtensions -contains $_.Extension.ToLower()

} | ForEach-Object {

    try {
        $lineCount = (Get-Content $_.FullName -ErrorAction Stop | Measure-Object -Line).Lines
        $totalLines += $lineCount

        $fileStats += [PSCustomObject]@{
            File = $_.FullName
            Lines = $lineCount
        }
    }
    catch {
        Write-Warning "Could not read file: $($_.FullName)"
    }
}

# ====== OUTPUT ======
$fileStats | Sort-Object Lines -Descending | Format-Table -AutoSize

Write-Host "`n==============================="
Write-Host "TOTAL SOURCE CODE LINES: $totalLines"
Write-Host "==============================="
