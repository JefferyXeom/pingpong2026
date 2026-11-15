param(
    [string]$Source = (Join-Path ([Environment]::GetFolderPath('Desktop')) 'yolov5data'),
    [string]$ReportDirectory = (Join-Path $PSScriptRoot '..\reports\datasets\pingpong_v1'),
    [double]$MinimumFreeGiB = 5
)

# Read the source only. Write reports inside the workspace; never copy or alter samples.
$ErrorActionPreference = 'Stop'
$sourceItem = Get-Item -LiteralPath $Source
if (-not $sourceItem.PSIsContainer) { throw "Source is not a directory: $Source" }
$sourceRoot = $sourceItem.FullName.TrimEnd('\')
$workspaceRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$reportRoot = [IO.Path]::GetFullPath($ReportDirectory)
if (-not $reportRoot.StartsWith($workspaceRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Reports must be written inside the workspace.'
}
if ($reportRoot.StartsWith($sourceRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Reports must not be written inside the source dataset.'
}
$enumerationErrors = @()
$files = @(Get-ChildItem -LiteralPath $sourceRoot -Recurse -File -Force -ErrorAction SilentlyContinue -ErrorVariable +enumerationErrors)
$imageExtensions = @('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')
$images = @($files | Where-Object { $imageExtensions -contains $_.Extension.ToLowerInvariant() })
$texts = @($files | Where-Object { $_.Extension -ieq '.txt' })
$configs = @($files | Where-Object { $_.Extension.ToLowerInvariant() -in @('.yaml', '.yml', '.json') })
$totalBytes = [long](($files | Measure-Object -Property Length -Sum).Sum)
$volume = Get-PSDrive -Name ([IO.Path]::GetPathRoot($workspaceRoot).Substring(0, 1))
$reserveBytes = [long]($MinimumFreeGiB * 1GB)

$extensionSummary = @($files | Group-Object { $_.Extension.ToLowerInvariant() } | ForEach-Object {
    [pscustomobject]@{ extension = $_.Name; count = $_.Count; bytes = [long](($_.Group | Measure-Object Length -Sum).Sum) }
} | Sort-Object bytes -Descending)
$folderSummary = @($files | Group-Object {
    $relative = $_.FullName.Substring($sourceRoot.Length + 1)
    $parts = $relative -split '\\'
    if ($parts.Length -gt 2) { $parts[0] + '\' + $parts[1] }
    elseif ($parts.Length -gt 1) { $parts[0] }
    else { '(root)' }
} | ForEach-Object {
    [pscustomobject]@{ directory = $_.Name; files = $_.Count; bytes = [long](($_.Group | Measure-Object Length -Sum).Sum) }
} | Sort-Object bytes -Descending)

# Candidate matches are evidence for inspection, not permission to treat absent labels as negatives.
$textPaths = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$textsByStem = @{}
foreach ($label in $texts) {
    [void]$textPaths.Add($label.FullName)
    if (-not $textsByStem.ContainsKey($label.BaseName)) {
        $textsByStem[$label.BaseName] = [Collections.Generic.List[string]]::new()
    }
    $textsByStem[$label.BaseName].Add($label.FullName)
}
$matchedPaths = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$pairCounts = @{ adjacent = 0; images_to_labels = 0; unique_stem_candidate = 0; ambiguous = 0; missing = 0 }
$unresolvedExamples = [Collections.Generic.List[object]]::new()
foreach ($img in $images) {
    $adjacent = [IO.Path]::ChangeExtension($img.FullName, '.txt')
    $relativeImage = $img.FullName.Substring($sourceRoot.Length + 1)
    $mappedRelative = [regex]::Replace($relativeImage, '(^|\\)images(?=\\)', '${1}labels', 'IgnoreCase')
    $mapped = [IO.Path]::ChangeExtension((Join-Path $sourceRoot $mappedRelative), '.txt')
    $kind = 'missing'
    if ($textPaths.Contains($adjacent)) { $kind = 'adjacent'; [void]$matchedPaths.Add($adjacent) }
    elseif ($mapped -ne $adjacent -and $textPaths.Contains($mapped)) { $kind = 'images_to_labels'; [void]$matchedPaths.Add($mapped) }
    elseif ($textsByStem.ContainsKey($img.BaseName)) {
        if ($textsByStem[$img.BaseName].Count -eq 1) {
            $kind = 'unique_stem_candidate'
        } else { $kind = 'ambiguous' }
    }
    $pairCounts[$kind]++
    if ($kind -in @('missing', 'ambiguous', 'unique_stem_candidate') -and $unresolvedExamples.Count -lt 100) {
        $unresolvedExamples.Add([pscustomobject]@{ image = $relativeImage; status = $kind })
    }
}

# Bound the first inspection. The future preparation step must validate every selected label.
$labelSamples = [Collections.Generic.List[object]]::new()
foreach ($labelPath in @($matchedPaths | Sort-Object | Select-Object -First 100)) {
    $lines = @(Get-Content -LiteralPath $labelPath)
    $objects = 0
    $invalid = 0
    $classes = [Collections.Generic.HashSet[int]]::new()
    foreach ($line in $lines) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $fields = $line.Trim() -split '\s+'
        $classId = 0
        $valid = $fields.Count -eq 5
        if ($valid) {
            $valid = [int]::TryParse($fields[0], [ref]$classId) -and $classId -ge 0
        }
        if ($valid) {
            for ($i = 1; $i -lt 5; $i++) {
                $number = 0.0
                $parsed = [double]::TryParse($fields[$i], [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$number)
                if (-not $parsed -or [double]::IsNaN($number) -or [double]::IsInfinity($number) -or $number -lt 0 -or $number -gt 1 -or ($i -ge 3 -and $number -eq 0)) { $valid = $false }
            }
        }
        if ($valid) { $objects++; [void]$classes.Add($classId) } else { $invalid++ }
    }
    $labelSamples.Add([pscustomobject]@{
        path = $labelPath.Substring($sourceRoot.Length + 1)
        objects = $objects
        invalid_detection_rows = $invalid
        class_ids = @($classes | Sort-Object)
    })
}

$report = [ordered]@{
    generated_at = [DateTimeOffset]::Now.ToString('o')
    source = $sourceRoot
    workspace = $workspaceRoot
    source_modified = $false
    enumeration_complete = $enumerationErrors.Count -eq 0
    enumeration_errors = @($enumerationErrors | ForEach-Object { $_.ToString() })
    total_files = $files.Count
    total_bytes = $totalBytes
    image_files = $images.Count
    text_files = $texts.Count
    free_bytes = [long]$volume.Free
    minimum_free_bytes = $reserveBytes
    estimated_free_after_full_copy_bytes = [long]$volume.Free - $totalBytes
    full_copy_fits_reserve = ([long]$volume.Free - $totalBytes) -ge $reserveBytes
    space_estimate_note = 'Logical file bytes only; allocation overhead, training caches, environments and exports require additional space.'
    extensions = $extensionSummary
    directories = $folderSummary
    config_paths = @($configs | ForEach-Object { $_.FullName.Substring($sourceRoot.Length + 1) })
    label_pair_candidates = $pairCounts
    unresolved_examples = @($unresolvedExamples)
    label_sample_limit = 100
    sampled_labels = @($labelSamples)
    executables = @('python', 'py', 'nvidia-smi', 'trtexec' | ForEach-Object {
        $command = Get-Command $_ -ErrorAction SilentlyContinue
        [pscustomobject]@{ name = $_; path = if ($command) { $command.Source } else { $null } }
    })
}
[void](New-Item -ItemType Directory -Path $reportRoot -Force)
$reportPath = Join-Path $reportRoot 'dataset_inventory.json'
$report | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $reportPath -Encoding UTF8
[pscustomobject]@{
    Source = $sourceRoot
    Files = $files.Count
    Images = $images.Count
    SourceGiB = [math]::Round($totalBytes / 1GB, 2)
    FreeGiB = [math]::Round($volume.Free / 1GB, 2)
    EstimatedFreeAfterFullCopyGiB = [math]::Round(($volume.Free - $totalBytes) / 1GB, 2)
    EnumerationComplete = $enumerationErrors.Count -eq 0
    Report = $reportPath
} | Format-List
