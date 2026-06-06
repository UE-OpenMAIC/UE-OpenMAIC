
Add-Type -AssemblyName System.Drawing

$out = 'D:\code\teacherT2S\ourClap\result\paper_pid_branch_selection_table.png'

$width = 1980
$height = 430
$left = 45
$top = 35
$rowH = 78
$colW = @(680,280,280,360,280)

$bmp = New-Object System.Drawing.Bitmap $width, $height
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit

$g.FillRectangle([System.Drawing.Brushes]::White, 0, 0, $width, $height)

$blackBrush = [System.Drawing.Brushes]::Black
$pen = New-Object System.Drawing.Pen ([System.Drawing.Color]::Black), 2
$thinPen = New-Object System.Drawing.Pen ([System.Drawing.Color]::Black), 1

$fontHeader = New-Object System.Drawing.Font("Times New Roman", 24, [System.Drawing.FontStyle]::Bold)
$fontBody = New-Object System.Drawing.Font("Times New Roman", 23, [System.Drawing.FontStyle]::Regular)
$fontBold = New-Object System.Drawing.Font("Times New Roman", 23, [System.Drawing.FontStyle]::Bold)

function DrawCell([string]$text, [int]$x, [int]$y, [int]$w, [int]$h, [bool]$bold, [string]$align) {
    if ($bold) {
        $font = $fontBold
    } else {
        $font = $fontBody
    }

    $fmt = New-Object System.Drawing.StringFormat
    if ($align -eq "left") {
        $fmt.Alignment = [System.Drawing.StringAlignment]::Near
    } else {
        $fmt.Alignment = [System.Drawing.StringAlignment]::Center
    }
    $fmt.LineAlignment = [System.Drawing.StringAlignment]::Center

    $rect = New-Object System.Drawing.RectangleF($x, $y, $w, $h)
    $g.DrawString($text, $font, $blackBrush, $rect, $fmt)
}

$totalW = 0
foreach ($w in $colW) { $totalW += $w }
$totalH = $rowH * 4

$g.DrawRectangle($pen, $left, $top, $totalW, $totalH)

for ($r = 1; $r -le 3; $r++) {
    $y = $top + $r * $rowH
    $g.DrawLine($thinPen, $left, $y, $left + $totalW, $y)
}

$x = $left
for ($c = 0; $c -lt $colW.Count - 1; $c++) {
    $x += $colW[$c]
    $g.DrawLine($thinPen, $x, $top, $x, $top + $totalH)
}

DrawCell 'Method' 45 35 680 78 $true 'center'
DrawCell 'ARI' 725 35 280 78 $true 'center'
DrawCell 'NMI' 1005 35 280 78 $true 'center'
DrawCell 'Covering' 1285 35 360 78 $true 'center'
DrawCell 'AMI' 1645 35 280 78 $true 'center'
DrawCell 'Unselected branches' 70 113 655 78 $false 'left'
DrawCell '0.4779' 725 113 280 78 $false 'center'
DrawCell '0.5318' 1005 113 280 78 $false 'center'
DrawCell '0.7174' 1285 113 360 78 $false 'center'
DrawCell '0.5181' 1645 113 280 78 $false 'center'
DrawCell 'PID-selected branches' 70 191 655 78 $true 'left'
DrawCell '0.5179' 725 191 280 78 $true 'center'
DrawCell '0.5853' 1005 191 280 78 $true 'center'
DrawCell '0.7336' 1285 191 360 78 $true 'center'
DrawCell '0.5683' 1645 191 280 78 $true 'center'
DrawCell 'Relative gain' 70 269 655 78 $true 'left'
DrawCell '+8.37%' 725 269 280 78 $true 'center'
DrawCell '+10.06%' 1005 269 280 78 $true 'center'
DrawCell '+2.27%' 1285 269 360 78 $true 'center'
DrawCell '+9.69%' 1645 269 280 78 $true 'center'

$bmp.Save($out, [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose()
$bmp.Dispose()
