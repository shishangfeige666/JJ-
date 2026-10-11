Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class Cap {
  [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h, IntPtr hdc, uint flags);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool ClientToScreen(IntPtr h, ref POINT p);
  [DllImport("user32.dll")] public static extern IntPtr GetWindowDC(IntPtr h);
  [DllImport("user32.dll")] public static extern int ReleaseDC(IntPtr h, IntPtr dc);
  [DllImport("gdi32.dll")] public static extern bool BitBlt(IntPtr d, int x, int y, int w, int h, IntPtr s, int sx, int sy, int rop);
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int L,T,R,B; }
  [StructLayout(LayoutKind.Sequential)] public struct POINT { public int X,Y; }
}
"@

$hwnd = [IntPtr]4983344
$c = New-Object Cap+RECT; [void][Cap]::GetClientRect($hwnd, [ref]$c)
$w = $c.R - $c.L; $h = $c.B - $c.T
Write-Host "client = ${w}x${h}"

$dir = "C:\Users\admin\Documents\deepseek-harness\JJ\_probe"

function Save-Bmp($bmp, $name) {
  $p = Join-Path $dir $name
  $bmp.Save($p, [System.Drawing.Imaging.ImageFormat]::Png)
  Write-Host "saved $p"
}

# --- method 1: PrintWindow flags=0
$b1 = New-Object System.Drawing.Bitmap($w, $h, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
$g1 = [System.Drawing.Graphics]::FromImage($b1)
$hdc1 = $g1.GetHdc()
$ok1 = [Cap]::PrintWindow($hwnd, $hdc1, 0)
$g1.ReleaseHdc($hdc1); $g1.Dispose()
Write-Host "PrintWindow(0) -> $ok1"
Save-Bmp $b1 "m1_printwindow0.png"

# --- method 2: PrintWindow flags=2 (PW_RENDERFULLCONTENT)
$b2 = New-Object System.Drawing.Bitmap($w, $h, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
$g2 = [System.Drawing.Graphics]::FromImage($b2)
$hdc2 = $g2.GetHdc()
$ok2 = [Cap]::PrintWindow($hwnd, $hdc2, 2)
$g2.ReleaseHdc($hdc2); $g2.Dispose()
Write-Host "PrintWindow(2) -> $ok2"
Save-Bmp $b2 "m2_printwindow2.png"

# --- method 3: window DC BitBlt (only visible part)
$b3 = New-Object System.Drawing.Bitmap($w, $h, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
$g3 = [System.Drawing.Graphics]::FromImage($b3)
$hdc3 = $g3.GetHdc()
$src = [Cap]::GetWindowDC($hwnd)
$ok3 = [Cap]::BitBlt($hdc3, 0, 0, $w, $h, $src, 0, 0, 0x00CC0020)
[void][Cap]::ReleaseDC($hwnd, $src)
$g3.ReleaseHdc($hdc3); $g3.Dispose()
Write-Host "WindowDC BitBlt -> $ok3"
Save-Bmp $b3 "m3_windowdc.png"

foreach ($pair in @(@($b1,"m1_printwindow0.png"), @($b2,"m2_printwindow2.png"), @($b3,"m3_windowdc.png"))) {
  $b = $pair[0]
  $nonBlack = 0; $total = 0
  for ($y = 0; $y -lt $h; $y += 40) {
    for ($x = 0; $x -lt $w; $x += 40) {
      $px = $b.GetPixel($x, $y); $total++
      if ($px.R + $px.G + $px.B -gt 24) { $nonBlack++ }
    }
  }
  Write-Host ("{0}: non-black samples {1}/{2}" -f $pair[1], $nonBlack, $total)
}
