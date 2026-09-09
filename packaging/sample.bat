@echo off
setlocal

REM ============================================================
REM  PDF-OCR-Picker - batch runner (no GUI)
REM  Full explanation (Japanese) is in README.md next to this file.
REM
REM  Keep this .bat in the SAME folder as pdf-sorter.exe
REM  (inside the extracted "pdf-sorter" folder).
REM
REM  Edit the two folders below, then double-click this file.
REM ============================================================

REM Input folder  - where the PDFs are
set "INPUT=C:\pdf\in"

REM Output folder - Match / Review / Other are created here
set "OUTPUT=C:\pdf\out"


REM --- Do not edit below this line ----------------------------
REM  %~dp0 = the folder this .bat lives in, so the exe next to it
REM  is always found even after you move the whole folder.

echo Sorting PDFs...
echo   Input : %INPUT%
echo   Output: %OUTPUT%
echo.

"%~dp0pdf-sorter.exe" batch "%INPUT%" --move-to "%OUTPUT%" --out "%~dp0result.jsonl"

echo.
echo Exit code: %ERRORLEVEL%   ^(0 = OK / 9 = error^)
echo Log file : %~dp0result.jsonl
echo.

REM ------------------------------------------------------------
REM  Handy options - add them to the pdf-sorter.exe line above:
REM    --dry-run     show what WOULD happen, move nothing
REM    --recursive   also read PDFs in sub-folders
REM    --pages 0     read every page (default: first page only)
REM
REM  Classify a single file (result is the exit code):
REM    "%~dp0pdf-sorter.exe" classify "C:\pdf\in\example.pdf"
REM       0 = invoice / 1 = not invoice / 2 = needs review / 9 = error
REM ------------------------------------------------------------

echo Press any key to close this window.
pause >nul
endlocal
