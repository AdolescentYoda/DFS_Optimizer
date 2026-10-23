@echo off
setlocal
title NFL Lineup Optimizer - Odds API Updater

echo.
echo ==========================================
echo      NFL LINEUP OPTIMIZER - PROP UPDATE
echo ==========================================
echo.

set "WORKBOOK=NFL Current Week Projections.xlsx"
set /p API_KEY=Enter your The Odds API key: 

if "%API_KEY%"=="" (
    echo.
    echo No API key entered.
    pause
    exit /b 1
)

set "ODDS_API_KEY=%API_KEY%"

echo.
echo Updating sportsbook props...
echo.

python nfl_odds_api_to_excel_2.py "%WORKBOOK%"

echo.
if errorlevel 1 (
    echo UPDATE FAILED.
) else (
    echo UPDATE COMPLETE.
    echo Output saved to "%WORKBOOK%"
)
echo.
pause