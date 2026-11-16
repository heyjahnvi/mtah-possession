@echo off
REM One-click run on Windows (needs Python 3.12 or 3.13 and git).
REM Usage: RUN_ME_WINDOWS.bat TeamName "C:\path\to\bundle"
REM The bundle folder is the one that contains tactical\ (download it from the organisers' Drive).
setlocal
set TEAM=%~1
if "%TEAM%"=="" set TEAM=submission
set DATA=%~2
if "%DATA%"=="" set DATA=%~dp0data

cd /d "%~dp0"
if not exist .venv ( python -m venv .venv || goto :err )
call .venv\Scripts\activate.bat
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt || goto :err

python run.py --team "%TEAM%" --data "%DATA%" || goto :err

if not exist more-than-a-hack-2027 ( git clone --depth 1 https://github.com/metrica-sports/more-than-a-hack-2027.git )
copy /Y "outputs\%TEAM%.csv" "more-than-a-hack-2027\evaluation\results\%TEAM%.csv" >nul
cd more-than-a-hack-2027
python evaluation\validate_submission.py "evaluation\results\%TEAM%.csv"
python evaluation\run_evaluation.py --no-detail
cd ..
echo.
echo DONE. File: outputs\%TEAM%.csv    Log: outputs\%TEAM%_run_log.txt
pause
exit /b 0
:err
echo Something failed above, check the error message.
pause
exit /b 1
