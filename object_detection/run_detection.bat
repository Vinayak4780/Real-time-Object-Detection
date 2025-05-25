@echo off
echo Running real-time object detection...

:: Activate virtual environment if exists
if exist ..\myenv\Scripts\activate.bat (
    call ..\myenv\Scripts\activate.bat
)

:: Run the detection script
python real_time_detection.py --conf 0.5 --camera 0

:: Deactivate virtual environment
if exist ..\myenv\Scripts\deactivate.bat (
    call ..\myenv\Scripts\deactivate.bat
)

pause