@echo off
echo Recording object detection demo...

:: Activate virtual environment if exists
if exist ..\myenv\Scripts\activate.bat (
    call ..\myenv\Scripts\activate.bat
)

:: Create recordings directory if it doesn't exist
if not exist recordings\ mkdir recordings\

:: Run the detection script with recording enabled
python real_time_detection.py --conf 0.5 --camera 0 --record

echo.
echo Demo recording complete. Check the output directory for the video.

:: Deactivate virtual environment
if exist ..\myenv\Scripts\deactivate.bat (
    call ..\myenv\Scripts\deactivate.bat
)

pause