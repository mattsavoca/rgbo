@echo off

REM Script to stop the DHCR Parser app Docker container

set CONTAINER_NAME=dhcr-parser-container

REM Check if the container is running
docker ps -q -f name=^%CONTAINER_NAME%$ > temp_container_id.txt
set /p RUNNING_CONTAINER_ID=<temp_container_id.txt
del temp_container_id.txt

if not "%RUNNING_CONTAINER_ID%"=="" (
    echo Stopping container %CONTAINER_NAME%...
    docker stop %CONTAINER_NAME%
    if %errorlevel% equ 0 (
        echo Container %CONTAINER_NAME% stopped successfully.
        echo (Container was run with --rm, so it should be removed automatically^).
    ) else (
        echo Failed to stop container %CONTAINER_NAME%. Check Docker status.
    )
) else (
    echo Container %CONTAINER_NAME% is not currently running.
) 