@echo off

REM --- Start Setup Block ---
set ENV_FILE=.env
set GEMINI_API_KEY_PRESENT=false

REM Function-like block to prompt for API key and update .env
:prompt_and_update_env
echo.
echo ---------------------------------------------------------------------
echo You **need a Gemini API key** to run the analysis features.
echo Get one for free at https://ai.google.dev/
echo ---------------------------------------------------------------------
echo.
set /p GEMINI_API_KEY_INPUT="Please paste your Gemini API Key and press Enter: "

REM Validate if the key looks potentially empty (basic check)
if "%GEMINI_API_KEY_INPUT%"=="" (
    echo Warning: No API key entered. Analysis features requiring the key will fail.
    REM Decide if you want to exit or continue without a key
    REM exit /b 1 REM Uncomment to exit if key is mandatory
)

echo Updating %ENV_FILE%...
REM Overwrite or create the file with the new content
(
echo GEMINI_API_KEY=%GEMINI_API_KEY_INPUT%
echo GEMINI_MODEL="gemini-1.5-flash-latest"
) > "%ENV_FILE%"
set GEMINI_API_KEY_PRESENT=true
echo %ENV_FILE% has been updated.
echo ---------------------------------------------------------------------
echo.
goto :end_prompt

REM Check if .env file exists
if exist "%ENV_FILE%" (
    echo %ENV_FILE% found.
    REM Check if GEMINI_API_KEY is present and non-empty within the file
    findstr /B /C:"GEMINI_API_KEY=" "%ENV_FILE%" > nul
    if %errorlevel% equ 0 (
        REM Check if the value part is non-empty (findstr checks the line, not just the key)
        for /f "tokens=1,* delims==" %%a in ('findstr /B /C:"GEMINI_API_KEY=" "%ENV_FILE%"') do (
            if not "%%b"=="" (
                echo Gemini API Key found and seems non-empty in %ENV_FILE%.
                set GEMINI_API_KEY_PRESENT=true
            ) else (
                echo Gemini API Key found but is empty in %ENV_FILE%.
                goto :prompt_and_update_env
            )
        )
    ) else (
        echo Gemini API Key line not found in %ENV_FILE%.
        goto :prompt_and_update_env
    )
) else (
    echo %ENV_FILE% not found.
    goto :prompt_and_update_env
)

:end_prompt

REM Optional: Exit if API key is mandatory and wasn't provided
REM if "%GEMINI_API_KEY_PRESENT%"=="false" (
REM     echo Error: Gemini API Key is required but was not provided. Exiting.
REM     exit /b 1
REM )
REM --- End Setup Block ---


REM Script to build and run the Docker container for the DHCR Parser app

set IMAGE_NAME=dhcr-parser-app
set CONTAINER_NAME=dhcr-parser-container

REM Build the Docker image
echo Building Docker image: %IMAGE_NAME%...
docker build -t %IMAGE_NAME% .

if %errorlevel% neq 0 (
    echo Docker build failed. Exiting.
    exit /b %errorlevel%
)

echo Build successful.

REM Check if a container with the same name is already running and stop it
docker ps -q -f name=^%CONTAINER_NAME%$ > temp_container_id.txt
set /p RUNNING_CONTAINER_ID=<temp_container_id.txt
del temp_container_id.txt

if not "%RUNNING_CONTAINER_ID%"=="" (
    echo Container %CONTAINER_NAME% is already running. Stopping it...
    docker stop %CONTAINER_NAME%
)

REM Check if a container with the same name exists (but is stopped) and remove it
docker ps -aq -f status=exited -f name=^%CONTAINER_NAME%$ > temp_container_id.txt
set /p STOPPED_CONTAINER_ID=<temp_container_id.txt
del temp_container_id.txt

if not "%STOPPED_CONTAINER_ID%"=="" (
    echo Removing existing stopped container %CONTAINER_NAME%...
    docker rm %CONTAINER_NAME%
)

REM Run the Docker container
echo Running Docker container: %CONTAINER_NAME%...
echo Access the application at http://localhost:8080

REM Run in detached mode (-d)
REM Automatically remove the container when it exits (--rm)
REM Map port 8080 on the host to port 8080 in the container (-p 8080:8080)
REM Give the container a name (--name)
REM Optional: Mount a local .env file (replace C:\path\to\your\.env with actual path)
REM docker run --rm -d -p 8080:8080 --name %CONTAINER_NAME% --env-file C:\path\to\your\.env %IMAGE_NAME%
REM Use the .env file in the current directory
docker run --rm -d -p 8080:8080 --name %CONTAINER_NAME% --env-file .env %IMAGE_NAME%

if %errorlevel% neq 0 (
    echo Failed to start Docker container %CONTAINER_NAME%. Check Docker logs.
    exit /b %errorlevel%
)

echo Container %CONTAINER_NAME% started successfully.

REM Attempt to open the browser
echo Attempting to open the application in your default browser...
start http://localhost:8080 