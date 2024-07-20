@echo off

echo Running Ruff for linting...
ruff check .
set RESULT=%ERRORLEVEL%

if %RESULT% neq 0 (
    echo Ruff linting failed, commit aborted.
    exit /b 1
)

echo Ruff passed, running tests...

pytest --maxfail=1 --disable-warnings -v
set RESULT=%ERRORLEVEL%

if %RESULT% neq 0 (
    echo Tests failed, commit aborted.
    exit /b 1
)

echo All tests passed, proceeding with commit.
