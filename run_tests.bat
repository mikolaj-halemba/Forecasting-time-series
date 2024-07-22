@echo off

echo Running Ruff for linting...
ruff check . --exclude notebooks
set RESULT=%ERRORLEVEL%

if %RESULT% neq 0 (
    echo Ruff linting failed, commit aborted.
    exit /b 1
)

echo Ruff passed, running Black for formatting...
black . --exclude notebooks
set RESULT=%ERRORLEVEL%

if %RESULT% neq 0 (
    echo Black formatting failed, commit aborted.
    exit /b 1
)

echo Black passed, running tests...

pytest --maxfail=1 --disable-warnings -v --ignore=notebooks
set RESULT=%ERRORLEVEL%

if %RESULT% neq 0 (
    echo Tests failed, commit aborted.
    exit /b 1
)

echo All tests passed, proceeding with commit.
