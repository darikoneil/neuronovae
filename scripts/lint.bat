@echo off

echo [0;33m "Formatting & Linting --> neuronovae" [0m

:: move to project root
cd ..

echo [0;33m "Formatting (RUFF)..." [0m
:: run ruff formatter
ruff format

echo [0;33m "Linting (RUFF)..." [0m
:: run ruff linter
ruff check ./neuronovae ./tests -o .ruff.json --output-format json --fix --no-cache

echo [0;33m "Finished Formatting & Linting --> neuronovae" [0m