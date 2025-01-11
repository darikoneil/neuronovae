@echo off

:: move to project root
cd ..

:: formatting imports (paths because the config is broken & don't care to make it work)
isort ./neuronovae ./tests

:: formatting code (paths because the config is broken & don't care to make it work)
black ./neuronovae ./tests

:: linting (putting paths here too because autism)
flake8 ./neuronovae
