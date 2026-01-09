@echo off

:: move to project root
cd..

:: run test suite with coverage
coverage run

:: export coverage to json / lcov for processing
coverage json
coverage lcov

:: export coverage to html for development in IDE
coverage html

:: report to console
coverage report
