@echo off
flake8 %1
if %errorlevel% equ 0 python %*
