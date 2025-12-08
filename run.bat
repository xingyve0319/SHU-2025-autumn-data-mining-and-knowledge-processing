@echo off
REM 进入虚拟环境（如果你有）
call .venv\Scripts\activate.bat

REM 进入项目根目录
cd /d %~dp0

REM 运行训练程序
python -m src.train

REM 训练完成提示
echo 训练完成，模型和图像已保存！
pause
