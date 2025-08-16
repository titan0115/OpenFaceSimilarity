@echo off
echo Запуск TensorBoard для отслеживания тренировки...
echo TensorBoard будет доступен по адресу: http://localhost:6006
echo Нажмите Ctrl+C для остановки TensorBoard
echo.

.\venv\Scripts\python.exe -m tensorboard.main --logdir=runs --port=6006 --host=localhost

pause
