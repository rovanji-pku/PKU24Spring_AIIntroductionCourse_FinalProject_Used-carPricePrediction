import os
import winshell
from win32com.client import Dispatch
import sys
def create_shortcut():
    # 获取桌面路径
    desktop = winshell.desktop()
    shortcut_path = os.path.join(desktop, "used_car_prediction.lnk")

    # 指定 Python 可执行文件路径和目标脚本路径

    python_executable = sys.executable
    target_script = os.path.join(os.getcwd(), "ui_main.py")
    if not os.path.exists(target_script):
        raise FileNotFoundError(f"{target_script} 不存在，请确保文件路径正确。")

    # 创建快捷方式
    shell = Dispatch('WScript.Shell')
    shortcut = shell.CreateShortcut(shortcut_path)
    shortcut.TargetPath = python_executable
    shortcut.Arguments = f'"{target_script}"'
    shortcut.WorkingDirectory = os.getcwd()
    shortcut.IconLocation = target_script
    shortcut.save()
    print(f"快捷方式已创建：{shortcut_path}")

if __name__ == "__main__":
    create_shortcut()