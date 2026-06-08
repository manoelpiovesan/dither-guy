import PyInstaller.__main__
import platform

# The path separator for PyInstaller's add-data is different on Windows vs Unix
sep = ';' if platform.system() == 'Windows' else ':'

args = [
    'main.py',
    '--name=DitherGuy',
    '--windowed',
    '--icon=app_icon.png',
    
    # -------------------------------------------------------------
    # FORCE BUNDLE OPTIONAL DEPENDENCIES
    # PyInstaller normally ignores modules inside try/except blocks.
    # We must explicitly collect them here so they aren't skipped.
    # -------------------------------------------------------------
    
    # 1. Numba and CUDA GPU support
    '--collect-all=numba',
    '--collect-all=llvmlite',
    '--hidden-import=numba.cuda',
    '--hidden-import=numba._devicearray',
    
    # 2. Audio and Multimedia support (PySide6)
    '--collect-all=PySide6.QtMultimedia',
    '--hidden-import=PySide6.QtMultimedia',
    
    # 3. OpenCV
    '--collect-all=cv2',
    
    # -------------------------------------------------------------
    # ASSETS
    # -------------------------------------------------------------
    f'--add-data=app_icon.png{sep}.',
    f'--add-data=presets{sep}presets',
    
    # Clean previous builds
    '--clean',
    '--noconfirm'
]

print("Starting PyInstaller build with arguments:")
for arg in args:
    print(f"  {arg}")

PyInstaller.__main__.run(args)
