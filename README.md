![](https://komarev.com/ghpvc/?username=manoelpiovesan&repo=dither-guy&color=blue)

# Dither Guy 

[![Build Status](https://github.com/manoelpiovesan/dither-guy/actions/workflows/main.yml/badge.svg)](https://github.com/manoelpiovesan/dither-guy/actions/workflows/main.yml)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/manoelpiovesan/dither-guy)](https://github.com/manoelpiovesan/dither-guy/releases/latest)
[![GitHub downloads](https://img.shields.io/github/downloads/manoelpiovesan/dither-guy/total)](https://github.com/manoelpiovesan/dither-guy/releases)
[![GitHub downloads (latest release)](https://img.shields.io/github/downloads/manoelpiovesan/dither-guy/latest/total)](https://github.com/manoelpiovesan/dither-guy/releases/latest)

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/L3L61SRB88)

Dither Guy is inspired by the [Dither Boy](https://studioaaa.com/product/dither-boy/) software, used to create graphic dithering effects, similar to old screens.

Is a simple Python script that converts images to dithered images. It supports multiple dithering algorithms to achieve the effect. The script is written in Python and uses the Pillow library to process images.

## 📥 Download

Download the latest release for your platform:

- **Windows**: [DitherGuy.exe](https://github.com/manoelpiovesan/dither-guy/releases/latest/download/DitherGuy.exe) | [DitherGuyVideo.exe](https://github.com/manoelpiovesan/dither-guy/releases/latest/download/DitherGuyVideo.exe)
- **Linux**: [DitherGuy](https://github.com/manoelpiovesan/dither-guy/releases/latest/download/DitherGuy) | [DitherGuyVideo](https://github.com/manoelpiovesan/dither-guy/releases/latest/download/DitherGuyVideo)
- **macOS**: [DitherGuy](https://github.com/manoelpiovesan/dither-guy/releases/latest/download/DitherGuy) | [DitherGuyVideo](https://github.com/manoelpiovesan/dither-guy/releases/latest/download/DitherGuyVideo)

Or visit the [releases page](https://github.com/manoelpiovesan/dither-guy/releases/latest) to see all available versions.

<img width="600" height="959" alt="image" src="https://github.com/user-attachments/assets/baf820e0-a0aa-4da2-8549-1f76f4531268" />

## Input x Output

<div style="display: inline-block; margin-right: 10px;">
  <img src="input/example.png" height="200">
</div>
<div style="display: inline-block;">
  <img src="output/example_output.png" height="200">
</div>
<div style="display: inline-block;">
  <img src="output/green_example_output.png" height="200">
</div>

## How to use

### Option 1: Use the pre-built executables (Recommended)

Download the executables from the [📥 Download section](#-download) above. No Python installation required!

### Option 2: Run from source

To use the script, you need to have Python installed on your computer. You also need to install the dependencies using the following command: 

```bash
pip install -r requirements.txt
```

After installing the dependencies, you can run the script using the following command:

```bash
python dither_guy.py
```

### Option 3: Build your own executable

To generate an executable file, you can use the following command:

```bash
pyinstaller --onefile dither_guy.py
```

Note: You need to have the PyInstaller library installed on your computer to generate the executable file. You can install
the PyInstaller library using the following command:

```bash
pip install pyinstaller
```

> **Note**: Executables are automatically built for Windows, Linux, and macOS via GitHub Actions when a new release tag is created.

## BETA Video Dithering
![Gif](input/computer_input.gif)
![Gif](output/computer.gif)


