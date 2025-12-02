![](https://badgen.net/github/views/manoelpiovesan/dither-guy)

# Dither Guy (Updated)

Dither Guy is inspired by the [Dither Boy](https://studioaaa.com/product/dither-boy/) software, used to create graphic dithering effects, similar to old screens.

Is a simple Python script that converts images to dithered images. It supports multiple dithering algorithms to achieve the effect. The script is written in Python and uses the Pillow library to process images.

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

To use the script, you need to have Python installed on your computer. You also need to install the dependencies using the following command: 

```bash
pip install -r requirements.txt
```

After installing the dependencies, you can run the script using the following command:

```bash
python dither_guy.py
```

To generate an executable file, you can use the following command:

```bash
pyinstaller --onefile dither_guy.py
```

Note: You need to have the PyInstaller library installed on your computer to generate the executable file. You can install
the PyInstaller library using the following command:

```bash
pip install pyinstaller
```

## BETA Video Dithering
![Gif](input/computer_input.gif)
![Gif](output/computer.gif)


