# Dither Guy

[![Build Status](https://github.com/manoelpiovesan/dither-guy/actions/workflows/main.yml/badge.svg)](https://github.com/manoelpiovesan/dither-guy/actions/workflows/main.yml)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/manoelpiovesan/dither-guy)](https://github.com/manoelpiovesan/dither-guy/releases/latest)
[![GitHub downloads](https://img.shields.io/github/downloads/manoelpiovesan/dither-guy/total)](https://github.com/manoelpiovesan/dither-guy/releases)
[![GitHub downloads (latest release)](https://img.shields.io/github/downloads/manoelpiovesan/dither-guy/latest/total)](https://github.com/manoelpiovesan/dither-guy/releases/latest)

**Dither Guy** is a software inspired by [Dither Boy](https://studioaaa.com/product/dither-boy/), designed to create graphic dithering effects similar to those found on old displays and low color-depth systems.

The project focuses on providing a simple and straightforward tool for applying dithering to images and videos, preserving a controlled and retro visual aesthetic.

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/L3L61SRB88)

---

## Download

The latest binaries for each platform are available on the releases page:

👉 https://github.com/manoelpiovesan/dither-guy/releases/latest

<img width="600" height="959" alt="Dither Guy interface" src="https://github.com/user-attachments/assets/baf820e0-a0aa-4da2-8549-1f76f4531268" />

---

## Input × Output

Examples of input and output using different dithering configurations:

| Original | Dithered | Colored |
|----------|----------|---------|
|![Original](input/example.png)|![Dithered](output/example_output.png)| ![Colored](output/green_example_output.png)|

---

## Running in Development Mode

To run the project locally in development mode:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 dither_guy.py

```

## BETA Video Dithering
![Gif](input/computer_input.gif)
![Gif](output/computer.gif)


