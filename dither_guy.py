import tkinter as tk
from tkinter import filedialog, colorchooser
from PIL import Image, ImageTk
import numpy as np

def apply_dither(img, pixel_size, threshold, replace_color):
    img = img.convert('L')
    img = img.resize((img.width // pixel_size, img.height // pixel_size), Image.NEAREST)
    img_array = np.array(img, dtype=np.uint8)

    bayer_matrix = np.array([
        [  0, 128,  32, 160],
        [192,  64, 224,  96],
        [ 48, 176,  16, 144],
        [240, 112, 208,  80]
    ])

    bayer_matrix = (bayer_matrix / 255.0) * threshold

    for y in range(img_array.shape[0]):
        for x in range(img_array.shape[1]):
            if img_array[y, x] > bayer_matrix[y % 4, x % 4]:
                img_array[y, x] = 255
            else:
                img_array[y, x] = 0

    img = Image.fromarray(img_array)
    img = img.resize((img.width * pixel_size, img.height * pixel_size), Image.NEAREST)

    img = img.convert("RGB")
    img_data = np.array(img)

    img_data[(img_data[:, :, 0] == 255) & (img_data[:, :, 1] == 255) & (img_data[:, :, 2] == 255)] = replace_color

    img = Image.fromarray(img_data)
    return img

def update_image():
    if original_img:
        dithered_img = apply_dither(original_img, pixel_size.get(), threshold.get(), replace_color.get())
        img_tk = ImageTk.PhotoImage(dithered_img)
        img_label.config(image=img_tk)
        img_label.image = img_tk

def open_file():
    global original_img
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.*")])
    if file_path:
        original_img = Image.open(file_path)
        update_image()

def save_file():
    if original_img:
        dithered_img = apply_dither(original_img, pixel_size.get(), threshold.get(), replace_color.get())
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png")])
        if save_path:
            dithered_img.save(save_path)

def choose_color():
    color = colorchooser.askcolor(color="#658a00")[1]
    if color:
        replace_color.set(tuple(int(color[i:i+2], 16) for i in (1, 3, 5)))  # Convert hex to RGB

def invert_colors():
    global original_img
    if original_img:
        img_data = np.array(original_img.convert("RGB"))
        img_data = 255 - img_data
        inverted_img = Image.fromarray(img_data)
        original_img = inverted_img
        update_image()

root = tk.Tk()
root.title("Dither Guy")
root.geometry("900x700")

original_img = None

frame_buttons = tk.Frame(root)
frame_buttons.pack(pady=5)

btn_open = tk.Button(frame_buttons, text="Abrir Imagem", command=open_file)
btn_open.grid(row=0, column=0, padx=5, pady=5)

btn_save = tk.Button(frame_buttons, text="Salvar Imagem", command=save_file)
btn_save.grid(row=0, column=1, padx=5, pady=5)

btn_color = tk.Button(frame_buttons, text="Escolher Cor para Substituir", command=choose_color)
btn_color.grid(row=1, column=0, padx=5, pady=5)

btn_invert = tk.Button(frame_buttons, text="Inverter Cores", command=invert_colors)
btn_invert.grid(row=1, column=1, padx=5, pady=5)

pixel_size = tk.IntVar(value=4)
threshold = tk.IntVar(value=128)
replace_color = tk.Variable(value=(255, 255, 255))  # Default to white as tuple

frame = tk.Frame(root)
frame.pack()

tk.Label(frame, text="Pixelaridade:").pack(side='left')
tk.Scale(frame, from_=1, to=20, orient='horizontal', variable=pixel_size, command=lambda x: update_image()).pack(side='left')

tk.Label(frame, text="Threshold:").pack(side='left')
tk.Scale(frame, from_=0, to=400, orient='horizontal', variable=threshold, command=lambda x: update_image()).pack(side='left')



img_label = tk.Label(root)
img_label.pack()

root.mainloop()
