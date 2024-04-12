import tkinter as tk
from PIL import Image, ImageTk

def report_pixel(event):
    # Calculating scaling
    scale_x = original_width / target_width
    scale_y = original_height / target_height

    # Converting coordinates using scaling
    original_x = int(event.x * scale_x)
    original_y = int(event.y * scale_y)

    # Getting pixel values from the original image
    pixel = image.getpixel((original_x, original_y))

    # Updating Label Information
    pixel_info_var.set(f"Original Location: ({original_x}, {original_y}) | Pixel value: {pixel}")

# Creating the main window
root = tk.Tk()
root.title("Pixel Information")

# Create string variables to update pixel information
pixel_info_var = tk.StringVar()

# Load Image
image = Image.open('static/junction.png')  # Specify the image path

original_width, original_height = image.size  # Save original size

# Resize images to fit the screen
target_width = 800  # target width
target_height = 600  # target height
resized_image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)

image_tk = ImageTk.PhotoImage(resized_image)

# Display pixel information
pixel_info_label = tk.Label(root, textvariable=pixel_info_var)
pixel_info_label.pack()

# Create a Canvas to place the image, sized to match the resized image
canvas = tk.Canvas(root, width=target_width, height=target_height)
canvas.pack()
canvas.create_image(0, 0, anchor='nw', image=image_tk)

# Binding mouse click events
canvas.bind("<Button-1>", report_pixel)

# Start event loop
root.mainloop()
