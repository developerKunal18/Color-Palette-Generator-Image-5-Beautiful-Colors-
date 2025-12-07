import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import Tk
import os

def select_image():
    root = Tk()
    root.withdraw()  # hide GUI
    file = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    return file

def extract_palette(image_path, k=5):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # reshape image for clustering
    pixels = img.reshape((-1, 3))

    # run KMeans
    kmeans = KMeans(n_clusters=k, random_state=0).fit(pixels)
    colors = kmeans.cluster_centers_.astype("uint8")

    return colors, img

def show_palette(colors, img):
    plt.figure(figsize=(10, 4))

    # original image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    # palette
    plt.subplot(1, 2, 2)
    plt.title("Color Palette")

    palette = np.zeros((100, 500, 3), dtype="uint8")

    step = 500 // len(colors)
    for i, color in enumerate(colors):
        palette[:, i * step:(i + 1) * step] = color

    plt.imshow(palette)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def main():
    print("\n🎨 Color Palette Generator\n")
    print("Select an image to extract its palette...\n")

    path = select_image()
    if not path:
        print("No image selected!")
        return

    print("Extracting palette...")
    colors, img = extract_palette(path, k=5)

    print("\nTop 5 Colors (RGB):")
    for i, c in enumerate(colors, 1):
        print(f"{i}. {tuple(c)}")

    show_palette(colors, img)

if __name__ == "__main__":
    main()
