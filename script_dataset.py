import functions_dataset as functions
import tkinter as tk
from tkinter import simpledialog

ROOT = tk.Tk()
ROOT.withdraw()
USER_INP = simpledialog.askstring(title='Image Load', prompt='Which one is the starting image?')
im = USER_INP
 
a = simpledialog.askstring(title='Image Load', prompt='How many images in the sequence?')
    
sobel = []
k_means = []

for i in range(0,int(a)):
    
    image,name = functions.load(im)
    
    enhanced = functions.enhancement(im,name)
    
    seg_sobel = functions.sobel(enhanced,name)

    functions.Open_window(seg_sobel,name)
        

    im = str(int(im)+1)