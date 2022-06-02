import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
from tkinter import Entry
from test import tinhToan_ChePhu

my_w = tk.Tk()
my_w.geometry("400x300")  # Size of the window 
my_w.title('Tính độ che phủ rừng')
my_font1=('times', 18, 'bold')
l1 = tk.Label(my_w,text='Import image',width=30,font=my_font1)
l1.grid(row=1,column=1)
b1 = tk.Button(my_w, text='Chose File',
   width=20,command = lambda:upload_file())
b1.grid(row=2,column=1) 
l2 = tk.Label(my_w,width=30,font=my_font1)
l2.grid(row=8,column=1)
def upload_file():
    try:
        f_types = [('Jpg,Png Files', '*.jpg;*.png')]
        filename = filedialog.askopenfilename(filetypes=f_types)
    except:
        print("Lỗi mở file")
    else:
        tl_che_phu = tinhToan_ChePhu(filename)
        l2.configure(text =f'Tỷ lệ che phủ là {tl_che_phu}%')
my_w.mainloop()  # Keep the window open