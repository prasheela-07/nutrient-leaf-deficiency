import tkinter as tk
from tkinter import filedialog
from tkinter import Frame
import tkinter.messagebox
from tkinter.filedialog import askopenfilename
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors
from tkinter import *
import numpy as np
from tkinter import messagebox

def endprogram():
        print ("\nProgram terminated!")
        sys.exit()


def imshow(window,img):
    cv2.namedWindow(window)
    cv2.imshow(window,img)
    n = cv2.waitKey(0) & 0xFF
    if n == ord('q' or 'Q'):
        cv2.destroyWindow(window)

        

def upload_image(e):
        global imageFile
        global fileName
        imageFile = askopenfilename(initialdir = "testdata")
        fileName = os.path.basename(imageFile)
        print(imageFile)
        img = cv2.imread(imageFile)
        img = cv2.resize(img ,((int)(img.shape[1]/5),(int)(img.shape[0]/5)))
        imshow('Original',img)



def gblur():
    global img
    #Guassian blur
    blur1 = cv2.GaussianBlur(img,(3,3),1)
    
    #mean-shift algo
    newimg = np.zeros((img.shape[0], img.shape[1],3),np.uint8)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER , 10 ,1.0)

    img = cv2.pyrMeanShiftFiltering(blur1, 20, 30, newimg, 0, criteria)
    #imshow('Mean Shift',img)

def canny():
    global img, roi
    global Tarea, perimeter
    global originalroi
    #Guassian blur
    blur = cv2.GaussianBlur(img,(11,11),1)
    #imshow('Gaussain Blur',blur)
    
    #Canny-edge detection
    canny = cv2.Canny(blur, 160, 290)
    
    canny = cv2.cvtColor(canny,cv2.COLOR_GRAY2BGR)
    #imshow('canny',canny)
    
    #contour to find leafs
    bordered = cv2.cvtColor(canny,cv2.COLOR_BGR2GRAY)
    contours,hierarchy = cv2.findContours(bordered, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    maxC = 0
    for x in range(len(contours)): #if take max or one less than max then will not work in
        if len(contours[x]) > maxC: # pictures with zoomed leaf images
            maxC = len(contours[x])
            maxid = x
    perimeter = cv2.arcLength(contours[maxid],True)
    Tarea = cv2.contourArea(contours[maxid])
    cv2.drawContours(neworiginal,contours[maxid],-1,(0,0,255))
    imshow('Mean Shift Image',neworiginal)

    #Creating rectangular roi around contour
    height, width, _ = canny.shape
    min_x, min_y = width, height
    max_x = max_y = 0
    frame = canny.copy()
    
    # computes the bounding box for the contour, and draws it on the frame,
    for contour, hier in zip(contours, hierarchy):
        (x,y,w,h) = cv2.boundingRect(contours[maxid])
        min_x, max_x = min(x, min_x), max(x+w, max_x)
        min_y, max_y = min(y, min_y), max(y+h, max_y)
        if w > 80 and h > 80:
            roi = img[y:y+h , x:x+w]
            originalroi = original[y:y+h , x:x+w]
            
    if (max_x - min_x > 0 and max_y - min_y > 0):
        roi = img[min_y:max_y , min_x:max_x]    
        originalroi = original[min_y:max_y , min_x:max_x]
        
    img = roi

def calarea():
    global Tarea, Infarea

    imghls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
    imghls[np.where((imghls==[30,200,2]).all(axis=2))] = [0,200,0]

    #Only hue channel
    huehls = imghls[:,:,0]
    
    
    huehls[np.where(huehls==[0])] = [35]

    
    #Thresholding on hue image
    ret, thresh = cv2.threshold(huehls,28,255,cv2.THRESH_BINARY_INV)
    imshow('Threshold',thresh)
    
    #Masking thresholded image from original image
    mask = cv2.bitwise_and(originalroi,originalroi,mask = thresh)
    imshow('MaskOut Image',mask)

    #Finding contours for all infected regions
    contours,heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    Infarea = 0
    for x in range(len(contours)):
        cv2.drawContours(originalroi,contours[x],-1,(0,0,255))
        
        #Calculating area of infected region
        Infarea += cv2.contourArea(contours[x])
    if Infarea > Tarea:
        Tarea = img.shape[0]*img.shape[1]
    print ('Perimeter: %.2f' %(perimeter))
    print ('Total area: %.2f' %(Tarea))

    #Finding the percentage of infection in the leaf
    print ('Infected area: %.2f' %(Infarea))
    try:
        per = 100 * Infarea/Tarea
    except ZeroDivisionError:
        per = 0
    print ('Percentage of infection region: %.2f' %(per))

def processing():
    global img, original, neworiginal
    img = cv2.imread(imageFile)
    img = cv2.resize(img ,((int)(img.shape[1]/5),(int)(img.shape[0]/5)))
    original = img.copy()
    neworiginal = img.copy()
    
    gblur()
    canny()
    calarea()




def classify():
    global Infarea
    if(Infarea == 0):
            print("The leaf is sufficiently healthy!")
            messagebox.showinfo("Title", "The leaf is sufficiently healthy!")
    else:
            print("The leaf is infected!")
            messagebox.showinfo("Title", "The leaf is infected!")
    endprogram()





root = tk.Tk()
root.geometry("1000x550")       
root.title("Nutrient detection")

frame1 = Frame(master=root,height=200)
frame1.pack(fill=X)

frame2 = Frame(master=root,height=300)
frame2.pack(fill=X)

frame3 = Frame(master=root,height=200)
frame3.pack(fill=X)

frame4 = Frame(master=root,height=200)
frame4.pack(fill=X)

frame5 = Frame(master=root,height=200)
frame5.pack(fill=X)

label1 = tk.Label(frame1,text="Diagnosis of Nutrient Deficiency Symptoms in Plant Leaf Image",fg="red")
label1.pack(padx=150,pady=50)
label1.config(font=("times new roman", 15))

scale_w = 400
scale_h = 250

w1 = tk.Label(frame2, image='').pack(side="right",padx=10,pady=0)

w2 = tk.Label(frame2, image='')
w2.pack(padx=10,pady=20)

print(w2)

upload = tk.Button(frame3,text='Upload Image',fg="red",width=25,height=2)
upload.bind('<Button>', upload_image)
upload.pack(padx=115)

button_select = tk.Button(frame4,text='Image Processing',fg="red",width=25,height=2, command=processing)
button_select.pack(side='bottom', padx=115)

button = tk.Button(frame5,text='Classify',fg="black",bg="#ffc2b3",width=25,height=2,command=classify)
button.pack(side="bottom",pady=30)

root.mainloop()
