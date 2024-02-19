'''
PyPower Projects
Mask Detection Using Machine Learning
'''

#USAGE : python gui_mask.py

from tkinter import *
import tkinter.messagebox
from PIL import ImageTk, Image
import cv2
import wget
from tkinter import filedialog

from tensorflow.keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import tkinter.filedialog as tkFileDialog
import cv2
import numpy as np

from PIL import Image


path = ""
image1 = ""
image2 = ""
panelA = None
panelB = None


path2 = "./sample1.jpg"
#show_image1(path2)


def select_image():
    global panelA, panelB,image1,plate,plate_text
    global path
    path = tkFileDialog.askopenfilename()
    if len(path) > 0:
        path1=path
        path1=(path1.split('/'))
        imgOriginalScene  = cv2.imread(path)
    
        image1 = cv2.resize(imgOriginalScene,(390,240))
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image1= Image.fromarray(image1)
        image1= ImageTk.PhotoImage(image1)

        image2 = cv2.imread(path2)
        #cv2.imshow("im",image2)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        image2 = cv2.resize(image2,(390,240))
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        image2= Image.fromarray(image2)
        image2= ImageTk.PhotoImage(image2)

        if panelA is None or panelB is None:
            panelA= Label(MainFrame, image = image1)
            panelA.grid(row=4,column=0,sticky=W)

            panelB = Label(MainFrame, image = image2)
            panelB.grid(row=4,column=1,sticky=W)
        

        else:
            panelA.configure(image=image1)
            panelA.grid(row=4,column=0,sticky=W)
            panelB.configure(image=image2)
            panelB.grid(row=4,column=1,sticky=W)
            panelA.image = image1
            panelA.grid(row=4,column=0,sticky=W)
            panelB.image =image2
            panelB.grid(row=4,column=1,sticky=W)




def show_image1(path2):
    #print(path2)
    image2 = cv2.imread(path2)
    #cv2.imshow("im",image2)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    image2 = cv2.resize(image2,(390,240))
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image2= Image.fromarray(image2)
    image2= ImageTk.PhotoImage(image2)

    global panelA, panelB

    if panelA is None or panelB is None:
                    
        panelA= Label(MainFrame, image = image1)
        panelA.grid(row=4,column=0,sticky=W)


        

    # otherwise, update the image panels
    else:
        # update the pannels
        panelA.configure(image=image1)
        panelA.grid(row=4,column=0,sticky=W)
        panelB.configure(image=image2)
        panelB.grid(row=4,column=1,sticky=W)
        panelA.image = image1
        panelA.grid(row=4,column=0,sticky=W)
        panelB.image =image2
        panelB.grid(row=4,column=1,sticky=W)



'''
#select image
def fileselector():
    global img_path
    main_win = tkinter.Tk()
    main_win.withdraw()

    main_win.overrideredirect(True)
    main_win.geometry('0x0+0+0')

    main_win.deiconify()
    main_win.lift()
    main_win.focus_force()

    main_win.sourceFile = filedialog.askopenfilename(filetypes = (("Image Files",("*.jpg","*.png","*.jpeg")),("All Files","*")),parent=main_win, initialdir= "./Testing",
    title='Please select a X-Ray Image')
    main_win.destroy()

    img_path = main_win.sourceFile
    print(img_path)
    tkinter.messagebox.showinfo("Image Selected","Click on Detect Button. \nTo get the COVID Prediction")
'''
flag = False

def predict():

    if(path==""):
        tkinter.messagebox.showinfo("Image Not Selected","Please Select X-Ray Image \nTo get the COVID Prediction")
    else:
        print("[INFO] loading network...")
        model =load_model('./mask_imagenet.h5')

        labels = ['Mask ON','NO Mask'] #These labels will be used for showing output
        start_point = (15, 15)
        end_point = (370, 80)
        thickness = -1

        print("[INFO] reading image...")
        frame = cv2.imread(path)

        roi_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.resize(frame,(224,224))
        roi = roi_gray.astype('float')/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi,axis=0)

        print("[INFO] classifying image...")

        preds = model.predict(roi)[0]
        #print(preds)
        #print(preds.argmax())
        label=labels[preds.argmax()]


        if(label=='NO Mask'):
            image = cv2.rectangle(frame, start_point, end_point, (0,0,255), thickness)
            cv2.putText(image,label,(30,60),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3)
        if(label=='Mask ON'):
            image = cv2.rectangle(frame, start_point, end_point, (0,255,0), thickness)
            cv2.putText(image,label,(30,60),cv2.FONT_HERSHEY_SIMPLEX,1.6,(0,0,0),3)

        #cv2.imshow('COVID Detector',image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        print("[INFO] saving image...")
        cv2.imwrite("./Output/detected.jpg",frame)
        #select_image()
        show_image1('./Output/detected.jpg')
        global flag
        flag = True
        #if flag:
            #show_image(image)
            #frame = cv2.imread("C:/Users/rushi/Desktop/flag.png")
            
        


root = Tk()
root.title("GUI : Mask Detection")

root.geometry("970x530")

root.configure(background = 'white')
Tops = Frame(root,bg = 'red',pady = 1, width =1750, height = 90, relief = "ridge")
Tops.grid(row=0,column=0)


Title_Label = Label(Tops,font=('Comic Sans MS',20,'bold'),text = "     PyPower  Presents  'Mask Prediction' \n\t   using Machine Learning\t\t",pady=9,bg= 'white',fg='red',justify ="center")
Title_Label.grid(row=0,column=0)
MainFrame = Frame(root,bg = 'white',pady=2,padx=2, width =1350, height = 100, relief = RIDGE)
MainFrame.grid(row=1,column=0)



Label_1 =Label(MainFrame, font=('lato black', 17,'bold'), text="\tDetect MASK on a person's face",padx=2,pady=2, bg="white",fg ="black")
Label_1.grid(row=0, column=0)

Label_2 =Label(MainFrame, font=('arial', 15,'bold'), text="",padx=2,pady=2, bg="white",fg = "black")
Label_2.grid(row=1, column=0,sticky=W)

Label_9 =Button(MainFrame, font=('arial', 19,'bold'), text="  Select Image ",padx=2,pady=2, bg="blue",fg = "white",command=select_image)
Label_9.grid(row=2, column=0)

Label_9 =Button(MainFrame, font=('arial', 19,'bold'), text="  Detect Mask ",padx=2,pady=2, bg="blue",fg = "white",command=predict)
Label_9.grid(row=2, column=1,sticky=W)

Label_2 =Label(MainFrame, font=('arial', 10,'bold'), text="",padx=2,pady=2, bg="white",fg = "black")
Label_2.grid(row=3, column=0,sticky=W)

Label_3 =Label(MainFrame, font=('arial', 30,'bold'), text="          \t\t\t",padx=2,pady=2, bg="white",fg = "black")
Label_3.grid(row=4, column=0)

'''
img = cv2.imread("./Picture1.png")
img = cv2.resize(img,(420,200))
cv2.imwrite('Picture1.png',img)
img = ImageTk.PhotoImage(Image.open("Picture1.png"))
panel = Label(MainFrame, image = img).grid(row=4,column=0,sticky=E)

img1 = cv2.imread("./Picture3.png")
img1 = cv2.resize(img1,(170,170))
cv2.imwrite('Picture3.png',img1)
img1 = ImageTk.PhotoImage(Image.open("Picture3.png"))
panel = Label(MainFrame, image = img1).grid(row=4,column=1,sticky=E)
'''
Label_3 =Label(MainFrame, font=('arial', 10,'bold'), text="\t\t\t\t          ",padx=2,pady=2, bg="white",fg = "black")
Label_3.grid(row=5, column=1)





root.mainloop()
