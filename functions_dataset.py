# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

drawing = False # true if mouse is pressed
mode = 0 # if True, draw circle.
ix,iy = -1,-1
radius = 2
img = np.zeros((1080,1440,3), np.uint8)
enhanced = img = np.zeros((1080,1440,3), np.uint8)
ref_point = []
#-----------------Get Dark Channel--------------------------------------------
def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz,1);
    imvec = im.reshape(imsz,3);

    indices = darkvec.argsort();
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A

def TransmissionEstimate(im,A,sz):
    omega = 0.8;
    im3 = np.empty(im.shape,im.dtype);

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz);
    return transmission

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r));
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r));
    cov_Ip = mean_Ip - mean_I*mean_p;

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r));
    var_I   = mean_II - mean_I*mean_I;

    a = cov_Ip/(var_I + eps);
    b = mean_p - a*mean_I;

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r));

    q = mean_a*im + mean_b;
    return q;

def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray)/255;
    r = 15;
    eps = 0.05;
    t = Guidedfilter(gray,et,r,eps);

    return t;

def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype);
    t = cv2.max(t,tx);

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]
    return res

#---------------------- Contrast Enhancement ----------------------------------
def enhancement(image,name): 
    global enhanced
    res = name.split('_')
    I = img.astype('float32')/255  
    '''
    plt.hist(I.ravel(),256,[0,1])
    plt.title('Histogram before enhancement '+res[1])
    plt.xlabel('Pixels')
    plt.ylabel('Intensity')
    plt.savefig('Histograms\\Histogram_before_'+res[1]+'.png')
    plt.clf()
    '''
    dark = DarkChannel(I,15)
    A = AtmLight(I,dark)
    te = TransmissionEstimate(I, A, 15)
    t = TransmissionRefine(img,te)
    J = Recover(I,t,A,0.6)
    '''
    plt.hist(J.ravel(),256,[0,1])
    plt.title('Histogram after enhancement '+res[1])
    plt.xlabel('Pixels')
    plt.ylabel('Intensity')
    plt.savefig('Histograms\\Histogram_after_'+res[1]+'.png')
    plt.clf()
    '''
    
    enhanced = J
    
    return J
#------------------------Load Image-------------------------------------------

def load(user):
    
    global img
    
    img = cv2.imread(r'in\\'+'in_'+user+'.png')
    print(img.shape)
    window = 'in_'+user
    
    return img,window


#-------------------------Contour Area-----------------------------------------

def is_contour_bad(c,a): # Decide what I want to find and its features
    peri = cv2.contourArea(c) # Find areas
    return peri < a

#------------------------ SOBEL SEGMENTATION-----------------------------------


def sobel(image, name):
    
    res = name.split('_')
           
    image_filtered = cv2.medianBlur(image,5)
    
    gray = cv2.cvtColor(image_filtered, cv2.COLOR_BGR2GRAY)
    
    kernel = np.ones((10,10),np.uint8)
    kernel_2 = np.ones((4,4),np.uint8)
    kernel_3 = np.ones((25,25),np.uint8)
    
    H = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    V = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    
    abs_grad_H = cv2.convertScaleAbs(H,alpha=255,beta=-15)
    abs_grad_V = cv2.convertScaleAbs(V,alpha=255,beta =-15)
    
    edges = cv2.addWeighted(abs_grad_H,1,abs_grad_V,1,0)
                    
    edges_binary = cv2.threshold(edges,80,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
    edges_binary = cv2.morphologyEx(edges_binary,cv2.MORPH_OPEN,kernel_2)
    edges_binary = cv2.morphologyEx(edges_binary, cv2.MORPH_CLOSE, kernel)

    
    asd, contours, hierarchy = cv2.findContours(edges_binary,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    for c in contours:
        # If the contour satisfies "is_contour_bad", draw it on the mask
        if is_contour_bad(c,15000):
            # Draw black contour on gray image, instead of using a mask
            cv2.drawContours(edges_binary, [c], -1, 0, -1)
        else:
            cv2.drawContours(edges_binary, [c], -1, 255, -1)
   
    edges_binary = cv2.morphologyEx(edges_binary, cv2.MORPH_CLOSE, kernel_3)
    
    mask = cv2.bitwise_not(edges_binary)
        
    lower_black = np.array([0,0,0], dtype = "float32")/255
    upper_black = np.array([1,1,1], dtype = "float32")/255
       
    blacked_image = cv2.bitwise_and(image,image,mask = mask)
    
    mask2 = cv2.inRange(blacked_image,lower_black,upper_black)
    
    aux_image = image.copy()
    aux_image[mask2>0]= (0,1,0)
    
    mask_chars(image, aux_image, 983, 1024, 50, 295, 5)
    mask_chars(image, aux_image, 940, 975, 50, 295, 5)
    mask_chars(image, aux_image, 45, 150, 1315, 1395, 30)
    mask_chars(image, aux_image, 35, 90, 55, 250, 6)
    mask_chars(image, aux_image, 80, 125, 75, 225, 6)
    mask_chars(image, aux_image, 942, 975, 1218, 1372, 6)
    mask_chars(image, aux_image, 988, 1022, 1218, 1372, 6)
    
    overlay_mask = cv2.addWeighted(image,1,aux_image,0.5,0)
    
    return overlay_mask

#----------------- MASKING CHARACTERS -----------------------------------------

def mask_chars(image, aux_image, startY, endY, startX, endX, kernel_size):
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    
    roi = image[startY:endY, startX:endX]
    roi_mask = aux_image[startY:endY, startX:endX]
    
    white_l = np.array([110,110,110],dtype= 'float64')/255
    white_h = np.array([255,255,255],dtype= 'float64')/255

    letters = cv2.inRange(roi, white_l,white_h)
    
    letters = cv2.morphologyEx(letters,cv2.MORPH_CLOSE,kernel)
    
    letters = cv2.bitwise_not(letters)    
    mask = cv2.bitwise_and(roi, roi, mask=letters)
    
    lower_black = np.array([0,0,0], dtype = "float64")/255
    upper_black = np.array([1,1,1], dtype = "float64")/255
    
    mask2 = cv2.inRange(mask, lower_black, upper_black)
    
    roi[mask2>0] = (0,0,1)
    roi_mask[mask2>0] = (0,0,1)
    
    
    return roi
    

#-------------------------- K MEANS SEGMENTATION ------------------------------
def k_means(image,name,K):
    res_name = name.split('_')
    
    kernel = np.ones((20,20),np.uint8)
    kernel_2 = np.ones((3,3),np.uint8)
    
    image_filtered = cv2.GaussianBlur(image,(7,7),0)
    Z = image_filtered.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center*255)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))    
                  
    res2 = cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY)
    res2 = cv2.fastNlMeansDenoising(res2,None,16,5,25)
        
    mask = cv2.adaptiveThreshold(res2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,51,1)
    
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel_2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    plt.imshow(mask)
    
    asd, contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
    # If the contour satisfies "is_contour_bad", draw it on the mask
        if is_contour_bad(c,5000):
        # Draw black contour on gray image, instead of using a mask
            cv2.drawContours(mask, [c], -1, 0, -1)
        else:
            cv2.drawContours(mask, [c], -1, 255, -1)
            
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.bitwise_not(mask)
    
    lower_black = np.array([0,0,0], dtype = "float64")/255
    upper_black = np.array([1,1,1], dtype = "float64")/255

    blacked_image = cv2.bitwise_and(image,image, mask = mask)
    
    mask2 = cv2.inRange(blacked_image, lower_black, upper_black)
    aux_image = image.copy()
    aux_image[mask2>0]= (0,1,0)
    
    
    overlay_mask = cv2.addWeighted(image,1,aux_image,0.5,0)
    
    #cv2.imwrite('C:\\Users\\Utilizador\\Desktop\\Tese\\Scripts\\Dataset Creation\\Outputs_1.1\\'+'out_clustering_'+res_name[1]+'.png',overlay_mask*255)
    return overlay_mask


    
#------------------Mouse Controls----------------------------------------------
def draw(event,x,y,flags,param):
    
    global ix,iy,drawing,mode,radius,img, enhanced, ref_point
    copy = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
        if mode == 3:
            ref_point =[x,y]
        
    elif (event==cv2.EVENT_MOUSEWHEEL):
    # change eraser radius
        if flags > 0:
            radius +=   2
        elif flags<=0:
    # prevent issues with < 0
            if radius > 5:
                radius -=   2
                
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == 1:
                cv2.circle(img,(x,y),radius,(0,255,0),-1)
            elif mode == 2:
                cv2.circle(img,(x,y),radius,(0,0,0),-1)
            elif mode == 3:
                img[ref_point[1]-radius:ref_point[1]+radius,ref_point[0]-radius:ref_point[0]+radius] = enhanced[ref_point[1]-radius:ref_point[1]+radius,ref_point[0]-radius:ref_point[0]+radius]

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == 1:
            cv2.circle(img,(x,y),radius,(0,255,0),-1)
        elif mode == 2:
            cv2.circle(img,(x,y),radius,(0,0,0),-1)
        elif mode ==3:
            img[ref_point[1]-radius:ref_point[1]+radius,ref_point[0]-radius:ref_point[0]+radius] = enhanced[ref_point[1]-radius:ref_point[1]+radius,ref_point[0]-radius:ref_point[0]+radius]
            
#-------------------Open a Window and start drawing--------------------------- 
def Open_window(image, window_name):
    global mode
    global img, enhanced
    
    img = image.copy()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, draw)
    
    while(1):
        image = img
        cv2.imshow(window_name, image)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('g'):
            mode = 1
        elif k== ord('b'):
            mode = 2;
        elif k == ord('z'):
            mode = 3
        elif k == 27:
            break
        
    save = cv2.addWeighted(enhanced,0.2,image,1,0)
    res = window_name.split('_')
    cv2.imwrite(r'out\\' +'mask_'+res[1]+'.png', save*210)   

    cv2.destroyAllWindows()

    return image

def build_video(sequence, method):
    cap = cv2.VideoCapture(0)
    out =cv2.VideoWriter('video_'+method+'.avi',-1,5.0,(1440,1080))
    
    while (cap.isOpened()):
        for image in sequence:
            out.write(np.uint8(image*160))
    
        out.release()
        cap.release()
        break