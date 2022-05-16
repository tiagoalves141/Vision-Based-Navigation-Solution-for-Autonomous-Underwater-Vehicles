import cv2 
import tkinter as tk
from tkinter import simpledialog
import pandas as pd
import datetime

periods = []
periods_seconds = []

ROOT = tk.Tk()
ROOT.withdraw()
video = simpledialog.askstring(title='Video Load', prompt='What video do you want to sample?')


capture = cv2.VideoCapture('C:\\Users\\Utilizador\\Downloads\\Menez Gwen\\'+video+'.mov')
fps = capture.get(cv2.CAP_PROP_FPS)
print(fps)

df = pd.read_excel(r'C:\Users\Utilizador\Desktop\Tese\Vídeos\Videos 2.0.xlsx', sheet_name ='Menez Gwen')

flag = 0

for count, i in df['Vídeo'].items():
    if i == video:
        flag = 1
        periods.append(df.iloc[count][1])
    elif pd.isnull(i) and flag == 1:
        periods.append(df.iloc[count][1])
        if not pd.isnull(df['Vídeo'][count+1]):
            break;

for a in periods:
    if a=='-':
        continue
    res = a.split(' ')
    periods_seconds.append(int(res[0].split(':')[0])*60+int(res[0].split(':')[1]))
    periods_seconds.append(int(res[2].split(':')[0])*60+int(res[2].split(':')[1]))
    
print(periods_seconds)

index = 1600

for count, period in enumerate(periods_seconds):
    if count %2 == 0:
        frame = period * fps
        
        end_frame = periods_seconds[count+1]*fps
        
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        success, image = capture.read()
        C=0
        
        while success and frame <= end_frame:
            if C % 10 == 0:
                cv2.imwrite('C:\\Users\\Utilizador\\Desktop\\Tese\\Scripts\\Dataset Creation\\Inputs Menez Gwen\\'+'in_'+str(index)+'.png', image)
                index +=1
            frame += 5
            C += 1
            success, image = capture.read()
            
    else:
        continue


