# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:11:57 2019

@author: rafme
"""

import os
import sys
import numpy as np
import cv2
import time
import glob
import matplotlib
from matplotlib import pyplot as plt
import easygui
import seaborn as sns
from scipy.optimize import curve_fit
from matplotlib import cm
import csv
from tqdm import tqdm


sns.set_context("talk", font_scale=2, rc={"lines.linewidth": 2})
#sns.set_style("white")
sns.set_style("ticks")
sns.set_palette(sns.color_palette("GnBu_d", 4))

plt.close('all')
cv2.destroyAllWindows()

'''Function that defines the effects of clicking with the mouse
when drawing a line'''
def on_mouse(event, x, y, flags, params):
    #The scaled image and the scale value are passed as params
    #Params[0] is the image
    #Params[1] is the scaling IN FULL PERCENTAGE (not decimal)
    im = params[0]
    image = im.copy()
    s = params[1]
    global line #Coordinates of the line
    global lineFull #Coordinates of the line if the image was scaled down
    global btn_down
    if event == cv2.EVENT_LBUTTONDOWN:
        line = [] #If we press the button, we reinitialize the line
        btn_down = True
        print('Start Mouse Position: '+str(x)+', '+str(y))
        sbox = [x, y]
        line.append(sbox) 
        cv2.imshow("Calibration", image)
    elif event == cv2.EVENT_MOUSEMOVE and btn_down:
        #this is just for line visualization
        cv2.line(image, (line[0][0],line[0][1]), (x, y), (256,0,0), 3)
        cv2.imshow("Calibration", image)
    elif event == cv2.EVENT_LBUTTONUP:
        #if you release the button, finish the line
        btn_down = False
        print('End Mouse Position: '+str(x)+', '+str(y))
        ebox = [x, y]
        line.append(ebox)
        cv2.line(image, (line[0][0],line[0][1]), (x, y), (256,0,0), 3)
        cv2.imshow("Calibration", image)
        lineFull = [[int(line[i][j]/(s/100)) for j in [0,1]] for i in [0,1]]

###line has four components:
#[0][0] is the x-coordinate of the first point
#[0][1] is the y-coordinate of the first point
#[1][0] is the x-coordinate of the second point
#[1][1] is the y-coordinate of the second point


def calculateError(reference, image):
    
    pixelError = [np.abs(int(reference[i][j])-int(image[i][j])) for i in range(len(image)) for j in range(len(image[0]))]
    sumError = 0
    for i in range(0,len(pixelError)):
        sumError += pixelError[i]

    return sumError

def calculateSpeed(x,y,dt,skip=1):
    
    r1 = list()
    r2 = list()
    v1 = list()
    v2 = list()
    v = list()
    for i in range(0,len(x),skip):
        r1.append(x[i])
        r2.append(y[i])
    
    for i in range(1,len(r1)-1):
        v1.append((r1[i+1]-r1[i-1])/((2*skip)*dt))
        v2.append((r2[i+1]-r2[i-1])/((2*skip)*dt))
        
    for i in range(len(v1)):
        v.append(v1[i]**2 + v2[i]**2)
    '''THE SQURE ROOT SHOULD BE MADE AT THE END, OTHERWISE THERE IS
    TOO MUCH ERROR'''
    return v

def linearFit(x,a):
    return a*x

# Set up tracker.
# Instead of MIL, you can also use
# BOOSTING, KCF, TLD, MEDIANFLOW or GOTURN
     
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
tracker_type = tracker_types[0]

if tracker_type == tracker_types[0]:
    tracker = cv2.TrackerBoosting_create()
elif tracker_type == tracker_types[1]:
    tracker = cv2.TrackerMIL_create()
elif tracker_type == tracker_types[2]:
    tracker = cv2.TrackerKCF_create()
elif tracker_type == tracker_types[3]:
    tracker = cv2.TrackerTLD_create()
elif tracker_type == tracker_types[4]:
    tracker = cv2.TrackerMedianFlow_create()
elif tracker_type == tracker_types[5]:
    tracker = cv2.TrackerGOTURN_create()

dn = os.path.dirname(os.path.realpath(__file__))

if 'initialPath' in locals():
    default = initialPath
else:
    default = dn
    
fileName = easygui.fileopenbox(default=default)
initialPath = fileName.split(fileName.split('\\')[-1])[0]

# Read video
video = cv2.VideoCapture(fileName)

length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = round(video.get(cv2.CAP_PROP_FPS))
seconds = length/fps

f = 1 #Scaling factor of the video


currentDir = fileName.split(fileName.split('\\')[-1])[0]
file = fileName.split('\\')[-1].split('.')[0]
saveDir = currentDir+'\\'+file+'\\'

if not os.path.exists(currentDir+'\\'+file):
    os.makedirs(currentDir+'\\'+file)

newVideoName = file+'_TRACKING_'+tracker_type+'.avi'
newVideoNameSmall = file+'_TRACKING_'+tracker_type+'_bbox.avi'
newVideo = currentDir+'\\'+file+'\\'+newVideoName
newVideoSmall = currentDir+'\\'+file+'\\'+newVideoNameSmall
 
# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()
 
# Read first frame.
ok, initialFrame = video.read()
#ok, initialFrame = video.read()
#ok, initialFrame = video.read()
#ok, initialFrame = video.read()
#ok, initialFrame = video.read()
#ok, initialFrame = video.read()
#ok, initialFrame = video.read()
#ok, initialFrame = video.read()
#ok, initialFrame = video.read()
#ok, initialFrame = video.read()
#ok, initialFrame = video.read()
#ok, initialFrame = video.read()
#ok, initialFrame = video.read()
#ok, initialFrame = video.read()
#ok, initialFrame = video.read()

if not ok:
    print('Cannot read video file')
    sys.exit()
    

'''
#####################
Recreate tracking

It reads the bounding box file, the calibration file and the tracking in px
If one of them is missing or is unreadable, recreateTracking will be false

This allows to recreate the tracking changing any of the display options
(scale bar, text on top, etc) without having to do the tracking again
#####################
'''
recreateTracking = True

if recreateTracking:
    try:
        recreatebbox = list()
        with open(saveDir+'boundingBox.txt') as ff:
            reader = csv.reader(ff, delimiter='\t')
            recreateFPS = float(next(reader)[1])
            for row in reader:
                recreatebbox.append((int(row[0]),int(row[1]),int(row[2]),int(row[3])))
    except:
        recreateTracking = False
        print('RecreateTracking set to False because bounding box file could not be read')
    
'''
#####################
Calibration code
#####################
'''
'''First the video is calibrated using the squared paper sheet below the petri
Each square is 4 mm
The software will ask the user to draw a line in one square to calibrate the video'''

calibrate = True

forceCalibration = False

#Check if calibration file exists
if os.path.exists(saveDir+'calibration.txt') and not forceCalibration:
    try:
        with open(saveDir+'calibration.txt') as ff:
            ff.readline()
            scale = float(ff.readline())
        calibrate = False
        print('Calibration read from file')
        print('Scale: %.3f' % (scale))
    except:
        print('Reading calibration failed')
        calibrate = True
        recreateTracking = False
        print('RecreateTracking set to False because reading calibration failed')
        
if calibrate or forceCalibration:
    if forceCalibration: 
        print('Forcing calibration')
        recreateTracking = False
        print('RecreateTracking set to False because calibration was forced')
    else:
        print('Performing calibration')
    cv2.imshow("Calibration",initialFrame)
    cv2.setMouseCallback("Calibration", on_mouse, [initialFrame,f*100])
    
    while(True):
    
        cv2.setMouseCallback("Calibration", on_mouse, [initialFrame,f*100])
    
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    
    cv2.destroyAllWindows()
    
    try:
        if len(lineFull) == 2: #If a line was drawn
            #Line is resized to be drawn in the resized edge frame
            lineResized = [[int(lineFull[i][j]*(f/100)) for j in [0,1]] for i in [0,1]]
    except NameError:
        print('No calibration was done')
        raise NameError('lineFull was not defined')
    
    
    ###line has four components:
    #[0][0] is the x-coordinate of the first point
    #[0][1] is the y-coordinate of the first point
    #[1][0] is the x-coordinate of the second point
    #[1][1] is the y-coordinate of the second point
    
    
    lineLengthPx = np.sqrt((lineFull[0][0]-lineFull[1][0])**2 + (lineFull[0][1]-lineFull[1][1])**2)
    scale = lineLengthPx/4 #In px/mm, knowing that each square is 4 mm
    print('Calibration done')
    print('Scale: %.3f' % (scale))

    
    try:
        with open(saveDir+'\\calibration.txt', 'w') as ff: 
            ff.write('Scale (px/um)\n')
            ff.write("%.3f\n" % (scale))
            ff.close()
        print('Calibration saved in file')
    except:
        print('Could not save calibration')


'''
#####################
End of calibration code
#####################
'''


'''
#####################
Tracking code
#####################
'''

frameResized = cv2.resize(initialFrame,(0,0),fx=f,fy=f)    
    
# Define an initial bounding box
if recreateTracking:
    bbox = recreatebbox[0]
else:
    bbox = (287, 23, 86, 320) #Random
    
bboxList = list()

###bbox has four components:
#The first one is the x-coordinate of the upper left corner
#The second one is the y-coordinate of the upper left corner
#The third one is the x-coordinate of the lower right corner
#The four one is the y-coordinate of the lower right corner

if recreateTracking:
    try:
        timeList = list()
        recreateCenters = list()
        centers = list()
        count = 1 #First frame is already read
        realcount = 1 #First frame is already read
        with open(saveDir+'trackingCV2pixels.txt') as ff:
            reader = csv.reader(ff, delimiter='\t')
            header = next(reader) #header
            for row in reader:
                timeList.append(float(row[0]))
                recreateCenters.append((int(row[1]),int(row[2])))
        centers.append(recreateCenters[0])
    except:
        recreateTracking = False
        print('RecreateTracking set to False because tracking file could not be read')

if not recreateTracking:
    bbox = cv2.selectROI(frameResized, False)
    
    bbox = tuple([int(z/f) for z in bbox]) #Resizes box to original file size
    bboxList.append(bbox)
    initialBbox = bbox
    initialCrop = initialFrame[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
    initialCrop = cv2.cvtColor(initialCrop, cv2.COLOR_BGR2GRAY)
    initialFrameResizedGray = cv2.cvtColor(frameResized, cv2.COLOR_BGR2GRAY)
    
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(initialFrame, bbox)
    count = 1 #First frame is already read
    realcount = 1 #If there is skip frames
    timeList = list([0])
    centers = list()
    centers.append((int(initialBbox[0] + initialBbox[2]/2.),int(initialBbox[1] + initialBbox[3]/2.)))
    errorCrop = list([0])
    trackingError = list([0])


#Display options
displayFPS = True
displayTime = True
displayTracker = False
displayBox = True
displayTracking = True
displayScaleBar = True
displayScaleBarText = True
displayTrackedObject = True
displayVideo = False
generalOffset = 0 #This is to apply a general offset to all the details, to be moved down

#Skip frames
skipFrames = False
skip = 2
newfps = fps

#Necessary to write videos
if skipFrames and not recreateTracking:
    #New fps
    newfps = fps/skip
    #New video name
    newVideo = newVideo.split('.avi')[0] + '_skip'+str(skip)+'.avi'
    newVideoSmall = newVideoSmall.split('.avi')[0] + '_skip'+str(skip)+'.avi'
elif recreateTracking:
    #If the tracking is recreated, skip frames doesn't manually. It is set
    #to that corresponding of the FPS read in the file
    skip = int(fps/recreateFPS)
    skipFrames = True
    newfps = recreateFPS
    if recreateFPS != fps:
        newVideo = newVideo.split('.avi')[0] + '_skip'+str(skip)+'.avi'
        newVideoSmall = newVideoSmall.split('.avi')[0] + '_skip'+str(skip)+'.avi'
        
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(newVideo, fourcc, newfps, (int(width),int(height)))

outSmall = cv2.VideoWriter(newVideoSmall, fourcc, newfps, (int(bbox[2]),int(bbox[3])))

#Get colormap for trajectory
cmap = sns.color_palette("RdYlBu",int(seconds*newfps))

pbar = tqdm(total=length-1) #For profress bar

while True:
    # Read a new frame
    ok, frame = video.read()
    count += 1
    pbar.update()
    
    
    if not ok:
        break
    
    if skipFrames and count%skip != 0:
        #If frame is to be skipped, continue with the loop
        continue
    
    realcount += 1
    if f != 1:
        frameResized = cv2.resize(frame,(0,0),fx=f,fy=f)        

    # Update tracker
    if recreateTracking:
        ok = True
        bbox = recreatebbox[realcount-1]
        centers.append(recreateCenters[realcount-1])
        imCrop = frame.copy()[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
        imCropGray = cv2.cvtColor(imCrop, cv2.COLOR_BGR2GRAY)
    else:
        ok, bbox = tracker.update(frame)


    #Resize bbox
    bboxScaled = tuple(int(z*f) for z in bbox)
    bboxList.append(bbox)

            
    if not recreateTracking:
        #This is not useful if the tracking is recreated. No txt files are saved.
        cropMovementIndex = frame[int(initialBbox[1]):int(initialBbox[1]+initialBbox[3]), 
                                  int(initialBbox[0]):int(initialBbox[0]+initialBbox[2])]
        cropMovementIndex = cv2.cvtColor(cropMovementIndex, cv2.COLOR_BGR2GRAY)
        
        #Copy cropped image to analyze the error
        imCrop = frame.copy()[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
        imCropGray = cv2.cvtColor(imCrop, cv2.COLOR_BGR2GRAY)
        errorCrop.append(calculateError(initialCrop,imCropGray))
        
        centers.append((int(bbox[0] + bbox[2]/2.),int(bbox[1] + bbox[3]/2.)))
        
        trackingError.append(calculateError(initialCrop,cropMovementIndex))
            
    if f != 1:
        imCropResized = frameResized.copy()[int(bboxScaled[1]):int(bboxScaled[1]+bboxScaled[3]), 
                                         int(bboxScaled[0]):int(bboxScaled[0]+bboxScaled[2])]
    else:
        imCropResized = imCrop
 

    
    # Draw bounding box
    if ok:
        if displayBox:
            if f != 1:
                p1Scaled = (int(bboxScaled[0]), int(bboxScaled[1]))
                p2Scaled = (int(bboxScaled[0] + bboxScaled[2]), int(bboxScaled[1] + bboxScaled[3]))
                cv2.rectangle(frameResized, p1Scaled, p2Scaled, (255,0,0),thickness=2)
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0),thickness=2)
#        color = (int(cmap[count][2]*255),int(cmap[count][1]*255),int(cmap[count][0]*255))
#        cv2.circle(frameResized,radius = 2,center=(int(centers[-1][0]),int(centers[-1][1])),color=color,thickness = -1)
        if len(centers) > 1 and displayTracking:
            count2 = 0
            for point1, point2 in zip(centers, centers[1:]): 
                color = (int(cmap[count2][2]*255),int(cmap[count2][1]*255),int(cmap[count2][0]*255))
                cv2.line(frame, point1, point2, color, 3) 
                if f != 1:
                    p1Scaled = tuple([int(p1*f) for p1 in point1])
                    p2Scaled = tuple([int(p2*f) for p2 in point2])
                    cv2.line(frameResized, p1Scaled, p2Scaled, color, 3)
                count2 += 1

    # Display tracker type on frame
    if displayTracker:
        if f != 1:
            cv2.putText(frameResized, tracker_type + " Tracker", (int(f*width*0.15),int(f*height*0.05+generalOffset*f)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
        cv2.putText(frame, tracker_type + " Tracker", (int(width*0.15),int(height*0.05)+generalOffset), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
    
    # Display FPS on frame
    if displayFPS:
        if f != 1:
            cv2.putText(frameResized, "FPS: " + str(newfps), (int(f*width*0.15),int(f*height*0.05+generalOffset*f)+30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
        cv2.putText(frame, "FPS: " + str(newfps), (int(width*0.15),int(height*0.05)+30+generalOffset), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

    # Display elapsed time
    elapsed = realcount/newfps
    if displayTime:
        if f != 1:
            cv2.putText(frameResized, "Time: " + "%.2f s" % elapsed, (int(f*width*0.15),int(f*height*0.05+generalOffset*f)+60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);        
        cv2.putText(frame, "Time: " + "%.2f s" % elapsed, (int(width*0.15),int(height*0.05)+60+generalOffset), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);        
    timeList.append(elapsed)
    
    # Display scale bar
    if displayScaleBar:
        p1 = (int(width*0.7),int(height*0.07)+generalOffset)
        p2 = (int(width*0.7+10*scale),int(height*0.075)+generalOffset)
        cv2.rectangle(frame, p1,p2, (255,255,255), -1)
        if f != 1:
            p1Scaled = (int(p1[0]*f),int(p1[1]*f+generalOffset*f))
            p2Scaled = (int(p2[0]*f),int(p2[1]*f+generalOffset*f))
            cv2.rectangle(frameResized, p1Scaled,p2Scaled, (255,255,255), -1)
        if displayScaleBarText:
            if f != 1:
                cv2.putText(frameResized,"10 mm",(int(f*width*0.7),int(f*height*0.06)+generalOffset),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
            cv2.putText(frame,"10 mm",(int(width*0.72),int(height*0.06)+generalOffset),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
    
    # Display tracked object
    if displayTrackedObject:
#        offSetRight = int(width*0.55)
        offSetRight = 0
        p1 = int(height*0.05)+100+generalOffset
        p2 = int(width*0.15) + offSetRight
        sizeIncrease = 2
        dim = (int(imCrop.shape[1] * sizeIncrease),int(imCrop.shape[0] * sizeIncrease))
        imCropIncreased = cv2.resize(imCrop, dim, interpolation = cv2.INTER_AREA)
        frame[p1:p1+imCropIncreased.shape[0],p2:p2+imCropIncreased.shape[1]] = imCropIncreased 
        cv2.rectangle(frame, (p2,p1),(p2+imCropIncreased.shape[1],p1+imCropIncreased.shape[0]), (255,255,255), 2)
        if f != 1:
            p1Scaled = int(f*height*0.05+generalOffset*f)+100
            p2Scaled = int(f*width*0.15 + offSetRight*f)
            dim = (int(imCropResized.shape[1] * sizeIncrease),int(imCropResized.shape[0] * sizeIncrease))
            imCropResizedIncreased = cv2.resize(imCrop, dim, interpolation = cv2.INTER_AREA)
            frameResized[p1Scaled:p1Scaled+imCropResizedIncreased.shape[0],
                         p2Scaled:p2Scaled+imCropResizedIncreased.shape[1]] = imCropResizedIncreased 
            cv2.rectangle(frameResized, (p2Scaled,p1Scaled),(p2Scaled+imCropResizedIncreased.shape[1],
                          p1Scaled+imCropResizedIncreased.shape[0]), (255,255,255), 2)            
        
    
    
    # Display result
    if displayVideo:
        if f != 1:
            cv2.imshow("Tracking", frameResized)
        else:
            cv2.imshow("Tracking", frame)        
    out.write(frame)
    outSmall.write(imCrop) 
 
    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break

pbar.close() #Close progress bar
cv2.destroyAllWindows()
out.release()
outSmall.release()
video.release()

if not recreateTracking:
    #Only do plots if the tracking is recreated
        
    #Write bounding boxes
    with open(currentDir+'\\'+file+'\\'+'boundingBox.txt', 'w')  as ff:
        ff.write('FPS: \t%.2f\n' % (newfps))
        for i in range(len(bboxList)):
            ff.write("%.f\t%.f\t%.f\t%.f\n" % (bboxList[i][0],bboxList[i][1],bboxList[i][2],bboxList[i][3]))
    
    
    initialCenter = (int(initialBbox[0] + initialBbox[2]/2.),int(initialBbox[1] + initialBbox[3]/2.))
    initialCenterx = initialCenter[0]/scale
    initialCentery = initialCenter[1]/scale
    
    centerx = [centers[i][0]/scale for i in range(len(centers))]
    centery = [centers[i][1]/scale for i in range(len(centers))]
    center = [np.sqrt((centerx[i]-initialCenterx)**2 + (centery[i]-initialCentery)**2) for i in range(len(centerx))]
    
    times = [i/fps for i in range(len(center))]
    
    popt, pcov = curve_fit(linearFit, timeList, center)
    
    fig = plt.figure(figsize=(12,8))
    plt.plot(timeList,center, label = 'Data')
    plt.plot(timeList, popt[0]*np.asarray(timeList),'r-',label = 'Fitting\nSpeed = %.2f $\mu$m/s' % (popt[0]*1000))
    #plt.text(s='Speed = %.2f $\mu$m/s' % (popt[0]*1000))
    plt.legend(fontsize = 20)
    plt.xlabel('Time [s]')
    plt.ylabel('Distance [mm]')
    plt.tight_layout()
    fig.savefig(currentDir+'\\'+file+'\\'+'motion.png', dpi = 500)
    fig.savefig(currentDir+'\\'+file+'\\'+'motion.svg',format='svg',dpi=1200)   
    
    with open(currentDir+'\\'+file+'\\'+'motion.txt', 'w') as ff:
        ff.write('Time (s)\tDistance (mm)\n')
        for i in range(len(timeList)):
            ff.write("%.3f\t%.6f\n" % (timeList[i],center[i]))
    
    with open(currentDir+'\\'+file+'\\'+'speedFitting.txt', 'w') as ff:
        ff.write('Fitted speed (um/s)\n')
        ff.write('%.6f' % (popt[0]*1000))
        
    
    fig2 = plt.figure(figsize=(12,8))
    plt.plot(timeList,errorCrop)
    plt.xlabel('Time [s]')
    plt.ylabel('Tracking difference [a.u.]')
    plt.tight_layout()
    fig2.savefig(currentDir+'\\'+file+'\\'+'trackingDiff.png', dpi = 500)
    fig2.savefig(currentDir+'\\'+file+'\\'+'trackingDiff.svg',format='svg',dpi=1200)   
    
    with open(currentDir+'\\'+file+'\\'+'trackingDiff.txt', 'w') as ff:
        ff.write('Time (s)\tTracking Difference (a.u.)\n')
        for i in range(len(timeList)):
            ff.write("%.3f\t%.6f\n" % (timeList[i],errorCrop[i]))
    
    
    
    '''In CV2, the coordinates are:
        
    0/0---X--->
     |
     |
     Y
     |
     |
     v
    
    Instead of being (like in plots):
    
     ^
     |
     |
     Y
     |
     |
    0/0---X--->
    
    Therefore, to plot the trajectory we need to change the sign of the y coordinate
    '''
    
    
    centerx_norm = [centerx[i]-initialCenterx for i in range(len(centerx))]
    centery_norm = [-(centery[i]-initialCentery) for i in range(len(centery))]
    
    xmax = max(centerx_norm)
    xmin = min(centerx_norm)
    ymax = max(centery_norm)
    ymin = min(centery_norm)
    lims = (min((xmin,ymin)),max((xmax,ymax)))
    
    fig3 = plt.figure(figsize=(12,8))
    plt.plot(centerx_norm,centery_norm)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel('X position [mm]')
    plt.ylabel('Y position [mm]')
    plt.tight_layout()
    fig3.savefig(currentDir+'\\'+file+'\\'+'tracking.png', dpi = 500)
    fig3.savefig(currentDir+'\\'+file+'\\'+'tracking.svg',format='svg',dpi=1200)   
    
    #Write tracking in pixels
    with open(currentDir+'\\'+file+'\\'+'trackingCV2pixels.txt', 'w')  as ff:
        ff.write('Time (s)\tX (opencv px)\tY (opencv px)\n')
        for i in range(len(timeList)):
            ff.write("%.3f\t%.f\t%.f\n" % (timeList[i],centers[i][0],centers[i][1]))
    
    #Write tracking in mm
    with open(currentDir+'\\'+file+'\\'+'trackingPlotMm_norm.txt', 'w')  as ff:
        ff.write('Time (s)\tX (mm)\tY (mm)\n')
        for i in range(len(timeList)):
            ff.write("%.3f\t%.6f\t%.6f\n" % (timeList[i],centerx_norm[i],centery_norm[i]))
    
    
    
