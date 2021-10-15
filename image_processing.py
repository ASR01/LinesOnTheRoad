import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./test_images/solidWhiteRight.jpg')
img_show = img[:,:,::-1]
#print(img.shape)
# To do's




def filter_colours(img):
    # This filter will filter out the ccolor oiut of white.
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    low_filter = (0,200,0)
    high_filter = (180, 255, 255)

    colour_mask = cv2.inRange(img_hsv, low_filter, high_filter)
    colour_mask = cv2.bitwise_not(colour_mask)
    colour_mask = np.stack((colour_mask,) * 3, axis=-1)
    
#    img_hsv[colour_mask != 0] = [0,0,0]
    img_bgr = cv2.bitwise_and(img, colour_mask)
    #img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    
    return img_bgr
    return colour_mask
    
def define_ROI(img, corners):

    mask = np.zeros_like(img)
    white = (255,255,255)

    cv2.fillPoly(mask, pts = [corners], color = white)
    
    
    img = cv2.bitwise_and(img, mask)
    
    return img

def blurring(img, kernel):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img, (kernel, kernel), 0)
    return blurred

def edging(img, low, high):
    edged = cv2.Canny(img, low, high, None, 3)
    return edged


def detect_lines(img, angle):
    limit = np.tan(angle)
    img_c = img.copy()
    lines = cv2.HoughLinesP(img_c, 0.75, np.pi / 180 , threshold=10, minLineLength=10, maxLineGap=30)
    img_c = cv2.cvtColor(img_c, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for i in range(len(lines)):
            x0, y0, x1, y1 = lines[i][0]
            if x1 != x0:
                tangent = ((y1-y0)/(x1-x0))
            if abs(tangent) > limit:
                cv2.line(img_c, (x0, y0), (x1, y1), (255,0,0), 3, cv2.LINE_AA)
    rl, ll =checking_lines(lines, angle)
    
    return img_c, rl, ll

def checking_lines(lines, angle):
    print(len(lines))
    left_lines = []
    right_lines = []
    limit = np.tan(angle)
            
    for i in range(len(lines)):
        x0, y0, x1, y1 = lines[i][0]
        if x1 != x0:
            tangent = ((y1-y0)/(x1-x0))
        if abs(tangent) > limit:
            if tangent > 0:
                left_lines.append([x0, y0, x1, y1])
            elif tangent < 0:
                right_lines.append([x0, y0, x1, y1])
        print(lines[i][0], tangent, limit)
    #print('right', right_lines)
    #print('left',left_lines)
     
            
    return right_lines, left_lines

def unifying_lines(lin):
       
    tangent = []
    constant = []
    weights  = [] 
    
    for i in range(len(lin)):
        print(i, lin[i])
        for x0, y0, x1, y1 in lin[i]:
            if x1 !=x0:
                tan = (y1-y0)/(x1-x0)
                b = y1 - tangent*x1
                length = np.sqrt((y1-y0)**2+(x1-x0)**2)
                tangent.appen(tan)
                constant.append(b)
                length.append(length)
                
    
    # add more weight to longer lines
    if len(weights) > 0:
        lane  = np.dot(weights,lines ) /np.sum(weights)  if len(weights) >0 else None
    else:
        lane = None
    return lane
    

def write_lines():
    pass


def process_img(img, type):
    
    # Cropping Params
    row = img.shape[0]
    col = img.shape[1]
    angle = 20*(np.pi/180) #(Tangent in degrees)


    up_left = [col* 0.35, row * 0.30]
    up_right = [col* 0.65, row * 0.3]
    bottom_left = [col * 0, row * 1]
    bottom_right = [col* 1,row * 1]
    corners = np.array([up_left, up_right, bottom_right, bottom_left], dtype = 'int32')
    # Blurring Pars
    blur_kernel = 15
    sigma = 0
    #Edging pars
    low = 50
    high = 150

    #Img_processing

    filtered = filter_colours(img)
    blurred = blurring(filtered, blur_kernel)
    edged = edging(blurred, low, high)
    detection_ROI = define_ROI(edged, corners)
    lines_detected, rl, ll = detect_lines(detection_ROI, angle)
    #group the lines
    #print('right lines:',rl)
    #print(rl[0])
    #right = unifying_lines(rl)
    #left = unifying_lines(ll)
    #print(left, right)





    # display for test
    if type == 0:
        fig = plt.figure(figsize = (15,10))
        fig.add_subplot(2,3,1)
        plt.imshow(img_show)
        fig.add_subplot(2,3,2)
        plt.title('Color Filter')
        plt.imshow(filtered)
        fig.add_subplot(2,3,3)
        plt.title('Blurred')
        plt.imshow(blurred, cmap = 'gray')
        fig.add_subplot(2,3,4)
        plt.title('Edge')
        plt.imshow(edged, cmap = 'gray')
        fig.add_subplot(2,3,5)
        plt.title('Region of Interest')
        plt.imshow(detection_ROI, cmap = 'gray')
        fig.add_subplot(2,3,6)
        plt.title('Lines')
        plt.imshow(lines_detected)
        plt.show()

    lines = checking_lines(lines, angle)



process_img(img, 0)

