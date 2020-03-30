#!/usr/bin/env python3
import os
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import xmltodict

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)
    
def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise NotADirectoryError(string)

def grid_segmentation(atlas, debug=False):
    gray = cv2.cvtColor(atlas,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=4)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,0)
    ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    # Marker labelling
    nmarkers, markers, stats, centroids = cv2.connectedComponentsWithStats(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv2.watershed(atlas,markers)
    markers = markers - 1
    if debug:
        for marker in range(1, nmarkers):
            if stats[marker,cv2.CC_STAT_AREA] < 100:
                continue
            marked_at = atlas.copy()
            marked_at[markers == marker] = [255,0,0]
            print(np.mean(gray[markers == marker]))
            left = stats[marker, cv2.CC_STAT_LEFT]; right = left + stats[marker, cv2.CC_STAT_WIDTH]
            top = stats[marker, cv2.CC_STAT_TOP]; bottom = top + stats[marker, cv2.CC_STAT_HEIGHT]
            plt.subplot(1,2,1)
            plt.imshow(marked_at)
            plt.subplot(1,2,2)
            cx = int(centroids[marker,0])
            cy = int(centroids[marker,1])
            if cx < 10 or cy <10: continue
            subimg = atlas[cy-10:cy+10, cx-10:cx+10]
            plt.imshow(subimg)
            plt.show()
            
            # dft = cv2.dft(np.float32(subimg),flags = cv2.DFT_COMPLEX_OUTPUT)
            # dft_shift = np.fft.fftshift(dft)
            
            # magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
            
            # plt.subplot(121),plt.imshow(subimg, cmap = 'gray')
            # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
            # plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
            # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
            # plt.show()
            

    return markers, stats, centroids

def get_transform(src_at, dst_at, debug=False):
    # Get keypoints from both atlases and the descriptors for both using AKAZE
    akaze = cv2.AKAZE_create()
    src_kp, src_desc = akaze.detectAndCompute(src_at, None)
    dst_kp, dst_desc = akaze.detectAndCompute(dst_at, None)

    # Match keypoints from both images.
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    matches = matcher.knnMatch(src_desc, dst_desc, 2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]

    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>10:
        src_pts = np.float32([ src_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ dst_kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w,d = src_at.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        dst_at = cv2.polylines(dst_at,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    if debug:
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
        match_img = cv2.drawMatches(src_at,src_kp,dst_at,dst_kp,good,None,**draw_params)
        plt.imshow(match_img, 'gray'),plt.show()

        dsize = (src_at.shape[0], src_at.shape[1])
        dst = cv2.warpPerspective(src_at, M, dsize, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT);

        # [blend_images]
        alpha = .5
        beta = (1.0 - alpha)
        blended = cv2.addWeighted(dst_at, alpha, dst, beta, 0.0)
        # [blend_images]
        # [display]
        cv2.imshow('dst', blended)
        cv2.waitKey(0)
        # [display]
        cv2.destroyAllWindows()

    return M

def metadata_test(metadata_path, square_path):
    xmls =[]
    for file in os.listdir(metadata_path):
        if file.endswith(".xml"):
            xmls.append(os.path.join(metadata_path, file))
    
    Xs = []
    Ys = []
    i = 0
    for fpath in xmls:
        with open(fpath) as fd:
            meta = xmltodict.parse(fd.read())
        x = float(meta['MicroscopeImage']['microscopeData']['stage']['Position']['X'])
        y = float(meta['MicroscopeImage']['microscopeData']['stage']['Position']['Y'])
        Xs.append(x)
        Ys.append(y)
        plt.text(x, y, str(i), fontsize=9)
        i+=1
    with open(square_path) as fd:
        meta = xmltodict.parse(fd.read())
        x = float(meta['MicroscopeImage']['microscopeData']['stage']['Position']['X'])
        y = float(meta['MicroscopeImage']['microscopeData']['stage']['Position']['Y'])
        plt.scatter(x,y)
    plt.scatter(Xs,Ys)
    plt.show()

if __name__ == "__main__":
    from argparse import ArgumentParser, FileType
    parser = ArgumentParser()
    parser.add_argument('source', type=dir_path,
                        help="path to directory containing the source atlas")
    parser.add_argument('destination', type=dir_path,
                        help="path to directory containing the destination atlas")
    # TODO make this a list of arguments
    parser.add_argument('gridsquare', type=file_path,
                        help="path to file containing the grid square we are trying to locate")
    parser.add_argument("--tform", action="store_true",
                        help="Calculate src to dest transform")
    parser.add_argument("--segment", action="store_true",
                        help="Calculate segmentation")
    parser.add_argument("--xml", action="store_true",
                        help="print xml data")
    
    args = parser.parse_args()
    at1 = cv2.imread(args.source + "/Atlas_1.jpg", cv2.IMREAD_COLOR)
    at2 = cv2.imread(args.destination + "/Atlas_1.jpg", cv2.IMREAD_COLOR)
    if args.segment:
        grid_segmentation(at1)
        grid_segmentation(at2)

    if args.tform:
        transform = get_transform(at1,at2, debug=True)
        print(transform)

    if args.xml:
        metadata_test(args.source, args.gridsquare)
