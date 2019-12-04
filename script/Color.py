import cv2
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import const
filedir='/home/fer/Documentos/imagenescv/junglespeedcards'
filedir2='/home/fer/Documentos/imagenescv/shapes'
def imagelist(path):
    address = []
    name = []
    # r=root, d=directories, f = filesasd
    for r, d, f in os.walk(path):
        for file in f:
            address.append(os.path.join(r,file))
            name.append(file)
    return [address,name]
def skeletonize(in1):
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    skel = np.zeros(in1.shape,np.uint8)
    done = False
    size = np.size(in1)
    toskeletonize=in1
    while( not done):
        eroded = cv2.erode(toskeletonize,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(toskeletonize,temp)
        skel = cv2.bitwise_or(skel,temp)
        toskeletonize = eroded.copy()
        zeros = size - cv2.countNonZero(toskeletonize)
        if zeros==size:
            done = True

    return skel

def gethistogram3d(image,nsegments):
    dofone=np.zeros((nsegments,nsegments,nsegments))
    div=(image.max()/nsegments)+1
    for r in range(len(image)):
        for c in range(len(image[0])):
            [a,b,d]=image[r][c]/div
            dofone[a][b][d]=dofone[a][b][d]+1
    print dofone
    return[]
def getmeancolorUmbrals(image,low,high,nsegments):
    div=(image.max()/nsegments)
    mean=[0,0,0]
    ncolors=0
    for c in range(len(image)):
        for r in range(len(image[0])):
            compare=((image[c][r]/div)*div)
            if (compare[0]<high or \
            compare[1]<high or \
            compare[2]<high)and \
            (compare[0]>low or \
            compare[1]>low or \
            compare[2]>low):
                mean=mean+compare
                ncolors=ncolors+1
    if ncolors==0:
        ncolors=ncolors+1
    return (mean/ncolors)


[allimages,name]=imagelist(filedir)
for img2r,nasd in zip(allimages,name):

    image = cv2.imread(img2r)
    color=getmeancolorUmbrals(image,10,250,10)

    bestcol=-1;
    bestfit=float("inf")
    for cont in range(len(const.ALL_COLORS)):
        error=(np.absolute(color-const.ALL_COLORS[cont]).sum())
        if(error<bestfit):
            bestfit=error
            bestcol=cont
    print(nasd)
    print(const.COLOR_LIST[bestcol])

    margen=100
    umbral_bajo = (color[0]-margen,color[1]-margen,color[2]-margen)
    umbral_alto = (color[0]+margen,color[1]+margen,color[2]+margen)

    mask = cv2.inRange(image, umbral_bajo, umbral_alto)
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(mask,kernel,iterations = 1)
    _, binary1 = cv2.threshold(erosion, 127, 255, cv2.THRESH_BINARY_INV)
    # Calculate Moments

    bestshape=-1;
    bestshapexor=-1;
    bestshapesk=-1;
    bestshapesift=-1;
    bestshapesurf=-1;
    bestfitsk=float("inf")
    bestfit=float("inf")
    bestfitxor=float("inf")
    bestfitsift=float("-inf")
    bestfitsurf=float("-inf")
    width = int(binary1.shape[1])
    height = int(binary1.shape[0])
    dim = (width, height)
    [allshapes,shapen]=imagelist(filedir2)
    skeletoncomparing=skeletonize(binary1)
    for shapedir,namesh in zip(allshapes,shapen):
        shape=cv2.imread(shapedir,cv2.IMREAD_GRAYSCALE)

        this_score = cv2.matchShapes(erosion, shape, 3, 0.0)
        if(this_score<bestfit):
            bestshapeimg=shape
            bestfit=this_score
            bestshape=namesh

        _, binary2 = cv2.threshold(shape, 127, 255, cv2.THRESH_BINARY_INV)
        resized=cv2.resize(binary2,dim,interpolation = cv2.INTER_AREA)
        compared = cv2.bitwise_xor(binary1,resized,mask=None)
        this_score= compared.sum()
        if(this_score<bestfitxor):
            bestshapeimgxor=compared
            bestfitxor=this_score
            bestshapexor=namesh
        skeletontocompare=skeletonize(binary2)
        this_score = cv2.matchShapes(skeletoncomparing, skeletontocompare, 3, 0.0)
        if(this_score<bestfitsk):
            bestshapesk=shape
            bestfitsk=this_score
            bestshapenamesk=namesh

        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(binary1,None)
        kp2, des2 = sift.detectAndCompute(binary2,None)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 10)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1,des2,k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        if(len(good)>bestfitsift):
            bestshapesift=shape
            bestfitsift=len(good)
            bestnamesift=namesh


        surf = cv2.xfeatures2d.SURF_create()
        kp1, des1 = surf.detectAndCompute(binary1,None)
        kp2, des2 = surf.detectAndCompute(binary2,None)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 10)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1,des2,k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        if(len(good)>bestfitsurf):
            bestshapesurf=shape
            bestfitsurf=len(good)
            bestnamesurf=namesh
        # orb = cv2.ORB_create(nfeatures=1500)
        # kpts1, descs1 = orb.detectAndCompute(binary1,None)
        # kpts2, descs2 = orb.detectAndCompute(binary2,None)
        # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # matches = bf.match(descs1, descs2)
        # matches = matches.knnMatch(des1,des2,k=2)
        # good = []
        # for m, n in matches:
        #     if m.distance < 0.7*n.distance:
        #         good.append(m)
        # if(len(good)>bestfitorb):
        #     bestshapeorb=shape
        #     bestfitorb=len(good)
        #     bestshapeorb=namesh



    print("Shapesort:   "+bestshape)
    print("Xor comparison:  "+bestshapexor)
    print("skeletonshape:   "+bestshapenamesk)
    print("shift:   "+bestnamesift)
    print("surf:    "+bestnamesurf)

    moments = cv2.moments(binary1)
    huMoments1 = cv2.HuMoments(moments)
    moments = cv2.moments(binary2)
    huMoments2 = cv2.HuMoments(moments)


    #cv2.rectangle(image,(100,100),(200,200),color,-1)
    cv2.imshow("comparing", erosion)
    cv2.imshow("shapedetection", bestshapeimg)
    #cv2.imshow("results "+img2r, erosion)
    cv2.imshow("xor", bestshapeimgxor)
    cv2.imshow("shapeskeleton", bestshapesk)
    cv2.imshow("SIFT",bestshapesift )
    cv2.imshow("SURF",bestshapesurf )
    cv2.waitKey()
