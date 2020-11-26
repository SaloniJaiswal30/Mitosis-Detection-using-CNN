import cv2
import os
import numpy as np
import csv as csv

def smooth(img):
	dest=cv2.medianBlur(img,7)
	return dest

def process(path,img):
	image=cv2.imread(path+img,1)
	image=smooth(image)
	return image
	
def kmeans(img,name):
	output=[]
	image=img.reshape(img.shape[0]*img.shape[1],3)
	image=np.float32(image)
	nclusters=6
	criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
	attempts=10
	flags=cv2.KMEANS_RANDOM_CENTERS
	compactness,labels,centers=cv2.kmeans(image,nclusters,None,criteria,attempts,flags)
	#print labels
	#print centers
	#print labels.flatten()
	centers = np.uint8(centers)
	res = centers[labels.flatten()]
	res2 = res.reshape((img.shape))
	d=[]
	data=csv.reader(open("../labels/"+name[:-5]+"_mitosis.csv",'r'))
	for j in data:
		d.append([j[0],j[1]])
	e=[]
	data=csv.reader(open("../labels/"+name[:-5]+"_not_mitosis.csv",'r'))
	for j in data:
		e.append([j[0],j[1]])
	params=cv2.SimpleBlobDetector_Params()
	params.filterByArea = True
	params.minArea=250
	params.maxArea=2500
	params.filterByConvexity = False	
	params.minConvexity=0.95	
	params.filterByCircularity = True
	params.minCircularity=0.65
	detector=cv2.SimpleBlobDetector_create(params)
	keypoints=detector.detect(res2)
	maxd=0
	true=[1]
	false=[0]
	for i in range (0,len(keypoints)):
		x=int(keypoints[i].pt[0])
		y=int(keypoints[i].pt[1])
		dm=int(keypoints[i].size)
		maxd=max(maxd,dm)
		#print x,' ',y,' ',dm
		for j in range(0,len(d)):
			if(x>=float(d[j][0])-200 and x<=float(d[j][0])+200 and y>=float(d[j][1])-200 and y<=float(d[j][1])+200):
				output.append(res2[y-dm:y+dm,x-dm:x+dm])
				cv2.imwrite(dest+name[:-5]+str(i)+'true.png', cv2.resize(res2[y-dm:y+dm,x-dm:x+dm],(70,70)))
				d.pop(j)
				with open("testlabels.csv",'a', newline='') as myfile:
					wr=csv.writer(myfile,quoting=csv.QUOTE_ALL)
					wr.writerow(true)
				break
		for j in range(0,len(e)):
			if(x>=float(e[j][0])-200 and x<=float(e[j][0])+200 and y>=float(e[j][1])-200 and y<=float(e[j][1])+200):
				output.append(res2[y-dm:y+dm,x-dm:x+dm])
				cv2.imwrite(dest+name[:-5]+str(i)+'false.png', cv2.resize(res2[y-dm:y+dm,x-dm:x+dm],(70,70)))
				e.pop(j)
				with open("testlabels.csv",'a', newline='') as myfile:
					wr=csv.writer(myfile,quoting=csv.QUOTE_ALL)
					wr.writerow(false)
				break
	im_with_keypoints=cv2.drawKeypoints(res2,keypoints,np.array([]),(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	#cv2.imshow('Keypoints',im_with_keypoints)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	return output
	
def preprocess(path):
	images=[]
	j=0
	print( "Median Blur")
	for i in os.listdir(path):
		print(i,"--",j)        
		images.append(process(path,i))
		images[j]=kmeans(images[j],i)
		j+=1

	

dest='../output2/'
print ("Preprocess")
preprocess('../input2/')
