import pandas as pd
import requests
import json
import numpy as np
import xml.etree.ElementTree as ET

# ++++++++++++++++++++++++
#     Image Parameters
# ++++++++++++++++++++++++
#
# 1)  Entropy
# 2)  Mean
# 3)  Std. Deviation
# 4)  Fractal Dim.
# 5)  Skewness
# 6)  Kurtosis
# 7)  Uniformity
# 8)  Rel. Smoothness
# 9)  T. Contrast
# 10) T. Directionality
# 
"""
This function prepares X matrix, by keeping all image parameters against the 9 different wavelengths.
Each Column in X represents all the image parameters at a specific time t1. 
Dimensions of X: 3686400 X N 
"""
def prepare_x(start_time):
	wavelength_list = [94, 131, 171, 193, 211, 304, 335, 1600, 1700]
	image_dimensions = 64

	req_x = np.zeros((image_dimensions*image_dimensions*10,len(wavelength_list)))
	req_x = list()
	for wave in wavelength_list:
		url = "http://dmlab.cs.gsu.edu/dmlabapi/params/SDO/AIA/64/full/?wave="+str(wave)+"&starttime="+str(start_time)
		response = requests.get(url)
		tree = ET.fromstring(response.content)
		image_param_lst = []
		for cell in tree.findall('cell'):
			image_param_lst.append([param.text for param in cell.find('params')])
		image_param_lst = np.array((image_param_lst)).flatten()
		req_x.append(image_param_lst)
	return np.array(req_x).flatten()

"""
This function is used for preparing Y matrix which contains either 1 or 0 based on event occurence.
Dimensions of Y: 4 X N
"""
def prepare_y(event_list):
	req_y = [0,0,0,0]
	for event in event_list:
		if 'CH' in event.values():
			req_y[0]=1
		if 'FL' in event.values():
			req_y[1]=1
		if 'AR' in event.values():
			req_y[2]=1
		if 'SG' in event.values():
			req_y[3]=1					
	return req_y

"""
This function is used to load image data for the specified time frame and returns X and Y matrices.
"""			
def Load_data(start_time,end_time,N):
	date_list = list(pd.date_range(start=start_time, end=end_time, periods=N+1))
	batch_size = (end_time - start_time)/N
	date_list = list((current_date,current_date+batch_size) for current_date in date_list if current_date!=end_time)
	
	final_y_matrix  = np.zeros((len(date_list),4))
	x = []
	date_format = "%Y-%m-%dT%H:%M:%S"
	for idx,each_date in enumerate(date_list):
		url = "http://isd.dmlab.cs.gsu.edu/api/query/temporal?starttime="+str(each_date[0].strftime(date_format))+"&endtime="+str(each_date[1].strftime(date_format))+"&tablenames=ar,ch,sg&sortby=event_starttime&limit=100&offset=0&predicate=Overlaps"
		response = requests.get(url)
		jData = json.loads(response.content)
		event_types_list = [{list(k.keys())[0]:k[list(k.keys())[0]]} for k in jData['Result']]
		unique_event_types = [dict(y) for y in set(tuple(x.items()) for x in event_types_list)]
		y = np.array((prepare_y(unique_event_types)))
		final_y_matrix[idx]=y.T
		x.append(prepare_x(each_date[0].strftime(date_format)))
	x = np.array(x).T
	x= x.astype(np.float)
	return x,final_y_matrix.T