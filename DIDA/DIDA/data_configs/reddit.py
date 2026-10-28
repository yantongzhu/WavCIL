import os,sys
CUR_DIR= os.path.dirname(os.path.abspath(__file__))
dataroot = os.path.join(CUR_DIR,'../../data//reddit')
processed_datafile = f"{dataroot}/reddit_processed2.pt"

dataset = 'reddit'
testlength = 5
vallength = 1
length = 16
split = 0
