import os,sys
CUR_DIR= os.path.dirname(os.path.abspath(__file__))
dataroot = os.path.join(CUR_DIR,'../../data//wiki')
processed_datafile = f"{dataroot}/wiki_processed2.pt"

dataset = 'wiki'
testlength = 5
vallength = 1
length = 16
split = 0
