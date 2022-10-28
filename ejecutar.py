import os
import sys
import time

#os.system('ping 10.236.52.82')
os.system('python servidor_broadcast.py &')
time.sleep(3)

os.system('python detect.py --source 0 --weights best.pt')
#os.system('python detect1.py --weights Detection/exp4/weights/best.pt > data.txt')
#os.system('ping 10.236.52.82 | tee data.txt')
