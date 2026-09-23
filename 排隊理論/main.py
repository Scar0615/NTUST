# -*- coding: utf-8 -*-
import random

class server:
    def __init__(self, frequency, storage):
        self.frequency = frequency
        self.storage = storage

class task:
    def __init__(self, size, cycle, deadline, process_t, wait_t, RF, label):
        self.size = size
        self.cycle = cycle
        self.deadline = deadline
        self.process_t = process_t
        self.wait_t = wait_t
        self.RF = RF
        self.label = label

random.seed(48763)

"""
#storage 統一 MB為單位
sever1 = sever(1000,512) #先除百萬
sever2 = sever(1000,512)
sever3 = sever(1000,512)
sever4 = sever(1000,512)
sever5 = sever(1000,512)
sever6 = sever(1000,512)
sever7 = sever(1000,512)
sever8 = sever(1000,512)
sever9 = sever(1000,512)
sever10 = sever(1000,512)
sever11 = sever(2000,64*1024)
sever12 = sever(2000,64*1024)
"""

list_s = []
for i in range(10):
    list_s.append(server(1000,512))
list_s.append(server(2000,64*1024))
list_s.append(server(2000,64*1024))
#sever1.frequency = 5
#list_s = [sever1,sever2,sever3,sever4,sever5,sever6,sever7,sever8,sever9,sever10,sever11,sever12]   #list server

result_h = []
result_m = []
result_l = []

o_result_h = []
o_result_m = []
o_result_l = []

list_t = [] #list task
list_offload = []

class_high = []
class_med = []
class_low = []

queue = []
index = []
wait = 10*144/2*(1-11/12) #initial waiting time set the, arrival rate is 10, service rate is set 12(set by ourself)

process_time_h = 0
process_time_m = 0
process_time_l = 0

len_h = 0
len_m = 0
len_l = 0

wait_h = 0
wait_m = 0
wait_l = 0

all_time = 0
#list1.append(sever1)
#print(list1[0].frequency)

#lamda = 10, Z = 12, mu = 12
#task =  (size, cycle, deadline, process_t,wait_t)

task_number = 45

#for i in range(task_number):

for i in range(task_number):    #100 tasks
    list_t.append(task(random.random()*20+10,random.random()*100,random.random()*300000,0,wait,0,-1)) #the size will be [10,30]
    list_t[i].process_t = list_t[i].size/20 #off load

for i in range(task_number):
    if list_t[i].deadline <= 100000:
        list_t[i].label = 0
        class_high.append(list_t[i])
    elif list_t[i].deadline <= 200000 and list_t[i].deadline > 100000:
        list_t[i].label = 1
        class_med.append(list_t[i])
    elif list_t[i].deadline <= 300000 and list_t[i].deadline > 200000:
        list_t[i].label = 2
        class_low.append(list_t[i])

for i in range(len(class_high)):
    class_high[i].wait_t = class_high[i].wait_t * i
    
for i in range(len(class_med)):
    class_med[i].wait_t = class_med[i].wait_t * i + class_high[-1].wait_t + wait
    
for i in range(len(class_low)):
    class_low[i].wait_t = class_low[i].wait_t * i + class_med[-1].wait_t + wait
    
#print(len(class_med))
#print(class_low)

#threshold = 0.8
t1 = 0.8 * class_med[-1].wait_t
t2 = 0.8 * class_low[-1].wait_t



for i in range(len(class_med)):
    if class_med[i].wait_t > t1:
        class_high.append(class_med[i])
        index.append(i)
        

for i in range(len(index)):
    class_med.pop(index[i])
    for i in range(len(index)):
        index[i] = index[i] - 1

index = []

for i in range(len(class_low)):
    if class_low[i].wait_t > t1:
        class_med.append(class_low[i])
        index.append(i)

for i in range(len(index)):
    class_low.pop(index[i])
    for i in range(len(index)):
        index[i] = index[i] - 1

#print(len(class_high))
#print(len(class_med))
#print(len(class_low))

for i in range(len(class_high)):
    queue.append(class_high[i])
for i in range(len(class_med)):
    queue.append(class_med[i])
for i in range(len(class_low)):
    queue.append(class_low[i])

for i in range(len(queue)):
    queue[i].wait_t = wait * i

#for i in range(len(queue)):
#    print(queue[i].wait_t)  #calculate waiting time

for i in range(len(queue)):
    queue[i].process_t = queue[i].process_t + queue[i].wait_t
    
#for i in range(len(queue)):
#    print(queue[i].process_t)  #calculate waiting time

#for i in range(len(queue)):
#    print(queue[i].deadline - queue[i].process_t)
#for i in range(len(queue)):
#    print(queue[i].deadline)
for i in range(len(queue)):
    queue[i].RF = queue[i].size/(queue[i].deadline-queue[i].process_t)

#for i in range(len(queue)):
#    print(queue[i].RF)

for i in range(len(list_s)):
    for j in range(len(queue)):
        if queue[j].RF<=list_s[i].frequency and queue[j].size <= list_s[i].storage:
            list_s[i].storage = list_s[i].storage - queue[j].size
            queue[j].process_t =queue[j].process_t + queue[j].cycle/list_s[i].frequency
            
for i in range(len(queue)):
    all_time += queue[i].process_t
    #print(queue[i].process_t)
for i in range(len(queue)):
    if queue[i].label == 0:
        process_time_h += queue[i].process_t
        wait_h += queue[i].wait_t
        len_h += 1
    elif queue[i].label == 1:
        process_time_m += queue[i].process_t
        wait_m += queue[i].wait_t
        len_m += 1
    elif queue[i].label == 2:
        process_time_l += queue[i].process_t
        wait_l += queue[i].wait_t
        len_l += 1

#print(len_h,len_m,len_l)
#print(process_time_h/len_h,process_time_m/len_m,process_time_l/len_l)
#print(wait_h/len_h,wait_m/len_m,wait_l/len_l)

#print(process_time_h,process_time_m,process_time_l)
#print(int(wait_h),int(wait_m),int(wait_l))
#print(int(process_time_h-wait_h),int(process_time_m-wait_m),int(process_time_l-wait_l))
print(int(process_time_h),int(process_time_m),int(process_time_l))

result_h.append(wait_h)
result_m.append(wait_m)
result_l.append(wait_l)

o_result_h.append(process_time_h - wait_h)
o_result_m.append(process_time_m - wait_m)
o_result_l.append(process_time_l - wait_l)
    

#print(result_h)
#print(result_m)
#print(result_l)

#print(o_result_h)
#print(o_result_m)
#print(o_result_l)
    

#print(all_time)




"""
for i in range(len(class_high)):
    print(class_high[i].wait_t)
print('\n')
for i in range(len(class_med)):
    print(class_med[i].wait_t)
print('\n')
for i in range(len(class_low)):
    print(class_low[i].wait_t)
"""


#for i in range(100):
#    print(list_t[i].process_t)