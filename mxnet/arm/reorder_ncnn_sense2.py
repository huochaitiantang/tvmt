# coding=utf-8



import re
import sys
import csv

def get_number( line ):
    spend = re.findall( r"\d+\.?\d*", line)
#print(spend)
    return float(spend[0] ) 

def get_numbers( line ):
    spends = re.findall( r"\d+\.?\d*", line)
    return (spends) 



def getData(path='',data_list=[]):
  with open(path) as f:
    f_csv = csv.reader(f)
    #header = next(f_csv)
    for row in f_csv:
      data_list.append(row)

def write_results_to_csv(path, result_lists):
    with open( path, 'w') as f:
        csv_write = csv.writer(f)
        for line in (result_lists):
            csv_write.writerow(line)

path = './names_conv2d_layers1_19.csv'
raw_names_res50 = []
getData (path , raw_names_res50)
print(raw_names_res50)
data_names_res50 = []
index_integer = 2
for i in range(1, len(raw_names_res50)):
    line = raw_names_res50[i]
    tmp_line = []
    for j in range(index_integer,len(line)):
        tmp_line.append(int(line[j]))
    data_names_res50.append(tmp_line)
print(data_names_res50)

path = '../ncnn_f32_int8_vs_sense_int2-8.csv'
res_ncnn_sense = []
getData (path , res_ncnn_sense)
#print(res_ncnn_sense)

#print(res_ncnn_sense[1][0])
#print(get_numbers(res_ncnn_sense[1][0]))

data_ncnn_sense = []
for i in range(1,len(res_ncnn_sense)):
    tmp = get_numbers(res_ncnn_sense[i][0])
    tmp_data = []
    for ele in (tmp):
        tmp_data.append(int(ele))
    tmp_data[0]=1
    in_filt = tmp_data[3]
    del tmp_data[3]
    tmp_data.insert(1,in_filt)
    data_ncnn_sense.append(tmp_data)
print(data_ncnn_sense)

def match (list1, list2):
    if(len(list1)!=len(list2)):
        return False
    for i in range(len(list1)):
        if(list1[i]!=list2[i]):
            return False
    return True

results_rasp3b = []
for i in range(len(data_names_res50)):
    found =False
    for j in range(len(data_ncnn_sense)):
        if(match(data_names_res50[i],data_ncnn_sense[j])):
            found = True
            break
    if (found):
        results_rasp3b.append(raw_names_res50[i+1] + res_ncnn_sense[j+1][1:] )
#results_rasp3b.append([raw_names_res50[i+1][0]] + data_names_res50[i] + res_ncnn_sense[j+1][1:] )
    else:
        sys.exit("not found!")
        break
#print(raw_names_res50)
results_rasp3b.insert(0,raw_names_res50[0]+res_ncnn_sense[0][1:])

print(results_rasp3b)
for i in range(len(results_rasp3b)):
    del results_rasp3b[i][1]

path = './results_rasp3b.csv'
write_results_to_csv(path, results_rasp3b)

print(results_rasp3b)


