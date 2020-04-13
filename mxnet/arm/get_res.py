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

def getRawData(path='' ):
    with open(path, 'r') as f:
        raw_data_lists = f.readlines()
        return raw_data_lists

path='./res_aarch64_left'
raw_data_lists = getRawData(path )
#print(raw_data_lists)

for line in (raw_data_lists):
    if( re.search('model_name', line) ):
        print(line)
    if( re.search('Mean inference', line) ):
        print(line)









