#coding=utf8

SAMPEL_TIMES = 10

if __name__ == '__main__':
	import sys
	import random
	line_list = []
	for line in sys.stdin:
		line_list.append(line.strip())
	d1 = 0.0
	d2 = 0.0
	for i in range(0, SAMPEL_TIMES):
		uni_set = set()
		uni_num = 0
		bi_set = set()
		bi_num = 0
		for line in random.sample(line_list, min(2000, len(line_list))):
			flist = line.split(" ")
			for x in flist:
				uni_set.add(x)
				uni_num += 1
			for i in range(0, len(flist)-1):
				bi_set.add(flist[i] + "<XXN>" + flist[i + 1])
				bi_num += 1
		d1 += len(uni_set) / float(uni_num)
		d2 += len(bi_set) / float(bi_num)
	print "DIVERSE-1", d1 / SAMPEL_TIMES
	print "DIVERSE-2", d2 / SAMPEL_TIMES
	print "DISTINCT SENTENCES", len(set(line_list)) / float(len(line_list))
