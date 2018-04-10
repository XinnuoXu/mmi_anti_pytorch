#coding=utf8

MAX_UTT_LEN = 50
DATA_DIR = "../../data/"

def one_file(outfile, infile):
	fpout_en = open("../data/" + outfile + ".en", "w")
	fpout_de = open("../data/" + outfile + ".de", "w")
	for line in open(DATA_DIR + "/" + infile):
		flist = line.strip().split(" ")
		if len(flist) > MAX_UTT_LEN:
			continue
		flist = ['<s>'] + flist + ['</s>']
		fpout_en.write(" ".join(flist[:-1]) + "\n")
		fpout_de.write(" ".join(flist[1:]) + "\n")
	fpout_en.close()
	fpout_de.close()

if __name__ == '__main__':
	import sys
	one_file("train", "train.vi")
	one_file("test", "test.vi")
	one_file("valid", "dev.vi")
