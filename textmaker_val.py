from os import listdir
from os.path import isfile, join
from PIL import Image


mypath = "/u/misser11/InSilicoTem_Python_Version/SSD/validation/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(onlyfiles)
files = open('cvsfile.csv','w')
for i in onlyfiles:

    with Image.open(mypath+i) as img:
        width, height = img.size

        files.write('%s,%s,%s,%s,%s,%s\n' %(i,1,5,width-5,5,height-5) )
        img.close()
files.close()
