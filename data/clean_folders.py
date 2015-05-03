from PIL import Image
import os
import glob

input_folders = ['AC4-input','AC3-input']
label_folders = ['AC4-labels','AC3-labels']
folders = input_folders + label_folders

def sort_key(input_file):
    return int(input_file.split('.')[0].split('_')[-1])

for folder in folders:
    os.chdir(folder)
    try:
        files = sorted(glob.glob('*.tif'),key=sort_key)
    except:
        files = sorted(glob.glob('*.tif'))
    i = 0
    for filename in files:
        img = Image.open(filename)
        for a in range(99999):
            flag = False
            try:
                img.seek(a)
                img.save(folder +  '_%s.tif'%(i,))
            except EOFError:
                flag = True
                break
            if flag == False:
                i += 1
        if a>1:
            os.remove(filename)

    os.chdir('../')

