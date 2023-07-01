import os
import sys
import glob

import nibabel as nb
import PIL.Image
import tqdm
import numpy as np
import subprocess


if os.path.isfile("oasis_jpg.zip") :
    print("file exist! oasis_jpg.zip")
    # unzip file
    subprocess.run(["tar", "-xzvf","oasis_jpg.zip","--skip-old-files"]) 
    
else:


	file_search = glob.glob('oasis/*/aligned_norm.nii.gz')
	print(len(file_search))


	labmap = {
				0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9,
				10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17, 18:18, 19:19,
				20:1, 21:2, 22:3, 23:4, 24:5, 25:6, 26:7, 27:8, 28:9, 29:10,
				30:14, 31:15, 32:16, 33:17, 34:18, 35:19
			}

	def save_jpeg(pth, lab=False):
		img = nb.load(pth).get_fdata()
		if lab:
			img = img.astype(int)
			img[img>35] = 35
			img = np.vectorize(labmap.get)(img)
			img = img / 20
			img = img * 255
		else:
			if img.min() != 0 or img.max() !=0:
				img = img - img.min()
				img = img / img.max()
				img = img * 255
		img = img.astype(np.uint8)
		PIL.Image.fromarray(img).resize((192,192), 3).save(os.path.join('oasis_jpg', pth.split('/')[-2], pth.split('/')[-1].split('.')[0]+'.jpg'))
		
	for pth in tqdm.tqdm(file_search, total=len(file_search)):
		os.makedirs(os.path.join('oasis_jpg', pth.split('/')[-2]), exist_ok=True)
		save_jpeg(pth)
		save_jpeg(pth.replace('norm', 'seg35'), lab=True)

	#create zip folder
	subprocess.run(["tar", "-zcvf","oasis_jpg.zip","oasis_jpg/"]) 
	#tar -zcvf oasis_jpg.zip oasis_jpg/