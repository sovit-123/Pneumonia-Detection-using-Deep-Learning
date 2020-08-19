import cv2
import os
import pydicom
import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--type', type=str, required=True, 
                    choices=['train', 'test'], help='whether to convert train images or test images')
args = vars(parser.parse_args())

if args['type'] == 'train':
    print('Converting train images from .dcm to .jpg...')
    inputdir = 'input/stage_2_train_images/'
    outdir = 'input/images'
elif args['type'] == 'test':
    print('Converting test images from .dcm to .jpg...')
    inputdir = 'input/stage_2_test_images/'
    outdir = 'input/samples'
os.makedirs(outdir, exist_ok=True)

train_list = [f for f in  os.listdir(inputdir)]

for i, f in tqdm(enumerate(train_list[:]), total=len(train_list)):   
    ds = pydicom.read_file(inputdir + f) # read dicom image
    img = ds.pixel_array # get image array
    # img = cv2.resize(img, (416, 416))
    cv2.imwrite(os.path.join(outdir, f.replace('.dcm','.jpg')), img) # write jpg image