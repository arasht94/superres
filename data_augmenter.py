import argparse 
import os
import sys

from PIL import Image


def image_augmentation(image_dir):
    print("Augmenting images from {}".format(image_dir))
    transforms = [
        (Image.FLIP_LEFT_RIGHT, 'FLIP_LEFT_RIGHT'),
        (Image.FLIP_TOP_BOTTOM, 'FLIP_TOP_BOTTOM'),
        (Image.ROTATE_90, 'ROTATE_90'),
        (Image.ROTATE_180, 'ROTATE_180'),
        (Image.ROTATE_270, 'ROTATE_270') 
    ]
    
    for file in os.listdir(image_dir):
        name, extension = os.path.splitext(file)
        if extension.lower() != '.jpg' and extension.lower() != '.jpeg':
            continue
        
        image = Image.open(os.path.join(image_dir, file))
        for transform, prefix in transforms:
            new_image = image.transpose(Image.FLIP_LEFT_RIGHT) 
            new_name = prefix + '_' + name + extension
            new_image.save(os.path.join(image_dir, new_name))
    
    print("Done augmenting images!")  
    

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True, 
                        help="Location of image directory to perform augmentation")

    return parser.parse_args(args)
    

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    image_augmentation(args.image_dir)
    
    