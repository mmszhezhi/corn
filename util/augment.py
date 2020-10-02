import imgaug as ia
import sys,os,glob
from imgaug.parameters import StochasticParameter,handle_continuous_param
import numpy as np
from PIL import Image
import imgaug.augmenters as iaa


class FixRotate(StochasticParameter):
    def __init__(self):
        super(FixRotate, self).__init__()

    def _draw_samples(self,x,y):
        return np.array([0,0,180,270])

    def __repr__(self):
        return self.__str__()
    def __str__(self):
        return "Normal(loc=%s, scale=%s)" % ( 0.3)


def gen_batches_vh(files,scale):

    skip = 0
    l = len(files)
    for img in files:
        try:
            skip += 1
            print(f" {skip} of {l}")
            raw_img = Image.open(img)
            new_size = (int(raw_img.width / scale), int(raw_img.height / scale))
            raw_img.thumbnail(new_size)
            raw_img = np.array(raw_img)
            if raw_img.shape[0] / raw_img.shape[1] > 1:
                raw_img = raw_img.transpose((1,0,2))
            hfliper = iaa.HorizontalFlip()
            vfliper = iaa.VerticalFlip()
            hfliped = hfliper(images=[raw_img])
            vfliped = vfliper(images=[raw_img,hfliped[0]])
            yield (img.split("\\")[-1],[raw_img,hfliped[0]] + vfliped)
        except Exception as e:
            print(repr(e))

def gen_batches_h(files,scale):

    skip = 0
    l = len(files)
    for img in files:
        try:
            skip +=1
            print(f" {skip} of {l}")
            raw_img = Image.open(img)
            new_size = (int(raw_img.width / scale), int(raw_img.height / scale))
            raw_img.thumbnail(new_size)
            raw_img = np.array(raw_img)
            if raw_img.shape[0] / raw_img.shape[1] < 1:
                raw_img.transpose((1,0,2))
            hfliper = iaa.HorizontalFlip()
            hfliped = hfliper(images=[raw_img])
            yield (img.split("\\")[-1],[raw_img,hfliped[0]])
        except Exception as e:
            print(repr(e))


if __name__ == '__main__':
    scale = 2
    src = "../../imgandlai/source"
    dst = "../../imgandlai/processe"
    os.makedirs(dst,exist_ok=True)
    source_dir = sys.argv[1] if len(sys.argv) == 3 else src
    target_dir = sys.argv[2] if len(sys.argv) == 3 else dst
    os.makedirs(target_dir, exist_ok=True)
    if not any([source_dir, target_dir]):
        print("need both source directory and target directory")
    batches = gen_batches_vh(glob.glob(f"{source_dir}/*"),scale)
    for name,images in batches:
        index = 0
        for img in images:
            io = Image.fromarray(img)
            io.save(os.path.join(dst,f"{name.split('.')[0]}-{index}.jpg"))
            index +=1

