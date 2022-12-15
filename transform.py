from utils.augmentations import Compose, ConvertFromInts, PhotometricDistort, ToAbsoluteCoords, ToPercentCoords, Expand, RandomSampleCrop, RandomMirror, Resize, SubtractMeans
from lib import *

class Transform():
    def __init__(self, input_size, color_mean):
        self.data_transform = {
            "train": Compose([
                ConvertFromInts(), #Convert image from int to float32
                ToAbsoluteCoords(), #Convert coordinates to origin type
                PhotometricDistort(), #Change color by random
                Expand(color_mean), #
                RandomSampleCrop(), #Random crop image
                RandomMirror(), #Mirror image
                ToPercentCoords(), #Normaliztion coordinates to (0,1)
                Resize(input_size), #Resize image
                SubtractMeans(color_mean) 
            ]),
            "val": Compose([
                ConvertFromInts(), #Convert image from int to float32
                Resize(input_size),
                SubtractMeans(color_mean)
            ])
        }

    def __call__(self, img, phase, boxes, labels):
        return self.data_transform[phase](img, boxes,labels)


if __name__ == "__main__":
    from make_datapath import make_datapath_list
    from extract_annotation import Anno_xml
    classes = ["with_mask", "without_mask", "mask_weared_incorrect"]

    anno_xml = Anno_xml(classes)

    root_path = "./data"
    imgs, anns = make_datapath_list(root_path=root_path)

    idx = 1
    img_file_path = imgs[idx]
    print(imgs[idx])

    img = cv2.imread(img_file_path) #[height, width, BGR]
    height, width, channels = img.shape #get size img

    anno_inform = anno_xml(anns[idx],width=width,height=height)
    import matplotlib.pyplot as plt
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    color_mean = (104, 117, 123)
    transfrom = Transform(input_size=300,color_mean = color_mean)
    transform_img, boxes, labels = transfrom(img,"train",anno_inform[:, :4],anno_inform[:,4])

    plt.imshow(cv2.cvtColor(transform_img, cv2.COLOR_BGR2RGB))
    plt.show()

    transform_img, boxes, labels = transfrom(img,"val",anno_inform[:, :4],anno_inform[:,4])

    plt.imshow(cv2.cvtColor(transform_img, cv2.COLOR_BGR2RGB))
    plt.show()