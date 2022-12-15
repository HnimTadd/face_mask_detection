from lib import *
from utils.random_array import *
def num_files(root_path , type) -> int:
    import fnmatch
    return len(fnmatch.filter(os.listdir(root_path), '*.'+type))


def make_datapath_list(root_path):
    '''
    params: string
    returns: imgs, anns
    '''
    image_path_template = osp.join(root_path, "images", "%s.png")
    annotation_path_template = osp.join(root_path, "annotations", "%s.xml")

    num_imgs = num_files(osp.join(root_path, 'images'), 'png')

    train_imgs = []
    train_anns = []
    for i in range(num_imgs):
        fileName = 'maksssksksss' + str(i)
        img_path = (image_path_template % fileName)
        ann_path = (annotation_path_template % fileName)
        train_imgs.append(img_path)
        train_anns.append(ann_path)

    return train_imgs, train_anns

if __name__ == "__main__":
    root_path = "./data"
    imgs, anns = make_datapath_list(root_path=root_path)
    imgs, anns = union_shuffled_copies(imgs, anns)
    train_imgs, train_anns, val_imgs, val_anns = train_val_separate(imgs, anns, 0.9)
    # for i in range(len(train_imgs)):
    #     print(train_imgs[i], train_anns[i])