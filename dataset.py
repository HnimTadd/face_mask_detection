from lib import *
from transform import Transform
from extract_annotation import Anno_xml
from utils.random_array import train_val_separate
class MyDataset(data.Dataset):
    def __init__(self, imgs_list , annos_list,phase, transform, anno_xml) -> None:
        self.imgs = imgs_list
        self.annos = annos_list
        self.phase = phase
        self.transform = transform
        self.anno_xml = anno_xml

    def __len__(self) -> int:
        return len(self.imgs)
    
    def __getitem__(self, index):
        img, gt, _, _ = self.pull_item(index)

        return img, gt

    def pull_item(self, index):
        img_file_path = self.imgs[index]
        img = cv2.imread(img_file_path)
        height, width, channels = img.shape #BGR

        #get anno information
        ann_file_path = self.annos[index]
        ann_info = self.anno_xml(ann_file_path, width, height)


        # processing
        img, boxes, labels = self.transform(img,self.phase, ann_info[:, :4], ann_info[:, 4])

        #BGR -> RGB: (height, width, channels) -> (channels, height, width)
        img =  torch.from_numpy(img[:,:,(2,1,0)]).permute(2,0,1)

        #Ground truth 
        gt = gt = np.hstack((boxes, np.expand_dims(labels,axis=1)))

        return img, gt, height, width

def my_collate_fn(batch):
    targets = []
    imgs = []
    for sample in batch: #Sample: [(imgs, anns)]
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        
    #-> imgs size: (batchsize, channels, height, width)
    imgs = torch.stack(imgs,dim=0)
    return imgs, targets


if __name__ == "__main__":
    from make_datapath import make_datapath_list
    from extract_annotation import Anno_xml
    classes = ["with_mask", "without_mask", "mask_weared_incorrect"]

    anno_xml = Anno_xml(classes)

    root_path = "./data"
    imgs, anns = make_datapath_list(root_path=root_path)

    idx = 1
    img_file_path = imgs[idx]
    # print(imgs[idx])

    # img = cv2.imread(img_file_path) #[height, width, BGR]
    # height, width, channels = img.shape #get size img

    color_mean = (104, 117, 123)


    train_imgs, train_anns, val_imgs, val_anns = train_val_separate(imgs, anns, 0.9)
    transfrom = Transform(input_size=300,color_mean = color_mean)
    
    train_dataset = MyDataset(imgs,anns,"train",transfrom,anno_xml)

    batch_size = 4
    train_dataLoader = data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,collate_fn=my_collate_fn)

    val_dataset = MyDataset(val_imgs, val_anns, "val", transfrom, anno_xml)

    val_dataLoader = data.DataLoader(val_dataset, batch_size= batch_size,shuffle= False,collate_fn=my_collate_fn)

    dataloader_dict = {
        "train": train_dataLoader,
        "val": val_dataLoader
    }

    batch_iter = iter(dataloader_dict["train"])
    images, targets = next(batch_iter) #get 1 sample
    
    print(images.size())
    print(len(targets))
    print(targets[1].size())