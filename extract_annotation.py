from lib import *

class  Anno_xml(object):
    def __init__(self, classes):
        self.classes = classes
        
    def __call__(self, xml_path, width, height) :


        # include image annotation
        ret = []

        # read file xml
        xml = ET.parse(xml_path).getroot()

        for obj in xml.iter('object'):

            # information for bounding box
            bndbox = []
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            pts = ["xmin", "ymin", "xmax", "ymax"]
            for pt in pts:
                pixel = int(bbox.find(pt).text) - 1

                if pt == 'xmin' or pt == 'xmax':
                    pixel /= width #ratio of width
                else:
                    pixel /= height #ratio of height
                
                bndbox.append(pixel)

            label_id = self.classes.index(name)
            bndbox.append(label_id)

            ret += [bndbox]

        return np.array(ret) #[[xmin, ymin, xmax, ymax, label_id], ...]



if __name__ == "__main__":
    classes = ["with_mask", "without_mask", "mask_weared_incorrect"]

    anno_xml = Anno_xml(classes)
    from make_datapath import make_datapath_list
    root_path = "./data"
    imgs, anns = make_datapath_list(root_path=root_path)

    idx = 1
    img_file_path = imgs[idx]
    print(imgs[idx])

    img = cv2.imread(img_file_path) #[height, width, BGR]
    height, width, channels = img.shape #get size img

    anno_inform = anno_xml(anns[idx],width=width,height=height)
