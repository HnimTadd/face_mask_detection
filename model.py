from lib import *
from default_box import DefBox
from l2_norm import L2Norm
def create_vgg():
    layers = []
    in_channels = 3

    cfgs = [ 64, 64, 'M', 128, 128, 'M',
    256, 256, 256, 'MC', 512, 512, 512, 'M',
    512, 512, 512]

    for cfg in cfgs:
        if cfg == 'M': #Floor
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif cfg == 'MC': #Ceiling
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, cfg, kernel_size=3, stride=1, padding=1)

            relu =nn.ReLU(inplace=True)

            layers += [conv2d, relu]
            in_channels = cfg

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1,padding=1)

    conv6 = nn.Conv2d(512, 1024,kernel_size=3, stride=1, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return nn.ModuleList(layers)
    
def create_extras():
    layers = []
    in_channels = 1024
    cfgs = [256, 512, 128, 256, 128, 256, 128, 256]
    #1024->256
    layers += [nn.Conv2d(in_channels,cfgs[0],kernel_size=1,stride=1, padding=1)]
    layers += [nn.Conv2d(cfgs[0],cfgs[1],kernel_size=3,stride=2, padding=1)]

    layers += [nn.Conv2d(cfgs[1],cfgs[2],kernel_size=1,stride=1)]
    layers += [nn.Conv2d(cfgs[2], cfgs[3],kernel_size=3, stride=2, padding=1)]

    layers += [nn.Conv2d(cfgs[3], cfgs[4],kernel_size=1)]
    layers += [nn.Conv2d(cfgs[4], cfgs[5], kernel_size=3)]

    layers += [nn.Conv2d(cfgs[5], cfgs[6],kernel_size=1)]
    layers += [nn.Conv2d(cfgs[6], cfgs[7], kernel_size=3)]

    return nn.ModuleList(layers)

def creat_loc_conf(num_classes=4, bbox_aspect_nums=[4,6,6,6,4,4]):
    loc_layers=[]
    loc_layers += [nn.Conv2d(512,bbox_aspect_nums[0]*4,kernel_size=3, stride=1,padding=1)]
    loc_layers += [nn.Conv2d(1024,bbox_aspect_nums[1] * 4,kernel_size=3,stride=1,padding=1)]
    loc_layers += [nn.Conv2d(512,bbox_aspect_nums[2] * 4,kernel_size=3,stride=1,padding=1)]
    loc_layers += [nn.Conv2d(256,bbox_aspect_nums[3] * 4,kernel_size=3,stride=1,padding=1)]
    loc_layers += [nn.Conv2d(256,bbox_aspect_nums[4] * 4,kernel_size=3,stride=1,padding=1)]
    loc_layers += [nn.Conv2d(256,bbox_aspect_nums[5] * 4,kernel_size=3,stride=1,padding=1)]


    conf_layers =[]
    conf_layers += [nn.Conv2d(512,bbox_aspect_nums[0]*num_classes, kernel_size=3, padding=1,stride=1)]
    conf_layers += [nn.Conv2d(1024,bbox_aspect_nums[1]*num_classes,kernel_size=3,stride=1,padding=1)]
    conf_layers += [nn.Conv2d(512,bbox_aspect_nums[2]*num_classes,kernel_size=3,stride=1,padding=1)]
    conf_layers += [nn.Conv2d(256,bbox_aspect_nums[3]*num_classes,kernel_size=3,stride=1,padding=1)]
    conf_layers += [nn.Conv2d(256,bbox_aspect_nums[4]*num_classes,kernel_size=3,stride=1,padding=1)]
    conf_layers += [nn.Conv2d(256,bbox_aspect_nums[5]*num_classes,kernel_size=3,stride=1,padding=1)]

    # (512)
    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)

cfg = {
    "num_classes": 4,
    "input_size": 300,
    "bbox_aspect_num": [4,6,6,6,4,4], #So luong khung hinh cho cac source
    "feature_map": [38,19,10,5,3,1], #Size of feature of each source
    "steps": [8, 16, 32, 64, 100, 300],
    "max_size" : [60, 111, 162, 213, 264, 315],
    "min_size" : [30, 60, 111, 162, 213, 264],
    "aspect_ratios": [[2],[2,3],[2,3],[2,3],[2],[2]]
}


class SSD(nn.Module):
    def __init__(self, phase, cfg):
        super(SSD, self).__init__()
        self.phase = phase
        self.cfg = cfg
        self.num_classes = cfg["num_classes"]
        
        self.vgg = create_vgg()
        self.extras = create_extras()
        self.loc, self.conf = creat_loc_conf(self.num_classes,cfg["bbox_aspect_num"])
        self.L2Norm = L2Norm()
        defBox = DefBox(cfg)
        self.defBox = defBox.create_defbox()

        if phase == 'inference':
            # self.detect = Detect()
            pass

def decode(loc, defbox_list):
    '''
    loc: [8732, 4] : cx_offset, cl_offsety, h_offset, w_offset
    defbox_list: [8732,4]: cx_d, cy_d, h_d, w_d

    returns:
    boxes: [xmin, ymin, xmax, ymax]
    '''

    boxes = torch.cat((
        defbox_list[:,:2] + 0.1 *loc[:,:2]*defbox_list[:,:2],
        defbox_list[:,2:] * torch.exp(loc[:,2:])),dim=1)
    boxes[:,:2] -= boxes[:,2:]/2 #calculate xmin, ymin
    boxes[:,2:] += boxes[:,:2] #calculate xmax, ymax

    return boxes



if __name__ == '__main__':
    # x = create_vgg()
    # # print(x)

    # y = create_extras()
    # # print(y)

    # loc, conf = creat_loc_conf()
    # print(loc)
    # print(conf)

    ssd = SSD("train",cfg)
    print(ssd)