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
    layers += [nn.Conv2d(in_channels,cfgs[0],kernel_size=1,stride=1)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(cfgs[0],cfgs[1],kernel_size=3,stride=2, padding=1)]
    layers += [nn.ReLU(inplace=True)]

    layers += [nn.Conv2d(cfgs[1],cfgs[2],kernel_size=1,stride=1)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(cfgs[2], cfgs[3],kernel_size=3, stride=2, padding=1)]
    layers += [nn.ReLU(inplace=True)]

    layers += [nn.Conv2d(cfgs[3], cfgs[4],kernel_size=1)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(cfgs[4], cfgs[5], kernel_size=3)]
    layers += [nn.ReLU(inplace=True)]

    layers += [nn.Conv2d(cfgs[5], cfgs[6],kernel_size=1)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(cfgs[6], cfgs[7], kernel_size=3)]
    layers += [nn.ReLU(inplace=True)]

    return nn.ModuleList(layers)

def creat_loc_conf(num_classes=4, bbox_aspect_nums=[4,6,6,6,4,4]):
    loc_layers=[]
    loc_layers += [nn.Conv2d(512, bbox_aspect_nums[0]*4, kernel_size=3, stride=1, padding=1)]
    loc_layers += [nn.Conv2d(1024, bbox_aspect_nums[1]*4, kernel_size=3, stride=1, padding=1)]
    loc_layers += [nn.Conv2d(512, bbox_aspect_nums[2]*4, kernel_size=3, stride=1, padding=1)]
    loc_layers += [nn.Conv2d(256, bbox_aspect_nums[3]*4, kernel_size=3, stride=1, padding=1)]
    loc_layers += [nn.Conv2d(256, bbox_aspect_nums[4]*4, kernel_size=3, stride=1, padding=1)]
    loc_layers += [nn.Conv2d(256, bbox_aspect_nums[5]*4, kernel_size=3, stride=1, padding=1)]


    conf_layers =[]
    conf_layers += [nn.Conv2d(512, bbox_aspect_nums[0]*num_classes, kernel_size=3, stride=1, padding=1)]
    conf_layers += [nn.Conv2d(1024, bbox_aspect_nums[1]*num_classes, kernel_size=3, stride=1, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_aspect_nums[2]*num_classes, kernel_size=3, stride=1, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_nums[3]*num_classes, kernel_size=3, stride=1, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_nums[4]*num_classes, kernel_size=3, stride=1, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_nums[5]*num_classes, kernel_size=3, stride=1, padding=1)]

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

        self.detect = Detect()
        if phase == 'inference':
            # self.detect = Detect()
            pass

    def forward(self, x):
        sources = []
        loc = []
        con = []
        for i in range(23):
            x = self.vgg[i](x)
        
        source1 = self.L2Norm(x)
        sources.append(source1)

        for i in range(23, len(self.vgg)):
            x = self.vgg[i](x)
        sources.append(x)

        for k, v in enumerate(self.extras):
            x = v(x)
            if k % 4 == 3:
                sources.append(x)
        # print("sources")

        for (source, l, c) in zip(sources, self.loc, self.conf):
            #current source shape: (batch_size, 4*aspect_ratio_num, featuremap_height, featuremap_width)
            #-> (batch_size, featuremap_height, featuremap_width, 4*aspect_ratio_num)
            loc.append(l(source).permute(0, 2, 3, 1).contiguous())
            con.append(c(source).permute(0, 2, 3, 1).contiguous())
            # print(np.shape(source))
        # print("loc: ", sum([len(i) for i in loc]))
        # print("conf: ", len(con))

        # current loc shape: (batch_size, featuremap_height, featuremap_width, 4 *aspect_ratio_num
        # -> loc: (batch_size, 8742*4)
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1) 

        # current con shape: (batch_size, featuremap_height, featuremap_width, numclasses * aspect_ratio_num
        # -> con: (batch_size, 8732 * num_classes)
        con = torch.cat([o.view(o.size(0), -1) for o in con], 1)

        # curren_loc shape: (batch_size, 8732* 4)
        # -> (batch_size, 8732, 4)
        loc = loc.view(loc.size(0),-1, 4)

        # current con shape: (batch_size, 8732 * num_classes)
        # -> (batch_size, 8732, num_classes)
        con = con.view(con.size(0), -1, self.num_classes)


        output =  (loc, con, self.defBox)

        # print("dbox: ", self.defBox.size())
        if self.phase == "inference":
            return self.detect(output[0], output[1], output[2])
        else:
            return output

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

#Non maximum supression
def nms(boxes, scores,threshold=0.5, top_k= 200):
    '''
    boxes: [num_boxes, 4]
    scores: [num_boxes]
    '''
    count=0 
    keep = scores.new(scores.size(0)).zero_().long()

    #Box coordinate
    x1 = boxes[:, 0] #(num_boxes)
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    #calc area of boxes
    area = torch.mul(x2-x1,y2-y1)  #(num_boxes)

    tmp_x1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()

    value, idx = scores.sort(dim=0) #Sorted in increase order
    idx = idx[-top_k:] #list idx of top k box with greatest confidence score

    while len(idx) > 0:
        i = idx[-1] #idx of greatest confidence score box

        keep[count] = i
        count += 1
        
        if len(idx) == 1:
            break

        idx = idx[:-1] # remove idx of greatest confidence score box

        #get information box

        torch.index_select(input=x1, dim=0, index=idx, out=tmp_x1)
        torch.index_select(input=x2, dim=0, index=idx, out=tmp_x2)
        torch.index_select(input=y1, dim=0, index=idx, out=tmp_y1)
        torch.index_select(input=y2, dim=0, index=idx, out=tmp_y2)

        #Get coordinates of intersection box
        tmp_x1 = torch.clamp(tmp_x1, min=x1[i])
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x2, max=x2[i])
        tmp_y2 = torch.clamp(tmp_y2, max=y2[i])

        #size of tmp_w, tmp_h = (size(ofcurrentidx, dim=0))
        tmp_w.resize_as_(tmp_x2)
        tmp_h.resize_as_(tmp_y2)

        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1

        tmp_w = torch.clamp(tmp_w, min=0.0, max=1.0)
        tmp_h = torch.clamp(tmp_h, min=0.0, max=1.0)

        #intersection area
        inter = torch.mul(tmp_w, tmp_h)

        #area of others boxes
        others_area = torch.index_select(area, 0, idx)

        #union area
        union = area[i] + others_area - inter

        iou = torch.div(inter, union)

        idx=idx[iou.le(threshold)]
    
    return  keep, count

class Detect(Function):
    def __init__(self, conf_thresehold=0.01, top_k=200, nms_threshold=0.45):
        self.softmax = nn.Softmax(dim=-1)
        self.conf_thresh = conf_thresehold
        self.top_k = top_k
        self.nms_thresh = nms_threshold

    def __call__(self, loc_data, conf_data, dbox_list):
        num_batch = loc_data.size(0) #batch size
        num_dbo = loc_data.size(1) #8732
        num_classes = conf_data.size(2) #4

        conf_data = self.softmax(conf_data)

        #(batch_size, num_box, num_classes) -> (batch, num_classes, num_box)
        conf_preds = conf_data.transpose(2,1)

        output = torch.zeros(num_batch, num_classes, self.top_k, 5)
        #loop over batch
        for ele in range(num_batch):
            # calc decoded bbound from offset information and default box
            decoded_boxes = decode(loc_data[ele], dbox_list)

            #copy confidence of batch element
            conf_scores = conf_preds[ele].clone()
            
            #loop from class 1 to end (ignore background class)
            for cl in range(1, num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh) # get confidence > 0.01
                scores = conf_scores[cl][c_mask]

                if scores.nelement() == 0: 
                    continue


                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes) #(8732, 4)

                boxes = decoded_boxes[l_mask].view(-1, 4)

                ids, count = nms(boxes=boxes.detach(),scores=scores.detach(),threshold=self.nms_thresh,top_k=self.top_k)
                output[ele, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), dim=1)
            return output



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