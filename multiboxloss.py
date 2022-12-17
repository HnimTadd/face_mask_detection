# Jaccard:
# Hard negativive mining:
# Loss in regression: MSE SmoothL1
# Loss in classification: (multi class): cross entropy loss
from lib import *
from utils.box_utils import *

class MultiBoxLoss(nn.Module):
    def __init__ (self, jaccard_threshold=0.45, neg_pos = 3, device='gpu'):
        super(MultiBoxLoss, self).__init__()
        self.jaccard_threshold = jaccard_threshold
        self.neg_pos = neg_pos
        self.device = device

    def forward(self, predictions, targets):
        loc_data, conf_data, dbox_list = predictions

        # loc_data shape (batch_num, num_dbox, 4)
        num_batch = loc_data.size(0)
        num_dbox = loc_data.size(1)
        #conf shape (batch_num, num_bbox, num_classes)
        num_classes = conf_data.size(2)
        # print("loc data: ", loc_data.size())
        conf_t_label = torch.LongTensor(num_batch,num_dbox).to(self.device)

        loc_t = torch.Tensor(num_batch, num_dbox, 4).to(self.device)

        for idx in range(num_batch):
            truths = targets[idx][:, :-1].to(self.device) #bounding box (xmin, ymin, xmax, ymax)
            labels = targets[idx][:, -1].to(self.device) # label of bounding box (labels)

            dbox = dbox_list.to(self.device)
            variances = [0.1, 0.2]
            match(self.jaccard_threshold, truths, dbox, variances, labels, loc_t, conf_t_label, idx)
        
        #SmoothL1Loss
        pos_mask = conf_t_label > 0
        #loc data (num_batch, 8732, 4)
        pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data)

        #positive dbox, loc_data:
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)

        #Loss location
        loss_l = F.smooth_l1_loss(loc_p,loc_t, reduction="sum")





        #Loss confidence
        #CrossEntropy
        batch_conf = conf_data.view(-1, num_classes) #(numbatch, num_box, num_class)
        # print(np.shape(batch_conf))
        #shape of conf_t_label: (numbatch, numdbox * num_classes) -> (numbatch * numdbox* num_classes)
        # print(np.shape(conf_t_label.view(-1)))
        # exit()
        loss_conf = F.cross_entropy(batch_conf, conf_t_label.view(-1), reduction="none")

        num_pos = pos_mask.long().sum(dim=1, keepdim=True)
        loss_conf = loss_conf.view(num_batch, -1) #size(num_batch, 8732)

        _, loss_idx = loss_conf.sort(dim=1, descending=True)
        _, idx_rank = loss_idx.sort(1)

        num_neg = torch.clamp(self.neg_pos*num_pos, max = num_dbox)
        neg_mask = idx_rank < (num_neg).expand_as(idx_rank)

        
        
        #idx_rank is list of idx sorted by pos
        pos_idx_mask = pos_mask.unsqueeze(2).expand_as(conf_data)
        neg_idx_mask = neg_mask.unsqueeze(2).expand_as(conf_data)

        #get confidence prediction
        conf_t_pre = conf_data[(pos_idx_mask + neg_idx_mask).gt(0)].view(-1, num_classes)

        #get confidence target of bounding which is hard negative and positive 
        conf_t_label = conf_t_label[(pos_mask+neg_mask).gt(0)]

        loss_conf = F.cross_entropy(conf_t_pre, conf_t_label,reduction="sum")

        #calc totalloss = loss_l + alpha * loss_conf
        N = num_pos.sum()
        loss_l = loss_l/N
        loss_conf = loss_conf/N

        return loss_l, loss_conf
