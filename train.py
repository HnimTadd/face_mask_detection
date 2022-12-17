from lib import *
from make_datapath import make_datapath_list
from utils.random_array import train_val_separate
from dataset import MyDataset, my_collate_fn
from transform import Transform
from extract_annotation import Anno_xml
from model import SSD
from multiboxloss import MultiBoxLoss

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device:", device)
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

# create dataloader
root_path = "./data"
imgs, anns = make_datapath_list(root_path)

train_imgs, train_anns, val_imgs, val_anns = train_val_separate(imgs,anns,0.9)
classes = ["with_mask", "without_mask", "mask_weared_incorrect"]

color_mean = (104, 117, 123)
input_size = 300

transform = Transform(input_size, color_mean)
anno_xml = Anno_xml(classes)
train_dataset = MyDataset(train_imgs, train_anns, "train", transform, anno_xml)
val_dataset = MyDataset(val_imgs, val_anns, phase="val", transform=transform, anno_xml=anno_xml)

batch_size = 8
train_dataloader = data.DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=my_collate_fn)
val_dataloader = data.DataLoader(val_dataset, batch_size, shuffle=False, collate_fn=my_collate_fn)

dataLoader_dict = {
    "train": train_dataloader,
    "val": val_dataloader
}


# Create network
cfg = {
    "num_classes": 4,
    "input_size": 300,
    "bbox_aspect_num": [4,6,6,6,4,4], #Num of anchor box of each feature map position of each source
    "feature_map": [38,19,10,5,3,1], #Size of feature of each source
    "steps": [8, 16, 32, 64, 100, 300],
    "max_size" : [60, 111, 162, 213, 264, 315],
    "min_size" : [30, 60, 111, 162, 213, 264],
    "aspect_ratios": [[2],[2,3],[2,3],[2,3],[2],[2]]
}
net = SSD(cfg=cfg, phase="train")

vgg_weight = torch.load("./data/weights/vgg16_reducedfc.pth")

net.vgg.load_state_dict(vgg_weight)

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias,0.0)

#Init weight of network
net.extras.apply(weight_init)
net.loc.apply(weight_init)
net.conf.apply(weight_init)

#Create Multiboxloss
criterion = MultiBoxLoss(0.45, 3, device.type)

#create optimizer
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9,weight_decay=5e-4)


#Training:
def train_model(net, dataloader_dict, criterion, optimizer, num_epochs):
    # move network to gpu
    net.to(device)
    iteration = 1
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    logs  = []

    for epoch in range(num_epochs + 1):
        t_epoch_start = time.time()
        t_iter_start = time.time()

        print("----"*20)
        print("Epoch:{}/{}".format(epoch + 1, num_epochs))
        print("----"*20)

        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
                print("(Training)")
            else:
                if epoch + 1 % 10 == 0:
                    net.eval()
                    print("(Validating)")
                else:
                    continue
            for images, targets in dataloader_dict[phase]:
                # move images, annotation target to gpu
                images = images.to(device)
                targets = [ann.to(device) for ann in targets]

                # init optimizer
                optimizer.zero_grad()

                #forward ()
                # in phase trainging: weight trainable = true
                with torch.set_grad_enabled(phase=="train"):
                    outputs = net(images)
                    # print(np.shape(targets))
                    loss_l, loss_c = criterion(outputs, targets)
                    loss = loss_l + loss_c

                    if phase == "train":
                        loss.backward() #calcuate gradient

                        nn.utils.clip_grad.clip_grad_value_(net.parameters(),clip_value=2.0)

                        optimizer.step() #update parameter

                        if (iteration % 10) == 0:
                            t_iter_end = time.time()
                            duration = t_iter_end - t_iter_start
                            print("Iteration {} || Loss: {:4f} || 10iter:{:4f} secs".format(iteration, loss.item(), duration))
                            t_iter_start = time.time()
                        epoch_train_loss += loss.item()
                        iteration += 1
                    else:
                        epoch_val_loss += loss.item()
        t_epoch_end = time.time()
        epoch_duration = t_epoch_end - t_epoch_start
        print("----"*20)
        print("Epoch {} || epoch_train_loss: {:4f} || epoch_val_loss: {:4f}".format(epoch+1,epoch_train_loss, epoch_val_loss))
        print("Duration: {:4f} secs".format(epoch_duration))
        t_epoch_start = time.time()

        log_epoch = {
            "epoch": epoch + 1,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
        }

        logs.append(log_epoch)

        df = pd.DataFrame(logs)

        df.to_csv("./data/ssd_logs.csv")

        #reset loss using in next epochs
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        if (epoch + 1) % 10 == 0:
            torch.save(net.state_dict(), "./data/weights/ssd300_epoch{}.pth".format(epoch+1))

num_epochs = 30

train_model(net, dataLoader_dict,criterion, optimizer,num_epochs)
