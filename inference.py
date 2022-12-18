from lib import *
from model import SSD
from transform import Transform
import sys

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device:", device)
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()


classes = ["with_mask", "without_mask", "mask_weared_incorrect"]

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
net = SSD(cfg=cfg, phase="inference")
net.to(device)

net_weights = torch.load("./data/current_weights/ssd300_epoch30.pth")

net.load_state_dict(net_weights)
# net.to(device)

def show_predict(img_file_path):
    img = cv2.imread(img_file_path)

    color_mean = (104, 117, 123)
    input_size = 300
    transform = Transform(input_size, color_mean)

    phase = "val"
    img_tranformed, boxes, labels = transform(img, phase, "", "")
    img_tensor = torch.from_numpy(img_tranformed[:,:,(2,1,0)]).permute(2,0,1)
    net.eval()
    input = img_tensor.unsqueeze(0) #(1, 3, 300, 300)
    output = net(input.cuda())

    plt.figure(figsize=(10, 10))
    colors = [(255,0,0), (0,255,0), (0,0,255)]
    font = cv2.FONT_HERSHEY_SIMPLEX

    detections = output.data #(1, 21, 200, 5) 5: score, cx, cy, w, h
    scale = torch.Tensor(img.shape[1::-1]).repeat(2)

    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.5:
            score = detections[0, i, j, 0]
            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
            cv2.rectangle(img,
                          (int(pt[0]), int(pt[1])),
                          (int(pt[2]), int(pt[3])),
                          colors[i%3], 2
                          )
            display_text = "%s: %.2f"%(classes[i-1], score)
            cv2.putText(img, display_text, (int(pt[0]), int(pt[1])),
                font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            j += 1
    
    cv2.imwrite(os.path.join("./data/result_img/" + sys.argv[1]), img)
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_file_path = "./data/test_img/" + sys.argv[1]
        show_predict(img_file_path)

cv2.destroyAllWindows()