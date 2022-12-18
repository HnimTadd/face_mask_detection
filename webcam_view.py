from lib import *
from transform import Transform
from model import SSD

frameWidth = 640
frameHeight = 480

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device:", device)
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()


cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

detector = colors = [(255,0,0), (0,255,0), (0,0,255)]
font = cv2.FONT_HERSHEY_DUPLEX
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

net_weights = torch.load("./data/current_weights/ssd300_epoch30.pth")
net.load_state_dict(net_weights)
net.to(device)
color_mean = (104, 117, 123)
input_size = 300
transform = Transform(input_size=input_size, color_mean= color_mean)




def cv2_demo(net, transform: Transform):
    def predict(frame):
        # heigt, width = frame.shape[:2]
        img_tranformed = transform(frame, "val", "val", "")[0]
        img_tensor = torch.from_numpy(img_tranformed[:,:,(2,1,0)]).permute(2,0,1)
        net.eval()
        input = img_tensor.unsqueeze(0) #(1, 3, 300, 300)
        output = net(input.cuda())

        detections = output.data #(1, 21, 200, 5) 5: score, cx, cy, w, h
        scale = torch.Tensor(frame.shape[1::-1]).repeat(2)
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.7:
                score = detections[0, i, j, 0]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                cv2.rectangle(frame,
                            (int(pt[0]), int(pt[1])),
                            (int(pt[2]), int(pt[3])),
                            colors[i%3], 2
                            )
                display_text = "%s: %.2f"%(classes[i-1], score)
                cv2.putText(frame, display_text, (int(pt[0]) - 10, int(pt[1]) - 10),
                    font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                j += 1
        return frame

    print("Starting video stream thread...")
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame =  predict(frame)
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) == ord('q'):
            break
        


cv2_demo(net, transform)
cap.release()
cv2.destroyAllWindows()