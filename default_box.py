from lib import *

cfg = {
    "num_classes": 4,
    "input_size": 300,
    "bbox_aspect_num": [4,6,6,4,4], #So luong khung hinh cho cac source
    "feature_map": [38,19,10,5,3,1], #Size of feature of each source
    "steps": [8, 16, 32, 64, 100, 300],
    "max_size" : [60, 111, 162, 213, 264, 315],
    "min_size" : [30, 60, 111, 162, 213, 264],
    "aspect_ratios": [[2],[2,3],[2,3],[2,3],[2],[2]]
}

class DefBox():
    def __init__(self, cfg):
        self.img_size= cfg["input_size"]
        self.feature_maps = cfg["feature_map"]
        self.steps = cfg["steps"]
        self.min_size = cfg["min_size"]
        self.max_size = cfg["max_size"]
        self.aspect_ratios = cfg["aspect_ratios"]
    
    def create_defbox(self):
        defbox_list =[]
        for k, f in enumerate(self.feature_maps):
            for i, j in itertools.product(range(f), repeat=2):
                f_k = self.img_size / self.steps[k]
                #Calculate center of default box
                cx = (i+0.5)/f_k
                cy = (j+0.5)/f_k

                #Calculcate scale of default box
                #size of small box
                s_k = self.min_size[k]/self.img_size #first case: 30/300
                
                defbox_list += [cx, cy,s_k, s_k]

                #big box
                s_k_ = sqrt(s_k*self.max_size[k]/self.img_size)
                defbox_list += [cx, cy, s_k_, s_k_]

                for a_k in self.aspect_ratios[k]:
                    w_k = s_k * sqrt(a_k)
                    h_k = s_k / sqrt(a_k)
                    
                    #horizontal box
                    defbox_list += [cx, cy, w_k, h_k]

                    #vertical box
                    defbox_list += [cx, cy, h_k, w_k]

        output = torch.Tensor(defbox_list).reshape((-1,4))
        output.clamp_(min=0,max=1)
        return output

if __name__ == "__main__":
    defBoxs = DefBox(cfg)
    print(defBoxs.create_defbox().size())