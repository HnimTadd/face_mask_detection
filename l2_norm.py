from lib import *

class L2Norm(nn.Module):
    def __init__(self,input_channels=512, scale=20):
        super(L2Norm, self).__init__()
        self.weight = nn.parameter.Parameter(torch.Tensor(input_channels))
        self.scale = scale
        self.reset_parameters()
        self.eps = 1e-10

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.scale)

    def forward(self, x):
        #L2Norm
        # X shape = (batch_size, channels, height, width)
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)

        #weight.size = (512) -> (1, 512) -> (1, 512, 1) -> (1, 512, 1, 1)
        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        return weights * x


if __name__ == "__main__":
    l2Norm = L2Norm(3,scale=1)
    x = torch.Tensor(np.arange(0,9).reshape((3,3, 1,1)))
    print(L2Norm.forward(l2Norm,x))