import torch
# import torchvision.transforms as transforms
from torchvision.transforms import *
import numpy as np
import cv2
from torchtools import load_pretrained_weights
from .model import Net
from .patchnet import patchnet
class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        # self.net = Net(reid=True)
        self.net = patchnet(num_classes=1)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        # state_dict = torch.load(model_path)['net_dict']
        # self.net.load_state_dict(state_dict)
        self.net = load_pretrained_weights(self.net, model_path)
        print("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        # self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.norm = Compose([
                            Resize((384, 128)), # (h, w)
                            ToTensor(),
                            Normalize(mean=MEAN, std=STD)])

    def __call__(self, img):
        assert isinstance(img, np.ndarray), "type error"
        img = img.astype(np.float)#/255.
        # img = cv2.resize(img, (64,128))
        img = torch.from_numpy(img).float().permute(2,0,1)
        img = self.norm(img).unsqueeze(0)
        with torch.no_grad():
            img = img.to(self.device)
            feature = self.net(img)
        return feature.cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:,:,(2,1,0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)

