from RandMix import RandMix

import torch
from torchvision import transforms

class sampler:
    # def __from_torch__(self, image):
    #     res = image
    #     transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     res = transform(res).permute(0, 2, 3, 1).squeeze(0).detach().cpu()
    #     return res.numpy()

    # def __to_torch__(self):
        # self.torch_proto = (torch.from_numpy(self.prototype).permute(2, 0, 1).float() / 255.0).unsqueeze(0).cuda()

    def __init__(self, noise=1.0):
        # print("hello")
        self.noise = noise
        self.randomizer = RandMix(noise_lv=self.noise).cuda()
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def generate(self, images, labels, ratio):
        # print("bello")
        new_data = self.normalizer(torch.sigmoid(self.randomizer(images, ratio=ratio)))
        new_x = torch.cat([images, new_data])
        new_y = torch.cat([labels, labels])

        return RandMix(noise_lv=self.noise).forward(new_x), new_y

    # def generate(self):
    #     res = self.RandMix.forward(self.torch_proto)
    #     res = self.__from_torch__(res)
    #     return res