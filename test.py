from random import shuffle
from models.hypernetworks.hypernetwork import SharedWeightsHyperNetResNet,SharedWeightsHyperNetResNetSeparable
import torch
from options.train_options import TrainOptions
import os
import random
from PIL import Image
# from models.stylegan2.model import Generator
if __name__ == '__main__':
    # args=TrainOptions().parse()
    # args.n_hypernet_outputs = 20
    # print(args)
    # #model=Generator(self.opts.output_size, 512, 8, channel_multiplier=2)
    # #print(model)
    # model=SharedWeightsHyperNetResNetSeparable(args)
    # total = sum([param.nelement() for param in model.parameters()])
    # print("Number of parameter: %.2fM" % (total/1e6))

    # model2=SharedWeightsHyperNetResNet(args)
    # total = sum([param.nelement() for param in model2.parameters()])
    # print("Number of parameter: %.2fM" % (total/1e6))

    import clip
    device='cpu'
    model, preprocess = clip.load("ViT-B/32", device=device)

    image = preprocess(Image.open("AF1.jpg")).unsqueeze(0).to(device)
    text = clip.tokenize(["right-side face", "frontal face", "left-side face"]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
