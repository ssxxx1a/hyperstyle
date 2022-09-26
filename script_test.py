from models.hypernetworks.hypernetwork import SharedWeightsHyperNetResNet
import torch
if __name__ == '__main__':
    from options.train_options import TrainOptions
    opts=TrainOptions().parse()
    opts.n_hypernet_outputs=20
    opts.input_nc=3
    model=SharedWeightsHyperNetResNet(opts).cpu()
    x=torch.rand(size=(2,3,256,256)).cpu()
    aus=torch.rand(size=(2,17)).cpu()
    model.forward(x,aus)
    