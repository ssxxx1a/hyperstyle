from models.hypernetworks.hypernetwork import SharedWeightsHyperNetResNet,SharedWeightsHyperNetResNetSeparable

from options.train_options import TrainOptions

# from models.stylegan2.model import Generator
if __name__ == '__main__':
    args=TrainOptions().parse()
    args.n_hypernet_outputs = 20
    print(args)
    #model=Generator(self.opts.output_size, 512, 8, channel_multiplier=2)
    #print(model)
    model=SharedWeightsHyperNetResNetSeparable(args)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))

    model2=SharedWeightsHyperNetResNet(args)
    total = sum([param.nelement() for param in model2.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))
    