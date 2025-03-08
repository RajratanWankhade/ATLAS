import torch
from unet import UNet3D


def main():
    # Initialize model
    model = UNet3D(in_channel=1, out_channel=2)
    # Initialize optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])
    # Save my network
    torch.save(model.state_dict(), './my_network.pt')


if __name__ == '__main__':
    main()
