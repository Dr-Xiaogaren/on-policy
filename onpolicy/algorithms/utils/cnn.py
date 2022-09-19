import torch.nn as nn
import torchvision.models as models
import torch
"""CNN Modules and utils."""

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNNBase(nn.Module):
    def __init__(self, args, input_channels, input_size):
        super(CNNBase, self).__init__()

        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU

        cov_block = [
                     nn.Conv2d(input_channels, 10, kernel_size=3, stride=1, padding=1),
                     nn.BatchNorm2d(num_features=10),
                     nn.ReLU(),
                     nn.MaxPool2d(kernel_size=2, stride=2),
                     nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1),
                     nn.BatchNorm2d(num_features=10),
                     nn.ReLU(),
                     nn.MaxPool2d(kernel_size=2, stride=2),
                     nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1),
                     nn.BatchNorm2d(num_features=10),
                     nn.ReLU(),
                     nn.AdaptiveAvgPool2d((12,12))
                    ]
        self.Cov = nn.Sequential(*cov_block)
        test_input = torch.randn(1, input_channels, input_size, input_size)
        test_output = self.Cov(test_input)
        test_output_size = test_output.size(-1)

        # self.Flatten = nn.Conv2d(40, 256, kernel_size=test_output_size)
        
        self.output_size = test_output.view(-1).size(0)
        
    def forward(self, x):
        x = self.Cov(x)
        # x = self.Flatten(x)
        return x.view(x.size(0),-1)

def main():
    from onpolicy.envs.mpe.environment import MultiAgentEnv, CatchingEnv
    from onpolicy.envs.mpe.scenarios import load
    from onpolicy.config import get_config
    parser = get_config()
    args = parser.parse_known_args()[0]
    args.num_agents = 4
    cnn_net = CNNBase(args,5,320)
    x = torch.randn(1,5,320,320)
    y = cnn_net(x)
    print(y.shape)

if __name__=="__main__":
   main()