import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from models import alexnet, resnet, vgg

# Training settings
parser = argparse.ArgumentParser(description='Train AlexNet on CIFAR or STL dataset')
parser.add_argument('data_path', type=str,
                    help='path to dataset')
parser.add_argument('--dataset', type=str, choices = ['cifar10','cifar100','stl10'], default='cifar10',
                    help='choose dataset from cifar10/100 or stl10')
parser.add_argument('--trained_model', action='store_true', default=False,
                    help='load trained model')
parser.add_argument('--trained_model_path', type=str,
                    help='path to trained model')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size (default: 128)')
parser.add_argument('--workers', type=int, default=2,
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.01, 
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, 
                    help='momentum (default: 0.9)')
parser.add_argument('--decay', type=float, default=0.0005, 
                    help='Weight decay (L2) (default: 0.0005)')
parser.add_argument('--gamma', type=float, default=0.2, 
                    help='Learning rate step gamma (default: 0.2)')
parser.add_argument('--save-model', action='store_true', default=True,
                        help='saving model (default: True)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--zero_threshold', type=float, default=0.001, 
                    help='threshold to define zero weight (default: 0.001)')
parser.add_argument('--_lambda', type=float, default=0.001, 
                    help='hypaerparameter for regularization tearm (default: 0.001)')
parser.add_argument('--_lambda2', type=float, default=0.5, 
                    help='balancing parameter between regularization tearm (default: 0.5)')


args = parser.parse_args()

def main():

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    transform_test = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.485, 0.456, 0.406),
							(0.229, 0.224, 0.225)),
		])

    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
				transforms.RandomCrop(32,padding = 4),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize((0.485, 0.456, 0.406),
									(0.229, 0.224, 0.225)),
			])
        trainset = datasets.CIFAR10(root=args.data_path,train=True,download=False,transform=transform_train)
        testset = datasets.CIFAR10(root=args.data_path,train=False,download=False,transform=transform_test)
        num_classes = 10
    elif args.dataset == 'cifar100':
        transform_train = transforms.Compose([
				transforms.RandomCrop(32,padding = 4),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize((0.485, 0.456, 0.406),
									(0.229, 0.224, 0.225)),
			])
        trainset = datasets.CIFAR100(root=args.data_path,train=True,download=False,transform=transform_train)
        testset = datasets.CIFAR100(root=args.data_path,train=False,download=False,transform=transform_test)
        num_classes = 100
    elif args.dataset == 'stl10':
        transform_train = transforms.Compose([
				transforms.RandomCrop(96,padding = 4),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize((0.485, 0.456, 0.406),
									(0.229, 0.224, 0.225)),
			])
        trainset = datasets.STL10(root=args.data_path,train=True,download=False,transform=transform_train)
        testset = datasets.STL10(root=args.data_path,train=False,download=False,transform=transform_test)
        num_classes = 10
        
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=args.batch_size,shuffle=True,num_workers=args.workers)
    testloader = torch.utils.data.DataLoader(testset,batch_size=args.batch_size,shuffle=False, num_workers=args.workers)
    
    net = alexnet.alexnet(num_classes = num_classes).to(device)
    if args.trained_model:
        ckpt = torch.load(args.trained_model_path, map_location= device)
        net.load_state_dict(ckpt)
        
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)

    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120,160], gamma=args.gamma)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(args.epochs/3), gamma=args.gamma)

    regularization = sparse_regularization(net,device)
    
    for epoch in range(1, args.epochs + 1):
        train(args, net, device, trainloader, optimizer, criterion, epoch, regularization)
        test(args, net, device, testloader, criterion)
        scheduler.step()

    if args.save_model:
        torch.save(net.state_dict(), str(args.dataset)+"_alexnet.pt")
        

def train(args, model, device, trainloader, optimizer, criterion, epoch, regularization):

    sum_loss = 0.0
    sum_correct = 0
    sum_total = 0
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        sum_loss += loss.item()
        loss += args._lambda2*regularization.hierarchical_squared_group_l12_regularization(args._lambda)
        loss += (1-args._lambda2)*regularization.l1_regularization(args._lambda)
        loss.backward()
        optimizer.step()
        _, predicted = output.max(1)
        sum_total += target.size(0)
        sum_correct += (predicted == target).sum().item()
    print("train mean loss={}, accuracy={}, sparsity={}"
            .format(sum_loss*args.batch_size/len(trainloader.dataset), float(sum_correct/sum_total), sparsity(model)))


def test(args, model, device, testloader, criterion):
    model.eval()
    sum_loss = 0.0
    sum_correct = 0
    sum_total = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            sum_loss += loss.item()
            _, predicted = output.max(1)
            sum_total += target.size(0)
            sum_correct += (predicted == target).sum().item()
    print("test mean loss={}, accuracy={}"
            .format(sum_loss*args.batch_size/len(testloader.dataset), float(sum_correct/sum_total)))
    
def sparsity(model):
    number_of_conv_weight = 0
    number_of_zero_conv_weight = 0
    for n, _module in model.named_modules():
            if isinstance(_module, nn.Conv2d) and  (not 'downsample' in n):
                p = torch.flatten(_module.weight.data)
                number_of_conv_weight += len(p)
                number_of_zero_conv_weight += len(p[torch.abs(p)<args.zero_threshold])
    return number_of_zero_conv_weight/number_of_conv_weight

class sparse_regularization(object):
    
    def __init__(self, model: nn.Module, device):
        self.model = model
        self.device = device
        
    #L1 regularization
    def l1_regularization(self, _lambda):
        x = 0
        for n, _module in self.model.named_modules():
            if isinstance(_module, nn.Conv2d) and  (not 'downsample' in n):
                x += _lambda*torch.norm(torch.flatten(_module.weight),1)
        return x
	
    #group lasso regularization
    def group_lasso_regularization(self, _lambda):
        x = 0
        for n, _module in self.model.named_modules():
            if isinstance(_module, nn.Conv2d) and  (not 'downsample' in n):
                p = _module.weight
                p = p.reshape(p.shape[0],p.shape[1],p.shape[2]*p.shape[3])
                
                #group lasso regularization based on the feature wise grouping
                x += _lambda*torch.sum(torch.sqrt(torch.sum(torch.sum(p**2,0),1)))

                #group lasso regularization based on the neuron wise grouping
				#x += _lambda*torch.sum(torch.sqrt(torch.sum(torch.sum(p**2,1),1)))
        return x

    #hierarchical square rooted group lasso regularization
    def hierarchical_square_rooted_group_lasso_regularization(self, _lambda):
        x = 0
        for n, _module in self.model.named_modules():
            if isinstance(_module, nn.Conv2d) and  (not 'downsample' in n):
                p = _module.weight
                number_of_out_channels = p.shape[0]
                number_of_in_channels = p.shape[1]
                p = p.reshape(p.shape[0]*p.shape[1],p.shape[2]*p.shape[3])
                p = torch.sum(p**2,1)
                p = p.reshape(number_of_out_channels,number_of_in_channels)
                
                #hierarchical square rooted group lasso regularization based on the feature wise grouping
                x+= _lambda*torch.sum(torch.sqrt(torch.sum(torch.sqrt(p),0)))

                #hierarchical square rooted group lasso regularization based on the neuron wise grouping
                #x+= _lambda*torch.sum(torch.sqrt(torch.sum(torch.sqrt(p),1)))
        return x

    #hierarchical squared group lasso regularization
    def hierarchical_squared_group_lasso_regularization(self, _lambda):
        x = 0
        for n, _module in self.model.named_modules():
            if isinstance(_module, nn.Conv2d) and  (not 'downsample' in n):
                p = _module.weight
                number_of_out_channels = p.shape[0]
                number_of_in_channels = p.shape[1]
                p = p.reshape(p.shape[0]*p.shape[1],p.shape[2]*p.shape[3])
                p = torch.sum(p**2,1)
                p = p.reshape(number_of_out_channels,number_of_in_channels)
                
                #hierarchical squared group lasso regularization based on the feature wise grouping
                x+= _lambda*torch.sum((torch.sum(torch.sqrt(p),0))**2)

                #hierarchical squared group lasso regularization based on the neuron wise grouping
                #x+= _lambda*torch.sum((torch.sum(torch.sqrt(p),1))**2)
        return x
    
    #exclusive sparsity regularization
    def exclusive_sparsity_regularization(self, _lambda):
        x = 0
        for n, _module in self.model.named_modules():
            if isinstance(_module, nn.Conv2d) and  (not 'downsample' in n):
                p = _module.weight
                p = p.reshape(p.shape[0],p.shape[1],p.shape[2]*p.shape[3])

                #exclusive sparsity regularization based on the feature wise grouping
                x += _lambda*torch.sum((torch.sum(torch.sum(torch.abs(p),0),1))**2)

                #exclusive sparsity regularization based on the feature wise grouping
				#x += _lambda*torch.sum((torch.sum(torch.sum(torch.abs(p),1),1))**2)
        return x
    
    #hierarchical square rooted exclusive sparsity regularization
    def hierarchical_square_rooted_exclusive_sparsity_regularization(self, _lambda):
        x = 0
        for n, _module in self.model.named_modules():
            if isinstance(_module, nn.Conv2d) and  (not 'downsample' in n):
                p = _module.weight
                number_of_out_channels = p.shape[0]
                number_of_in_channels = p.shape[1]
                p = p.reshape(p.shape[0]*p.shape[1],p.shape[2]*p.shape[3])
                p = torch.sum(torch.abs(p),1)
                p = p.reshape(number_of_out_channels,number_of_in_channels)
                
                #hierarchical square rooted exclusive sparsity regularization based on the feature wise grouping
                x+= _lambda*torch.sum(torch.sqrt(torch.sum(p**2,0)))

                #hierarchical square rooted exclusive sparsity regularization based on the neuron wise grouping
                #x+= _lambda*torch.sum(torch.sqrt(torch.sum(p**2,1)))
        return x

    #hierarchical squared exclusive sparsity regularization
    def hierarchical_squared_exclusive_sparsity_regularization(self, _lambda):
        x = 0
        for n, _module in self.model.named_modules():
            if isinstance(_module, nn.Conv2d) and  (not 'downsample' in n):
                p = _module.weight
                number_of_out_channels = p.shape[0]
                number_of_in_channels = p.shape[1]
                p = p.reshape(p.shape[0]*p.shape[1],p.shape[2]*p.shape[3])
                p = torch.sum(torch.abs(p),1)
                p = p.reshape(number_of_out_channels,number_of_in_channels)
                
                #hierarchical squared exclusive sparsity regularization based on the feature wise grouping
                x+= _lambda*torch.sum((torch.sum(p**2,0))**2)

                #hierarchical squared exclusive sparsity regularization based on the neuron wise grouping
                #x+= _lambda*torch.sum((torch.sum(p**2,1))**2)
        return x

    #group l1/2 regularization
    def group_l12_regularization(self, _lambda):
        x = 0
        for n, _module in self.model.named_modules():
            if isinstance(_module, nn.Conv2d) and  (not 'downsample' in n):
                p = _module.weight
                p = p.reshape(p.shape[0],p.shape[1],p.shape[2]*p.shape[3])

                #group l1/2 regularization based on the feature wise grouping
                x += _lambda*torch.sum(torch.sqrt(torch.sum(torch.sum(torch.abs(p),0),1)))

                #group l1/2 regularization based on the feature wise grouping
                #x += _lambda*torch.sum(torch.sqrt(torch.sum(torch.sum(torch.abs(p),1),1)))
        return x
    
    #hierarchical square rooted group l1/2 regularization
    def hierarchical_square_rooted_group_l12_regularization(self, _lambda):
        x = 0
        for n, _module in self.model.named_modules():
            if isinstance(_module, nn.Conv2d) and  (not 'downsample' in n):
                p = _module.weight
                number_of_out_channels = p.shape[0]
                number_of_in_channels = p.shape[1]
                p = p.reshape(p.shape[0]*p.shape[1],p.shape[2]*p.shape[3])
                p = torch.sum(torch.abs(p),1)
                p = p.reshape(number_of_out_channels,number_of_in_channels)
                
                #hierarchical square rooted group l1/2 regularization based on the feature wise grouping
                x+= _lambda*torch.sum(torch.sqrt(torch.sum(torch.sqrt(p),0)))

                #hierarchical square rooted group l1/2 regularization based on the neuron wise grouping
                #x+= _lambda*torch.sum(torch.sqrt(torch.sum(torch.sqrt(p),1)))
        return x

    #hierarchical squared group l1/2 regularization
    def hierarchical_squared_group_l12_regularization(self, _lambda):
        x = 0
        for n, _module in self.model.named_modules():
            if isinstance(_module, nn.Conv2d) and  (not 'downsample' in n):
                p = _module.weight
                number_of_out_channels = p.shape[0]
                number_of_in_channels = p.shape[1]
                p = p.reshape(p.shape[0]*p.shape[1],p.shape[2]*p.shape[3])
                p = torch.sum(torch.abs(p),1)
                p = p.reshape(number_of_out_channels,number_of_in_channels)
                
                #hierarchical squared group l1/2 regularization based on the feature wise grouping
                x+= _lambda*torch.sum((torch.sum(torch.sqrt(p),0))**2)

                #hierarchical squared group l1/2 regularization based on the neuron wise grouping
                #x+= _lambda*torch.sum((torch.sum(torch.sqrt(p),1))**2)
        return x

if __name__ == '__main__':
    main()