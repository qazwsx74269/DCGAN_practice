import numpy as np
import torch as t
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from torchvision import transforms
import ipdb
import tqdm
import fire
import os
import visdom
from PIL import Image
from torchnet.meter import AverageValueMeter

#first step: build your own dataset class and Config class
class Config(object):
        path = '/home/szh/DCGAN/data/faces'
        imgsize = 96
        batch_size = 2048
        max_epoch = 200
        drop_last=True
        num_workers = 4
        generator_model_path = '/home/szh/checkpoints/mygenerator_epoch180.pth'
        discriminator_model_path = '/home/szh/checkpoints/mydiscriminator_epoch180.pth'

opt = Config()

class MyDataset(Dataset):
        def __init__(self,root):
                self.imgs = [os.path.join(root,img) for img in os.listdir(root)]
        def __getitem__(self,index):
                self.transforms = transforms.Compose([
                        transforms.Resize(opt.imgsize),
                        transforms.CenterCrop(opt.imgsize),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
                        )
                img = Image.open(self.imgs[index])
                return self.transforms(img)
        def __len__(self):
                return len(self.imgs)




#define your generator class
class Generator(nn.Module):
        def __init__(self):
                super(Generator,self).__init__()
                #Hout = (Hin-1)*stride-2*padding+kernel_size
                self.net = nn.Sequential(
                        nn.ConvTranspose2d(100,64*8,4,1,0,bias=False),
                        nn.BatchNorm2d(64*8),
                        nn.ReLU(inplace=True),

                        nn.ConvTranspose2d(64*8,64*4,4,2,1,bias=False),
                        nn.BatchNorm2d(64*4),
                        nn.ReLU(inplace=True),

                        nn.ConvTranspose2d(64*4,64*2,4,2,1,bias=False),
                        nn.BatchNorm2d(64*2),
                        nn.ReLU(inplace=True),

                        nn.ConvTranspose2d(64*2,64,4,2,1,bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),

                        nn.ConvTranspose2d(64,3,5,3,1,bias=False),
                        nn.Tanh()
                        )

        def forward(self,x):
                return self.net(x)

mygenerator = nn.DataParallel(Generator().cuda(),device_ids=[0,1,2,3])

#define your discriminator class
class Discriminator(nn.Module):
        def __init__(self):
                super(Discriminator,self).__init__()
                #Hin = (Hout-1)*stride-2*padding+kernel_size
                self.net = nn.Sequential(
                        nn.Conv2d(3,64,5,3,1,bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),

                        nn.Conv2d(64,64*2,4,2,1,bias=False),
                        nn.BatchNorm2d(64*2),
                        nn.ReLU(inplace=True),

                        nn.Conv2d(64*2,64*4,4,2,1,bias=False),
                        nn.BatchNorm2d(64*4),
                        nn.ReLU(inplace=True),

                        nn.Conv2d(64*4,64*8,4,2,1,bias=False),
                        nn.BatchNorm2d(64*8),
                        nn.ReLU(inplace=True),

                        nn.Conv2d(64*8,1,4,1,0,bias=False),
                        nn.Sigmoid())

        def forward(self,x):
                return self.net(x).view(-1)

mydiscriminator = nn.DataParallel(Discriminator().cuda(),device_ids=[0,1,2,3])

#train
def train(**kwargs):
        #parse your hyperparameters
        for k_,v_ in kwargs.items():
                setattr(opt,k_,v_)
        #load your data
        mydataset = MyDataset(opt.path)
        mydataloader = DataLoader(mydataset,batch_size=opt.batch_size,shuffle=True,num_workers=opt.num_workers,drop_last=True)
        #initialization of visualization
        vis = visdom.Visdom(env='szh')
        loss_gmeter = AverageValueMeter()
        loss_dmeter = AverageValueMeter()
        x_value = 0
        #true label、false label and noises
        true_labels = Variable(t.ones(opt.batch_size))
        false_labels = Variable(t.zeros(opt.batch_size))
        noises = Variable(t.randn(opt.batch_size,100,1,1))
        #define your optimizer and loss function
        generator_optimizer = t.optim.Adam(mygenerator.parameters(),lr=2e-4,betas=(0.5,0.999))
        discriminator_optimizer = t.optim.Adam(mydiscriminator.parameters(),lr=2e-4,betas=(0.5,0.999))
        criterion = nn.BCELoss()
        #use gpu
        if t.cuda.is_available:
                mygenerator.cuda()
                mydiscriminator.cuda()
                criterion.cuda()
                true_labels,false_labels = true_labels.cuda(),false_labels.cuda()
                noises = noises.cuda()
        #start training
        for i,epoch in enumerate(tqdm.tqdm(range(opt.max_epoch))):
                for ii,x in enumerate(tqdm.tqdm(mydataloader)):
                        #train discriminator every time
                        discriminator_optimizer.zero_grad()
                        output = mydiscriminator(Variable(x))
                        loss_real = criterion(output,true_labels)
                        loss_real.backward()
                        gen_img = mygenerator(Variable(t.randn(opt.batch_size,100,1,1).cuda()))
                        output = mydiscriminator(gen_img)
                        loss_false = criterion(output,false_labels)
                        loss_false.backward()
                        discriminator_optimizer.step()
                        loss = loss_real + loss_false
                        loss_dmeter.add(loss.data[0])

                        #train generator every five times
                        if ii%5==0:
                                generator_optimizer.zero_grad()
                                gen_img = mygenerator(Variable(t.randn(opt.batch_size,100,1,1).cuda()))
                                output = mydiscriminator(gen_img)
                                loss_ = criterion(output,true_labels)
                                loss_.backward()
                                generator_optimizer.step()
                                loss_gmeter.add(loss_.data[0])
                        if ii%20==0:
                                vis.line(Y=np.array([loss_gmeter.value()[0]]), X=np.array([x_value]),
                                    win=('g_loss'),
                                    opts=dict(title='g_loss'),
                                    update=None if x_value == 0 else 'append'
                                    )
                                vis.line(Y=np.array([loss_dmeter.value()[0]]), X=np.array([x_value]),
                                    win=('d_loss'),
                                    opts=dict(title='d_loss'),
                                    update=None if x_value == 0 else 'append'
                                    )
                                x_value += 1
                #visualize results every 20 epochs and save model
                if i%20 == 0:
                        vis.images(gen_img.data.cpu().numpy()[:64]*0.5+0.5,win='fake')
                        vis.images(x.cpu().numpy()[:64]*0.5+0.5,win='real')
                        t.save(mygenerator.state_dict(),'checkpoints/mygenerator_epoch%s.pth'%epoch)
                        t.save(mydiscriminator.state_dict(),'checkpoints/mydiscriminator_epoch%s.pth'%epoch)

def generate():
        vis = visdom.Visdom(env='szh')
        map_location = lambda storage,loc:storage
        #if you want to load the model then plus the sentence 'nn.DataParallel' otherwise an exception is thrown
        testgenerator = nn.DataParallel(Generator().eval().cuda(),device_ids=[0,1,2,3])
        testdiscriminator = nn.DataParallel(Discriminator().eval().cuda(),device_ids=[0,1,2,3])
        #load model state
        testgenerator.load_state_dict(t.load(opt.generator_model_path,map_location=map_location))
        testdiscriminator.load_state_dict(t.load(opt.discriminator_model_path,map_location=map_location))
        #test 100 noises
        noises = Variable(t.randn(100,100,1,1)).cuda()
        gen_img = testgenerator(noises)
        output = testdiscriminator(gen_img)
        #get top10 indexs
        indexs = output.data.topk(10)[1]
        results = []
        for index in indexs:
                results.append(gen_img.data[index])
        vis.images(t.stack(results).cpu().numpy()*0.5+0.5,win='fake')


if __name__ == '__main__':
        fire.Fire()


#how to run
'''
execute in current window:
python -m visdom.server
execute in another window if you want to train:
python mygan.py train --path=[your images path] --batch-size=[your bacth_size]
execute in another window if you want to test:
python mygan.py generate
'''
