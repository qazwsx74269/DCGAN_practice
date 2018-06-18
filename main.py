#coding:utf8
import os
import ipdb
import torch as t
import torchvision as tv
import tqdm
from model import NetG,NetD
from torch.autograd import Variable
from torch.meter import AverageValueMeter


class config(object):
	data_path = 'data/'
	num_workers = 4
	image_size = 96
	batch_size = 256
	max_epoch = 200
	lr1 = 2e-4
	lr2 = 2e-4
	beta1 = 0.5
	gpu = True
	nz = 100
	ngf = 64
	ndf = 64

	save_path = 'imgs/'

	vis = True
	env = 'GAN'
	plot_every = 20

	debug_file = '/tmp/debuggan'
	d_every = 1
	g_every = 5
	decay_every = 10
	netd_path = None
	netg_path = None

	gen_img = 'result.png'
	gen_num = 64
	gen_search_num = 512
	gen_mean = 0
	gen_std = 1

opt = Config()
def train(**kwargs):
	for k_,v_ in kwargs.items():
		setattr(opt,k_,v_)
	if opt.vis:
		from visualizer import Visualizer
		vis = Visualizer(opt.env)

	transforms = tv.transforms.Compose([
		tv.transforms.Scale(opt.image_size),
		tv.transforms.CenterCrop(opt.image_size),
		tv.transforms.ToTensor(),
		tv.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
		])

	dataset = tv.datasets.ImageFolder(opt.data_path,transform=transforms)
	dataloader = t.utils.data.DataLoader(dataset,
		batch_size = opt.batch_size,
		shuffle = True,
		num_workers = opt.num_workers,
		drop_last = True
		)

	netg,netd = NetG(opt),NetD(opt)
	map_location = lambda storage, loc:storage
	if opt.netd_path:
		netd.load_state_dict(t.load(opt.netd_path,map_location = map_location))
	if opt.netg_path:
		netg.load_state_dict(t.load(opt.netg_path,map_location = map_location))

	optimizer_g = t.optim.Adam(netg.parameters(),opt.lr1,betas=(opt.beta1,0.999))
	optimizer_d = t.optim.Adam(netd.parameters(),opt.lr2,betas=(opt.beta1,0.999))
	criterion = t.nn.BCELoss()

	true_labels = Variable(t.ones(opt.batch_size))
	fake_labels = Variable(t.zeros(opt.batch_size))
	fix_noises = Variable(t.randn(opt.batch_size,opt.nz,1,1))
	noises = Variable(t.randn(opt.batch_size,opt.nz,1,1))

	errord_meter = AverageValueMeter()
	errorg_meter = AverageValueMeter()

	if opt.gpu:
		netd.cuda()
		netg.cuda()
		criterion.cuda()
		true_labels,fake_labels = true_labels.cuda(),fake_labels.cuda()
		fix_noises,noises = fix_noises.cuda(),noises.cuda()

	epochs = range(opt.max_epoch)
	for epoch in iter(epochs):
		for ii,(img,_) in tqdm.tqdm(enumerate(dataloader)):
			real_img = Variable(img)
			if opt.gpu:
				real_img = real_img.cuda()
			if ii%opt.d_every==0:
				optimizer_d.zero_grad()
				output = netd(real_img)
				error_d_real = criterion(output,true_labels)
				error_d_real.backward()

				noises.data.copy_(t.randn(opt.batch_size,opt.nz,1,1))
				fake_img = netg(noises).detach()
				output = netd(fake_img)
				error_d_fake = criterion(output,fake_labels)
				error_d_fake.backward()
				optimizer_d.step()

				error_d = error_d_fake + error_d_real

				error_meter.add(error_d.data[0])

			if ii%opt.g_every==0:
				optimizer_g.zero_grad()
				noises.data.copy_(t.randn(opt.batch_size,opt.nz,1,1))
				fake_img = netg(noises)
				output = netd(fake_img).detach()
				error_g = criterion(output,true_labels)
				error_g.backward()
				optimizer_g.step()
				errorg_meter.add(error_g.data[0])

			if opt.vis and ii%opt.plot_every == opt.plot_every-1:
				if os.path.exists(opt.debug_file):
					ipdb.set_trace()
				fix_fake_imgs = netg(fix_noises)
				vis.images(fix_fake_imgs.data.cpu().numpy()[:64]*0.5+0.5,win='fixfake')
				vis.images(real_img.data.cpu().numpy()[:64]*0.5+0.5,win='real')
				vis.plot('errord',errord_meter.value()[0])
				vis.plot('errorg',errorg_meter.value()[0])

		if epoch%opt.decay_every==0:
			tv.utils.save_image(fix_fake_imgs.data[:64],'%s/%s.png'%(opt.save_path,epoch),Normalize=True,range=(-1,1))
			t.save(netd.state_dict(),'checkpoints/netd_%s.pth'%epoch)
			t.save(netg.state_dict(),'checkpoints/netg_%s.pth'%epoch)
			errord_meter.reset()
			errorg_meter.reset()
			
def generate(**kwargs):
	for k_,v_ in kwargs.items():
		setattr(opt,k_,v_)

	netg,netd = NetG(opt).eval(),NetD(opt).eval()
	noises = t.randn(opt.gen_search_num,opt.nz,1,1).normal_(opt.gen_mean,opt.gen_std)
	noises = Variable(noises,volatile=True)

	map_location = lambda storage, loc:storage
	netd.load_state_dict(t.load(opt.netd_path,map_location=map_location))
	netg.load_state_dict(t.load(opt.netg_path,map_location=map_location))

	if opt.gpu:
		netd.cuda()
		netg.cuda()
		noises = noises.cuda()

	fake_img = netg(noises)
	scores = netd(fake_img).data
	ipdb.set_trace()
	indexs = scores.topk(opt.gen_num)[1]
	result = []
	for ii in indexs:
		result.append(fake_img.data[ii])

	tv.utils.save_image(t.stack(result),opt.gen_img,normalize=True,range=(-1,1))

if __name__ == '__main__':
	import fire
	fire.Fire()