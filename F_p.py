import torch
import torch.nn as nn
from torch.nn import init

class SimSiamPatchSampleF(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[], n_mlps=2, position=False, activation='relu',arch=2):
        # use the same patch_ids for multiple images in the batch
        super(SimSiamPatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        print('Use MLP: {}'.format(use_mlp))
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids
        self.n_mlps = n_mlps
        self.position = position
        self.activation = activation
        self.arch=arch

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            norm = get_norm_layer(1, 'batch')
            Activation = get_actvn_layer(self.activation)

            if self.arch == 2: # turn off bias 
                mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc, bias=False), norm(self.nc), Activation,
                                        nn.Linear(self.nc, self.nc, bias=False), norm(self.nc), Activation,
                                        nn.Linear(self.nc, self.nc, bias=False), norm(self.nc, affine=False)])
                pred = nn.Sequential(*[nn.Linear(self.nc, self.nc//4, bias=False), norm(self.nc//4), Activation,
                                        nn.Linear(self.nc//4, self.nc)])
                
            elif self.arch==1:
                mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), norm(self.nc), Activation,
                                        nn.Linear(self.nc, self.nc), norm(self.nc), Activation,
                                        nn.Linear(self.nc, self.nc), norm(self.nc)])
                pred = nn.Sequential(*[nn.Linear(self.nc, self.nc//4), norm(self.nc//4), Activation,
                                        nn.Linear(self.nc//4, self.nc)])
                

            if self.gpu_ids == "tpu":
                mlp.to("xla:1")
                pred.to("xla:1")
            else:
                if len(self.gpu_ids) > 0 :
                    mlp.cuda()
                    pred.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
            setattr(self, 'pred_%d' % mlp_id, pred)

            print('mlp_%d created, input nc %d'%(mlp_id, input_nc))
            print('pred_%d created, bottleneck nc %d'%(mlp_id, self.nc//4))

        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def sample_by_id(self, feat, feat_id, num_patches, patch_id= None, mask_i=None, verbose=False):
        ndims = len(feat.size()[2:])
        if num_patches > 0:
            if patch_id is not None:  # sample based on given index
                coords = patch_id
                # patch_id is a torch tensor (share idx across batch)
                if ndims == 3:
                    x_sample = feat[:,:,patch_id[:,0],patch_id[:,1],patch_id[:,2]]
                elif ndims == 2:
                    x_sample = feat[:,:,patch_id[:,0],patch_id[:,1]]
                else:
                    raise NotImplementedError
                if verbose:
                    print('Sample basd on given {} idx w/o mask: sample shape: {}'.format(len(patch_id), x_sample.size()))
            else: # sample patch index
                fg_coords = torch.where(mask_i > 0)
                if ndims == 3:
                    (_,_, fg_x, fg_y, fg_z) = fg_coords
                    #print('coords', fg_x.size(), fg_y.size(), fg_z.size())
                elif ndims == 2:
                    (_,_, fg_x, fg_y) = fg_coords
                else:
                    raise NotImplementedError

                patch_id = torch.randperm(fg_x.shape[0], device=feat.device)
                patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]

                select_x, select_y = fg_x[patch_id], fg_y[patch_id]
                if ndims == 3:
                    select_z = fg_z[patch_id]
                    coords = torch.cat((select_x.unsqueeze(1), select_y.unsqueeze(1), select_z.unsqueeze(1)), dim=1)
                    x_sample = feat[:, :, select_x, select_y, select_z]

                elif ndims == 2:
                    coords = torch.cat((select_x.unsqueeze(1), select_y.unsqueeze(1)), dim=1)
                    x_sample = feat[:, :, select_x, select_y]

                else:
                    raise NotImplementedError

                if verbose:
                    print('Masked sampling, patch_id: {} sample shape: {}'.format(len(patch_id), x_sample.size()))

        else:
            x_sample = feat
            coords = []

        tps, nc, nsample = x_sample.size()
        x_sample = x_sample.permute(0,2,1).flatten(0,1)  # tps*nsample, nc

        mlp = getattr(self, 'mlp_%d' % feat_id)
        pred = getattr(self, 'pred_%d' % feat_id)

        x_sample = mlp(x_sample)
        x_pred = pred(x_sample)
        x_sample = x_sample.view(tps, nsample, -1)
        x_pred = x_pred.view(tps, nsample, -1)
        if verbose:
            print('MLP + reshape: {}'.format(x_sample.size()))
            print('feature range ', feat_id, x_sample.min().item(), x_sample.max().item())
            print('\n\n')
        return x_sample, x_pred, coords


    def forward(self, feats, num_patches=64, patch_ids=None, mask=None, verbose=False):
        return_ids = []
        return_feats = []

        if verbose:
            print(f'Net F forward pass: # features: {len(feats)}')

        ndims = len(feats[0].size()[2:])
        if mask is not None:
            if verbose:
                print(f'Using foreground mask {mask.size()}')
            masks = [F.interpolate(mask, size=f.size()[2:], mode='nearest') for f in feats]
        else:
            masks = [torch.ones(f.size()[2:]).unsqueeze(0).unsqueeze(0) for f in feats] ## masks 变为 [1, 1, D, H, W]

        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats) ## 创建了mlp
        for feat_id, feat in enumerate(feats):
            if verbose:
                print(feat_id, 'input -> {}'.format(feat.size()))
            if num_patches > 0:
                if patch_ids is not None:  # sample based on given index
                    patch_id = patch_ids[feat_id]
                    # patch_id is a torch tensor (share idx across batch)
                    if ndims == 3:
                        x_sample = feat[:,:,patch_id[:,0],patch_id[:,1],patch_id[:,2]]
                    elif ndims == 2:
                        x_sample = feat[:,:,patch_id[:,0],patch_id[:,1]]
                    else:
                        raise NotImplementedError
                    if verbose:
                        print('Sample basd on given {} idx w/o mask: sample shape: {}'.format(len(patch_id), x_sample.size()))

                else: # sample patch index
                    mask_i = masks[feat_id]
                    fg_coords = torch.where(mask_i > 0) ## 坐标值，对应各个维度的长度，类似于grid函数
                    if ndims == 3:
                        (_,_, fg_x, fg_y, fg_z) = fg_coords
                    elif ndims == 2:
                        (_,_, fg_x, fg_y) = fg_coords
                    else:
                        raise NotImplementedError

                    patch_id = torch.randperm(fg_x.shape[0], device=feats[0].device)
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]

                    select_x, select_y = fg_x[patch_id], fg_y[patch_id]
                    if ndims == 3:
                        select_z = fg_z[patch_id]
                        coords = torch.cat((select_x.unsqueeze(1), select_y.unsqueeze(1), select_z.unsqueeze(1)), dim=1) ## coords.shape = [num_patches, 3]
                        x_sample = feat[:, :, select_x, select_y, select_z]

                    elif ndims == 2:
                        coords = torch.cat((select_x.unsqueeze(1), select_y.unsqueeze(1)), dim=1)
                        x_sample = feat[:, :, select_x, select_y]

                    else:
                        raise NotImplementedError

                    if verbose:
                        print('Masked sampling, patch_id: {} sample shape: {}'.format(len(patch_id), x_sample.size()))

            else:
                x_sample = feat
                coords = []

            tps, nc, nsample = x_sample.size() ## [2， 16， 128]
            x_sample = x_sample.permute(0,2,1).flatten(0,1)  # tps*nsample, nc
            #print(x_sample.size())
            return_ids.append(coords)


            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                pred = getattr(self, 'pred_%d' % feat_id)

                x_sample = mlp(x_sample) ## [2*128, 16] -> [2*128, 2048]
                x_pred = pred(x_sample) ## [2*128, 2048]
                #x_sample = self.l2norm(x_sample)  # moved l2-norm outside
                x_sample = x_sample.view(tps, nsample, -1)
                #x_pred = self.l2norm(x_pred)
                x_sample = x_sample.view(tps, nsample, -1)
                x_pred = x_pred.view(tps, nsample, -1)
            else:
                x_sample = x_sample.view(tps, nsample, -1)
                x_pred = x_sample.view(tps, nsample, -1)
                
            if verbose:
                print('MLP + reshape: {}'.format(x_sample.size()))
                print('feature range ', feat_id, x_sample.min().item(), x_sample.max().item())
                if self.position:
                    print('ff feature range ', position_enc.min(), position_enc.max())
                #print('patch id check', coords[:20])
                print('\n\n')
            return_feats.append((x_sample, x_pred))

        return return_feats, return_ids
    


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        #print('Input before normalization: ', x.min().item(), x.max().item())
        assert len(x.size()) == 2, 'wrong shape {} for L2-Norm'.format(x.size())
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out
    

def get_norm_layer(ndims, norm='batch'):
    if norm == 'batch':
        Norm = getattr(nn, 'BatchNorm%dd' % ndims)
    elif norm == 'instance':
        Norm = getattr(nn, 'InstanceNorm%dd' % ndims)
    elif norm == 'none':
        Norm = None
    else:
        assert 0, "Unsupported normalization: {}".format(norm)
    return Norm

def get_actvn_layer(activation='relu'):
    if activation == 'relu':
        Activation = nn.ReLU(inplace=True)
    elif activation == 'lrelu':
        Activation = nn.LeakyReLU(0.3, inplace=True)
    elif activation == 'elu':
        Activation = nn.ELU()
    elif activation == 'prelu':
        Activation = nn.PReLU()
    elif activation == 'selu':
        Activation = nn.SELU(inplace=True)
    elif activation == 'tanh':
        Activation = nn.Tanh()
    elif activation == 'none':
        Activation = None
    else:
        assert 0, "Unsupported activation: {}".format(activation)
    return Activation



def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if gpu_ids == "tpu":
        net.to("xla:1")
    else:
        if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())
            net.to("cuda")
            # if not amp:
            #net = torch.nn.DataParallel(net, "gpu_ids")  # multi-GPUs for non-AMP training
        #    net = torch.nn.DataParallel(net, "cuda")
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net


def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1 or classname.find('BatchNorm3d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>