import torch
import torch.nn  as nn
import tensorly
tensorly.set_backend('pytorch')

class _Residual_Block(nn.Module):
    def __init__(self, inc=64, outc=64, groups=1, scale=1.0):
        super(_Residual_Block, self).__init__()

        midc = int(outc * scale)

        if inc is not outc:
            self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0,
                                         groups=1, bias=False)
        else:
            self.conv_expand = None

        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=midc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(midc)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=midc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.conv_expand is not None:
            identity_data = self.conv_expand(x)
        else:
            identity_data = x

        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.relu2(self.bn2(torch.add(output, identity_data)))
        return output

class conv_net_classifier(nn.Module):
    def __init__(self, cdim=1, output=10, channels=[64, 128, 256, 512, 512, 512], image_size=32,transform=torch.tanh,channels_fc=[]):
        super(conv_net_classifier, self).__init__()

        assert (2 ** len(channels)) * 4 == image_size

        self.output = output
        cc = channels[0]
        self.main = nn.Sequential(
            nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
            nn.BatchNorm2d(cc),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),
        )

        sz = image_size // 2
        for ch in channels[1:]:
            self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.main.add_module('down_to_{}'.format(sz // 2), nn.AvgPool2d(2))
            cc, sz = ch, sz // 2

        self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))
        # self.fc = nn.Linear((cc) * 4 * 4, self.output)
        self.fc = feature_map(d_in_x=(cc) * 4 * 4,layers_x=channels_fc,output_dim=self.output,transformation=transform,cat_size_list=[])


    def forward(self, x):
        y = self.main(x).reshape(x.size(0), -1)
        y = self.fc(y)
        return y

class conv_net_classifier_kernel(nn.Module):
    def __init__(self,k,Z, cdim=1, output=10, channels=[64, 128, 256, 512, 512, 512], image_size=32,transform=torch.tanh,channels_fc=[]):
        super(conv_net_classifier_kernel, self).__init__()
        self.k = k
        print(self.k)
        self.register_buffer('Z',Z)
        self.m = self.Z.shape[0]
        self.output_tensor = torch.nn.Parameter(torch.randn(self.m,self.m,output))


        assert (2 ** len(channels)) * 4 == image_size

        self.output = output
        cc = channels[0]
        self.main = nn.Sequential(
            nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
            nn.BatchNorm2d(cc),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),
        )

        sz = image_size // 2
        for ch in channels[1:]:
            self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.main.add_module('down_to_{}'.format(sz // 2), nn.AvgPool2d(2))
            cc, sz = ch, sz // 2

        self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))
        # self.fc = nn.Linear((cc) * 4 * 4, self.output)
        self.fc = feature_map(d_in_x=(cc) * 4 * 4,layers_x=channels_fc,output_dim=self.m,transformation=transform,cat_size_list=[])


    def forward(self, x):
        y = self.main(x).reshape(x.size(0), -1)
        y = self.fc(y)
        y  =tensorly.tenalg.mode_dot(self.output_tensor,y,0)
        kz = self.k(x.flatten(1),self.Z).evaluate().unsqueeze(-1)
        output = kz*y
        return output.sum(1)



class multi_input_Sequential(torch.nn.Sequential):
    def forward(self, inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class multi_input_Sequential_res_net(torch.nn.Sequential):
    def forward(self, inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                output = module(inputs)
                if inputs.shape[1]==output.shape[1]:
                    inputs = inputs+output
                else:
                    inputs = output
        return inputs

class nn_node(torch.nn.Module): #Add dropout layers, Do embedding layer as well!
    def __init__(self,d_in,d_out,cat_size_list,transformation=torch.tanh):
        super(nn_node, self).__init__()

        self.has_cat = len(cat_size_list)>0
        self.latent_col_list = []
        print('cat_size_list',cat_size_list)
        for i,el in enumerate(cat_size_list):
            col_size = el//2+2
            setattr(self,f'embedding_{i}',torch.nn.Embedding(el,col_size))
            self.latent_col_list.append(col_size)
        self.w = torch.nn.Linear(d_in+sum(self.latent_col_list),d_out)
        self.f = transformation
        # self.dropout = torch.nn.Dropout(dropout)
        self.bn = torch.nn.BatchNorm1d(d_out)

    def forward(self,X,x_cat=[]):
        if not isinstance(x_cat,list):
            seq = torch.unbind(x_cat,1)
            cat_vals = [X]
            for i,f in enumerate(seq):
                o = getattr(self,f'embedding_{i}')(f)
                cat_vals.append(o)
            X = torch.cat(cat_vals,dim=1)
        return self.bn(self.f(self.w(X)))

class feature_map(torch.nn.Module):
    def __init__(self,
                 d_in_x,
                 cat_size_list,
                 layers_x,
                 transformation=torch.tanh,
                 output_dim=10,
                 ):
        super(feature_map, self).__init__()
        self.output_dim=output_dim
        self.init_covariate_net(d_in_x,layers_x,cat_size_list,transformation,output_dim)

    def identity_transform(self, x):
        return x

    def init_covariate_net(self,d_in_x,layers_x,cat_size_list,transformation,output_dim):
        module_list = [nn_node(d_in=d_in_x,d_out=layers_x[0],cat_size_list=cat_size_list,transformation=transformation)]
        for l_i in range(1,len(layers_x)):
            module_list.append(nn_node(d_in=layers_x[l_i-1],d_out=layers_x[l_i],cat_size_list=[],transformation=transformation))
        self.covariate_net = multi_input_Sequential_res_net(*module_list)
        self.final_layer = torch.nn.Linear(layers_x[-1],output_dim)

    def forward(self,x_cov,x_cat=[]):
        return self.final_layer(self.covariate_net((x_cov,x_cat)))


class kernel_feature_map_regression(torch.nn.Module):
    def __init__(self,
                 k,
                 Z,
                 d_in_x,
                 cat_size_list,
                 layers_x,
                 transformation=torch.tanh,
                 output_dim=10,
                 ):
        super(kernel_feature_map_regression, self).__init__()
        self.k = k
        self.output_dim = output_dim
        self.register_buffer('Z',Z)

        self.init_covariate_net(d_in_x, layers_x, cat_size_list, transformation, self.Z.shape[0])
    def identity_transform(self, x):
        return x

    def init_covariate_net(self, d_in_x, layers_x, cat_size_list, transformation, output_dim):
        module_list = [
            nn_node(d_in=d_in_x, d_out=layers_x[0], cat_size_list=cat_size_list, transformation=transformation)]
        for l_i in range(1, len(layers_x)):
            module_list.append(
                nn_node(d_in=layers_x[l_i - 1], d_out=layers_x[l_i], cat_size_list=[], transformation=transformation))
        self.covariate_net = multi_input_Sequential_res_net(*module_list)
        self.final_layer = torch.nn.Linear(layers_x[-1], output_dim)

    def forward(self, x_cov, x_cat=[]):
        feature_map = self.final_layer(self.covariate_net((x_cov, x_cat)))
        kernel_weights = self.k(x_cov.flatten(1),self.Z).evaluate()
        output = feature_map*kernel_weights
        return output.sum(1)

