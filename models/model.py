import torch
import torch.nn as nn
import torch.nn.functional as F

# 1D conv usage:
# batch_size (N) = #3D objects , channels = features, signal_length (L) (convolution dimension) = #point samples
# kernel_size = 1 i.e. every convolution over only all features of one point sample
# 3D Single View Reconsturction (for 256**3 input voxelization) --------------------------------------
# ----------------------------------------------------------------------------------------------------

class NDF(nn.Module):

    def __init__(self, hidden_dim=256):
        super(NDF, self).__init__()

        self.conv_in = nn.Conv3d(1, 16, 3, padding=1, padding_mode='border')  # out: 32 ->m.p. 16
        self.conv_0 = nn.Conv3d(16, 32, 3, padding=1, padding_mode='border')  # out: 16
        self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1, padding_mode='border')  # out: 16 ->m.p. 8
        self.conv_1 = nn.Conv3d(32, 64, 3, padding=1, padding_mode='border')  # out: 8
        self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1, padding_mode='border')  # out:8 -> mp 4
        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1, padding_mode='border')  # out: 4
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='border')  # out: 4 -> mp 1
        feature_size = 1690
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim * 2, 1)
        self.fc_1 = nn.Conv1d(hidden_dim *2, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim , hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)
        self.actvn = nn.ReLU()
        self.maxpool = nn.MaxPool3d(2)
        self.conv_in_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(32)
        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        displacment = 0.0722
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)
        self.displacments = torch.Tensor(displacments).cuda()
        self.latent_dim = 128
        self.displacments = torch.Tensor(displacments).cuda()
        self.num_codebook_vectors = 2048*3
        self.beta = 0.8
        self.embedding_1 = nn.Embedding(self.num_codebook_vectors, self.latent_dim).cuda()
        self.embedding_1.weight.data.uniform_(0, 1).cuda()

    def encoder(self,x):
        x = x.unsqueeze(1)
        f_0 = x
        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        f_1 = net
        net = self.maxpool(net)  # out 128
        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        f_2 = net
        net = self.maxpool(net)  # out 64
        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        f_3 = net
        net = self.maxpool(net)
        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        f_4 = net
        f_0 = torch.nn.functional.interpolate(f_0, size=(8,8,8), mode="trilinear",align_corners =True)
        f_1 = torch.nn.functional.interpolate(f_1, size = (8,8,8),mode = "trilinear",align_corners=True)
        f_2 = torch.nn.functional.interpolate(f_2, size=(8,8,8), mode="trilinear", align_corners=True)
        f_3 = torch.nn.functional.interpolate(f_3, size=(8,8,8), mode="trilinear", align_corners=True)
        f_4 = torch.nn.functional.interpolate(f_4, size=(8,8,8), mode="trilinear", align_corners=True)
        z = torch.cat((f_0,f_1,f_2,f_3,f_4),dim=1)    ###(3,497,8,8,8)
        z_flattened = z.contiguous().view(-1, self.latent_dim)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
           torch.sum(self.embedding_1.weight ** 2, dim=1) - \
           2 * (torch.matmul(z_flattened, self.embedding_1.weight.t()))
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding_1(min_encoding_indices).view(z.shape)
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        z_q = z + (z_q - z).detach()
        return z_q, min_encoding_indices, loss

    def decoder(self, p,z):
        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=2)
        z_q_0 = z[:,0:1,:,:,:].view(z.shape[0],1,8,8,8)
        z_q_1 = z[:,1:1+16,:,:,:].view(z.shape[0],16,8,8,8)
        z_q_2 = z[:,1+16:1+16+32,:,:,:].view(z.shape[0],32,8,8,8)
        z_q_3 = z[:, 1+16+32:1+16+32+64, :, :, :].view(z.shape[0], 64, 8,8,8)
        z_q_4 = z[:, 1+16+32+64:1+16+32+64+128, :, :, :].view(z.shape[0], 128, 8,8,8)
        feature_0 = F.grid_sample(z_q_0, p, padding_mode='border', align_corners=True)
        feature_1 = F.grid_sample(z_q_1, p, padding_mode='border', align_corners=True)
        feature_2 = F.grid_sample(z_q_2, p, padding_mode='border', align_corners=True)
        feature_3 = F.grid_sample(z_q_3, p, padding_mode='border', align_corners=True)
        feature_4 = F.grid_sample(z_q_4, p, padding_mode='border', align_corners=True)

        # here every channel corresponds to one feature.
        features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4),
                            dim=1)  # (B, features, 1,7,sample_num)
        shape = features.shape
        features = torch.reshape(features,
                                 (shape[0], shape[1] * shape[3], shape[4]))  # (B, features_per_sample, samples_num)
        features = torch.cat((features, p_features), dim=1)  # (B, feature_size, samples_num)
        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = self.actvn(self.fc_out(net))
        out = net.squeeze(1)
        return  out

    def forward(self, p, x):
        encoding,min_indices,loss = self.encoder(x)
        out = self.decoder(p, encoding)
        return out,min_indices,loss

#T = torch.rand(3,32,32,32).cuda()
#p = torch.rand(3,5000,3).cuda()
#N = NDF().cuda()
#encoding,m,l = N.encoder(T)
#z = N.decoder(p,encoding)
