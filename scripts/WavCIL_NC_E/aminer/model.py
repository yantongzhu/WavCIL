import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv
import numpy as np

from sild.utils import DummyArgs
from torch.fft import rfft, irfft, fft, ifft

from sklearn.metrics import f1_score, accuracy_score
from sild.utils import EarlyStopping, move_to, cal_metric
from sild.utils import COLLECTOR
from tqdm import tqdm, trange
from utils import *


class SpecMask(nn.Module):
    def __init__(self, hid_dim, temporature, K_len) -> None:
        super().__init__()
        self.K_len = K_len
        self.node_spec_map = nn.Sequential(nn.Linear(self.K_len * hid_dim * 2, hid_dim * 2), nn.ReLU(),
                                           nn.Linear(hid_dim * 2, self.K_len))                          
        self.temporature = temporature
        self.K_len = K_len
        self.hid_dim = hid_dim
        self.spec_lin = nn.Sequential(
            nn.Linear(K_len * hid_dim * 2, hid_dim * 2),
            nn.ReLU(),
            nn.Linear(hid_dim * 2, hid_dim)
        )

    def forward(self, specs):
                         
                                         
        spec_real, spec_imag = specs.real, specs.imag             
        spec_real_imag = torch.stack([spec_real, spec_imag], dim=-1)                
        node_choice = self.node_spec_map(
            spec_real_imag.view(-1, self.K_len * self.hid_dim * 2))                            

        cmask_ = torch.sigmoid(node_choice / self.temporature)
        smask_ = torch.sigmoid(- node_choice / self.temporature)          
        if len(COLLECTOR.cache.get('cmask0', [])) == len(COLLECTOR.cache.get('loss', [])):
            COLLECTOR.add('cmask0', seq2str(cmask_[0].detach().cpu().numpy()))
            COLLECTOR.add('smask0', seq2str(smask_[0].detach().cpu().numpy()))

        cmask = cmask_.unsqueeze(-1).expand_as(spec_imag)
        smask = smask_.unsqueeze(-1).expand_as(spec_imag)

                                       
        c_spec_real = spec_real * cmask                         
        c_spec_imag = spec_imag * cmask                         

        s_spec_real = spec_real * smask                         
        s_spec_imag = spec_imag * smask                         

        c_spec = torch.cat([c_spec_real, c_spec_imag], dim=-1).flatten(-2, -1)
        s_spec = torch.cat([s_spec_real, s_spec_imag], dim=-1).flatten(-2, -1)

        c_spec = self.spec_lin(c_spec)          
        s_spec = self.spec_lin(s_spec)          

        return c_spec, s_spec


class DiscreteComplexWaveletMask(nn.Module):
    def __init__(self, hid_dim, temporature, K_len, num_scales=8, omega0=6.0):
        super().__init__()
        self.hid_dim = hid_dim
        self.temporature = temporature
        self.K_len = K_len
        self.num_scales = num_scales
        self.omega0 = omega0
        self.node_spec_map = nn.Sequential(
            nn.Linear(self.K_len * hid_dim * 2, hid_dim * 2),
            nn.ReLU(),
            nn.Linear(hid_dim * 2, self.K_len)
        )
        self.spec_lin = nn.Sequential(
            nn.Linear(K_len * hid_dim * 2, hid_dim * 2),
            nn.ReLU(),
            nn.Linear(hid_dim * 2, hid_dim)
        )
        self.scale_node_spec_map = nn.Sequential(
            nn.Linear(self.num_scales * hid_dim * 2, hid_dim * 2),
            nn.ReLU(),
            nn.Linear(hid_dim * 2, self.num_scales)
        )
        self.scale_spec_lin = nn.Sequential(
            nn.Linear(self.num_scales * hid_dim * 2, hid_dim * 2),
            nn.ReLU(),
            nn.Linear(hid_dim * 2, hid_dim)
        )
        self._cached_wavelets = {}
        self._cached_T = None

    def discrete_complex_morlet(self, T, scale, omega0=6.0):
        if T <= 0:
            raise ValueError(f"bad T {T}")
        if scale <= 0:
            scale = max(scale, 0.1)
        t = torch.arange(T, dtype=torch.float32, device=next(self.parameters()).device)
        t = t - T // 2
        sigma = scale / omega0
        sigma = max(sigma, 0.1)
        gaussian = torch.exp(-0.5 * (t / sigma) ** 2)
        complex_exp = torch.exp(1j * omega0 * t / scale)
        wavelet = gaussian * complex_exp
        norm_factor = torch.sqrt(torch.sum(torch.abs(wavelet) ** 2))
        if norm_factor > 1e-8:
            wavelet = wavelet / norm_factor
        else:
            wavelet = torch.ones_like(wavelet) / math.sqrt(T)
        return wavelet

    def _precompute_wavelets(self, T, device):
        if self._cached_T == T and len(self._cached_wavelets) > 0:
            return
        self._cached_T = T
        self._cached_wavelets.clear()
        if T < 4:
            scales = torch.linspace(1.0, max(2.0, T / 2), self.num_scales, device=device)
        else:
            max_scale = max(T // 4, 2)
            scales = torch.logspace(0, math.log10(max_scale), self.num_scales, device=device)
        wavelets_fft = []
        for scale in scales:
            wavelet = self.discrete_complex_morlet(T, scale.item(), self.omega0)
            wavelet_fft = torch.fft.fft(wavelet.conj().flip(dims=[0]), n=T, norm='ortho')
            wavelets_fft.append(wavelet_fft)
        self._cached_wavelets['fft'] = torch.stack(wavelets_fft).to(device)

    def discrete_cwt(self, signal):
        N, T, d = signal.shape
        device = signal.device
        self._precompute_wavelets(T, device)
        signal_fft = torch.fft.fft(signal, n=T, dim=1, norm='ortho')
        wavelets_fft = self._cached_wavelets['fft']
        signal_fft_expanded = signal_fft.unsqueeze(1)
        wavelets_fft_expanded = wavelets_fft.unsqueeze(0).unsqueeze(-1)
        conv_fft = signal_fft_expanded * wavelets_fft_expanded
        conv_result = torch.fft.ifft(conv_fft, dim=2, norm='ortho')
        W = 3 if T >= 3 else T
        cwt_coeffs = conv_result[:, :, T - W:, :].mean(dim=2)
        if self.num_scales != self.K_len:
            cwt_coeffs_real = cwt_coeffs.real.transpose(1, 2)
            cwt_coeffs_imag = cwt_coeffs.imag.transpose(1, 2)
            cwt_coeffs_real = F.interpolate(cwt_coeffs_real, size=self.K_len, mode='linear',
                                            align_corners=False).transpose(1, 2)
            cwt_coeffs_imag = F.interpolate(cwt_coeffs_imag, size=self.K_len, mode='linear',
                                            align_corners=False).transpose(1, 2)
            cwt_coeffs = torch.complex(cwt_coeffs_real, cwt_coeffs_imag)
        return cwt_coeffs

    def discrete_cwt_full(self, signal):
        N, T, d = signal.shape
        device = signal.device
        self._precompute_wavelets(T, device)
        signal_fft = torch.fft.fft(signal, n=T, dim=1, norm='ortho')
        wavelets_fft = self._cached_wavelets['fft']
        signal_fft_expanded = signal_fft.unsqueeze(1)
        wavelets_fft_expanded = wavelets_fft.unsqueeze(0).unsqueeze(-1)
        conv_fft = signal_fft_expanded * wavelets_fft_expanded
        conv_result = torch.fft.ifft(conv_fft, dim=2, norm='ortho')
        W = 3 if T >= 3 else T
        cwt_coeffs = conv_result[:, :, T - W:, :].mean(dim=2)
        if self.num_scales != self.K_len:
            cwt_coeffs_real = cwt_coeffs.real.transpose(1, 2)
            cwt_coeffs_imag = cwt_coeffs.imag.transpose(1, 2)
            cwt_coeffs_real = F.interpolate(cwt_coeffs_real, size=self.K_len, mode='linear',
                                            align_corners=False).transpose(1, 2)
            cwt_coeffs_imag = F.interpolate(cwt_coeffs_imag, size=self.K_len, mode='linear',
                                            align_corners=False).transpose(1, 2)
            cwt_coeffs = torch.complex(cwt_coeffs_real, cwt_coeffs_imag)
        return cwt_coeffs, conv_fft

    def forward(self, z):
        specs = self.discrete_cwt(z)
        spec_real, spec_imag = specs.real, specs.imag
        spec_real_imag = torch.stack([spec_real, spec_imag], dim=-1)
        node_choice = self.node_spec_map(spec_real_imag.view(-1, self.K_len * self.hid_dim * 2))
        cmask_ = torch.sigmoid(node_choice / self.temporature)
        smask_ = torch.sigmoid(-node_choice / self.temporature)
        cmask = cmask_.unsqueeze(-1).expand_as(spec_imag)
        smask = smask_.unsqueeze(-1).expand_as(spec_imag)
        c_spec_real = spec_real * cmask
        c_spec_imag = spec_imag * cmask
        s_spec_real = spec_real * smask
        s_spec_imag = spec_imag * smask
        c_spec = torch.cat([c_spec_real, c_spec_imag], dim=-1).flatten(-2, -1)
        s_spec = torch.cat([s_spec_real, s_spec_imag], dim=-1).flatten(-2, -1)
        c_spec = self.spec_lin(c_spec)
        s_spec = self.spec_lin(s_spec)
        return c_spec, s_spec

    def forward_time_reconstruct(self, z):
        N, T, d = z.shape
        spec_len_eff = max(1, 2 * (self.K_len - 1))
        T_eff = min(T, spec_len_eff)
        z = z[:, T - T_eff:, :]
        _, conv_fft = self.discrete_cwt_full(z)
        conv_result = torch.fft.ifft(conv_fft, dim=2, norm='ortho')
        W = 3 if T_eff >= 3 else T_eff
        scale_coeffs = conv_result[:, :, T_eff - W:, :].mean(dim=2)
        sc_real, sc_imag = scale_coeffs.real, scale_coeffs.imag
        sc_real_imag = torch.stack([sc_real, sc_imag], dim=-1)
        node_choice_scale = self.scale_node_spec_map(sc_real_imag.view(-1, self.num_scales * self.hid_dim * 2))
        cmask_ = torch.sigmoid(node_choice_scale / self.temporature)
        smask_ = torch.sigmoid(-node_choice_scale / self.temporature)
        cmask = cmask_.unsqueeze(-1).expand_as(sc_real)
        smask = smask_.unsqueeze(-1).expand_as(sc_real)
        cmask_exp = cmask.unsqueeze(2)
        smask_exp = smask.unsqueeze(2)
        c_conv_fft = conv_fft * cmask_exp
        s_conv_fft = conv_fft * smask_exp
        c_out_time = torch.fft.ifft(c_conv_fft, dim=2, norm='ortho').real.mean(dim=1)
        s_out_time = torch.fft.ifft(s_conv_fft, dim=2, norm='ortho').real.mean(dim=1)
        return c_out_time, s_out_time

    def forward_scale_compress(self, z):
        N, T, d = z.shape
        spec_len_eff = max(1, 2 * (self.K_len - 1))
        T_eff = min(T, spec_len_eff)
        z = z[:, T - T_eff:, :]
        _, conv_fft = self.discrete_cwt_full(z)
        conv_result = torch.fft.ifft(conv_fft, dim=2, norm='ortho')
        W = 3 if T_eff >= 3 else T_eff
        scale_coeffs = conv_result[:, :, T_eff - W:, :].mean(dim=2)
        sc_real, sc_imag = scale_coeffs.real, scale_coeffs.imag
        sc_real_imag = torch.stack([sc_real, sc_imag], dim=-1)
        node_choice_scale = self.scale_node_spec_map(sc_real_imag.view(-1, self.num_scales * self.hid_dim * 2))
        cmask_ = torch.sigmoid(node_choice_scale / self.temporature)
        smask_ = torch.sigmoid(-node_choice_scale / self.temporature)
        cmask = cmask_.unsqueeze(-1).expand_as(sc_real)
        smask = smask_.unsqueeze(-1).expand_as(sc_real)
        c_real = sc_real * cmask
        c_imag = sc_imag * cmask
        s_real = sc_real * smask
        s_imag = sc_imag * smask
        c_feat = torch.cat([c_real, c_imag], dim=-1).flatten(-2, -1)
        s_feat = torch.cat([s_real, s_imag], dim=-1).flatten(-2, -1)
        c_out = self.scale_spec_lin(c_feat)
        s_out = self.scale_spec_lin(s_feat)
        return c_out, s_out


class SGCNNet(torch.nn.Module):
    def __init__(self, args, data):
        super(SGCNNet, self).__init__()
        in_dim = args.nfeat
        hid_dim = 2 * args.nhid
        out_dim = 2 * args.nhid
        num_layers = args.n_layers
        time_length = args.length
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.time_length = time_length
        self.spec_len = time_length
        self.K_len = 1 + self.spec_len // 2
        self.args = args
        self.len_train = self.time_length - args.testlength - args.vallength
        self.len_val = args.vallength
        self.len_test = args.testlength
        self.len = time_length

                 
        x = data['x']
        self.x = [x for _ in range(self.time_length)] if len(x.shape) <= 2 else x

                      
        if args.backbone in 'GCN GIN GAT'.split():
            args.static_conv = args.backbone
            self.backbone = SpaSeqNetLast(args)
        else:
            raise NotImplementedError()

                        
        if args.spec_filt == "mask":
            self.spec_filt = SpecMask(hid_dim, args.temporature, self.K_len)
        elif args.spec_filt == 'dcwt':
            num_scales = getattr(args, 'dcwt_scales', 8)
            omega0 = getattr(args, 'dcwt_omega0', 6.0)
            self.spec_filt = DiscreteComplexWaveletMask(hid_dim, args.temporature, self.K_len, num_scales, omega0)
        else:
            raise NotImplementedError()

                  
        self.post_gnn = args.post_gnn
                           
                                                                                               

                 
        self.cs_decoder = self.backbone.cs_decoder
        self.ss_decoder = self.backbone.ss_decoder

        self.ctype = args.ctype

               
        self.optim = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def cal_loss(self, zs, data):
        cs, ss = zs                                             
        ss = ss.detach()
        args = self.args
        device = args.device
        intervention_times, la = args.intervene_times, args.intervene_lambda
        optimizer = self.optim

        def cal_y(embeddings, decoder, node_masks):
            preds = torch.tensor([]).to(device)
            for t in range(self.len_train):
                z = embeddings[t]
                mask = node_masks[t]
                pred = decoder(z)[mask]
                preds = torch.cat([preds, pred])
            return preds

        node_masks = data['node_masks']
        node_labels = torch.LongTensor([]).to(device)
        for t in range(self.len_train):
            label = data['y'][node_masks[t]].squeeze().to(args.device)
            node_labels = torch.cat([node_labels, label])

        criterion = torch.nn.CrossEntropyLoss()
        cy = cal_y(cs, self.cs_decoder, node_masks)         
        sy = cal_y(ss, self.ss_decoder, node_masks)         

        conf_loss = criterion(sy, node_labels)
        COLLECTOR.add(key='conf_loss', value=conf_loss.item())
        causal_loss = criterion(cy, node_labels)
        COLLECTOR.add(key='causal_loss', value=causal_loss.item())

        if self.args.intervene_times <= 0 or self.args.intervene_lambda <= 0:
            loss = causal_loss
        else:
            env_loss = torch.tensor([]).to(device)
            for i in range(intervention_times):
                s1 = np.random.randint(len(sy))
                s = torch.sigmoid(sy[s1]).detach()
                conf = s * cy
                env_loss = torch.cat([env_loss, criterion(conf, node_labels).unsqueeze(0)])
            env_mean = env_loss.mean()
            env_var = torch.var(env_loss * intervention_times)
            penalty = env_mean + env_var
            loss = causal_loss + la * penalty
            COLLECTOR.add('env_mean', env_mean.item())
            COLLECTOR.add('env_var', env_var.item())

        loss = loss + conf_loss

        return loss

    def spectral_filter(self, z):
        if not self.args.use_filt:
            return [z[-1], z[-1]]
                     
        ctype = self.ctype
        time_len = z.shape[0]
                                          
                                                      
                                                                            
         
                                           
                                                          
        if self.args.spec_filt == 'dcwt':
            z = torch.permute(z, (1, 0, 2))
            c_spec, s_spec = self.spec_filt.forward_scale_compress(z)
            out = [c_spec, s_spec]
            return out
        z = torch.permute(z, (1, 0, 2))
        specs = rfft(z, n=self.spec_len, norm="ortho", dim=1)
        c_spec, s_spec = self.spec_filt(specs)
        if self.post_gnn:
            c_spec = self.nodeconv(c_spec)
            s_spec = self.nodeconv(s_spec)
        out = [c_spec, s_spec]
        return out

    def get_final_emb(self, edge_list):
        cs, ss = [], []
        for t in range(len(edge_list)):
            e = edge_list[:t + 1]
            x = self.x[:t + 1]

            x_list, _, _ = self.backbone(e, x)
            cz, sz = self.spectral_filter(x_list)          
            cs.append(cz)
            ss.append(sz)
        cs = torch.stack(cs, dim=0)             
        ss = torch.stack(ss, dim=0)             
        return cs, ss

    def train_epoch(self, data):
        self.train()

        cz, sz = self.get_final_emb(data['edge_index'][:self.len_train])
        loss = self.cal_loss([cz, sz], data)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item()

    @torch.no_grad()
    def test_epoch(self, data):
        model = self
        model.eval()

        cz, sz = self.get_final_emb(data['edge_index'])
        embeddings = cz

        auc_list = []
        for t in range(self.len):
            node_mask = data['node_masks'][t]
            preds = model.cs_decoder(embeddings[t])
            preds = preds[node_mask]
            preds = preds.argmax(dim=-1).squeeze()

            target = data['y'][node_mask].squeeze()
            auc = cal_metric(preds, target, self.args)
            auc_list.append(auc)
        train = np.mean(auc_list[:self.len_train])
        val = np.mean(auc_list[self.len_train:self.len_train + self.len_val])
        test = np.mean(auc_list[self.len_train + self.len_val:])

        COLLECTOR.add(key='auc_list', value=auc_list)

        return train, val, test


