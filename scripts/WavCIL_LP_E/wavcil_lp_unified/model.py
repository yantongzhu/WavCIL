import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv
import numpy as np
from sild.utils import DummyArgs
from torch.fft import rfft, irfft, fft, ifft
import math
from DIDA.utils.loss import EnvLoss
from utils import *
from sild.utils import COLLECTOR
from wavcil_backbone import DGNN


class DiscreteComplexWaveletMask(nn.Module):
    """
    Discrete Complex Wavelet Transform (DCWT) with an adaptive dual-mask in the wavelet
    coefficient domain, producing causal and spurious representations.
    """

    def __init__(self, hid_dim, temporature, K_len, num_scales=8, omega0=6.0):
        super().__init__()
        self.hid_dim = hid_dim
        self.temporature = temporature
        self.K_len = K_len
        self.num_scales = num_scales
        self.omega0 = omega0

        # Φ_mask: map concatenated real/imag wavelet coefficients to per-scale logits.
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
        """
        Discrete complex Morlet wavelet:
        Gaussian envelope (temporal localization) × complex oscillation (frequency/scale selection).
        """
        if T <= 0:
            raise ValueError(f"Time length T must be positive, got: {T}")
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
        """
        Precompute FFT-domain wavelet filters for efficient circular convolution.
        """
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
        """
        DCWT mapping from node trajectories to a multi-scale complex coefficient domain.
        """
        N, T, d = signal.shape
        device = signal.device

        self._precompute_wavelets(T, device)

        signal_fft = torch.fft.fft(signal, n=T, dim=1, norm='ortho')  # [N, T, d]

        wavelets_fft = self._cached_wavelets['fft']  # [num_scales, T]

        signal_fft_expanded = signal_fft.unsqueeze(1)  # [N, 1, T, d]
        wavelets_fft_expanded = wavelets_fft.unsqueeze(0).unsqueeze(-1)  # [1, num_scales, T, 1]

        conv_fft = signal_fft_expanded * wavelets_fft_expanded  # [N, num_scales, T, d]
        conv_result = torch.fft.ifft(conv_fft, dim=2, norm='ortho')  # [N, num_scales, T, d]

        # Use a short tail-window mean to align with the latest observation and mitigate boundary effects.
        W = 3 if T >= 3 else T
        cwt_coeffs = conv_result[:, :, T - W:, :].mean(dim=2)  # [N, num_scales, d]

        if self.num_scales != self.K_len:
            cwt_coeffs_real = cwt_coeffs.real.transpose(1, 2)  # [N, d, num_scales]
            cwt_coeffs_imag = cwt_coeffs.imag.transpose(1, 2)  # [N, d, num_scales]

            cwt_coeffs_real = F.interpolate(
                cwt_coeffs_real,
                size=self.K_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # [N, K_len, d]

            cwt_coeffs_imag = F.interpolate(
                cwt_coeffs_imag,
                size=self.K_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # [N, K_len, d]

            cwt_coeffs = torch.complex(cwt_coeffs_real, cwt_coeffs_imag)

        return cwt_coeffs

    def discrete_cwt_full(self, signal):
        N, T, d = signal.shape  # [N,T,d]
        device = signal.device
        self._precompute_wavelets(T, device)  # wavelets_fft: [num_scales,T]
        signal_fft = torch.fft.fft(signal, n=T, dim=1, norm='ortho')  # [N,T,d]
        wavelets_fft = self._cached_wavelets['fft']  # [num_scales,T]
        signal_fft_expanded = signal_fft.unsqueeze(1)  # [N,1,T,d]
        wavelets_fft_expanded = wavelets_fft.unsqueeze(0).unsqueeze(-1)  # [1,num_scales,T,1]
        conv_fft = signal_fft_expanded * wavelets_fft_expanded  # [N,num_scales,T,d]
        conv_result = torch.fft.ifft(conv_fft, dim=2, norm='ortho')  # [N,num_scales,T,d]
        W = 3 if T >= 3 else T
        cwt_coeffs = conv_result[:, :, T - W:, :].mean(dim=2)  # [N,num_scales,d]
        if self.num_scales != self.K_len:
            cwt_coeffs_real = cwt_coeffs.real.transpose(1, 2)  # [N,d,num_scales]
            cwt_coeffs_imag = cwt_coeffs.imag.transpose(1, 2)  # [N,d,num_scales]
            cwt_coeffs_real = F.interpolate(cwt_coeffs_real, size=self.K_len, mode='linear', align_corners=False).transpose(1, 2)  # [N,K_len,d]
            cwt_coeffs_imag = F.interpolate(cwt_coeffs_imag, size=self.K_len, mode='linear', align_corners=False).transpose(1, 2)  # [N,K_len,d]
            cwt_coeffs = torch.complex(cwt_coeffs_real, cwt_coeffs_imag)
        return cwt_coeffs, conv_fft  # specs: [N,K_len,d], conv_fft: [N,num_scales,T,d]

    def forward(self, z):
        """
        Split wavelet coefficients into causal and spurious parts via complementary dual masks.
        """
        if len(z.shape) != 3:
            raise ValueError(f"Expected z as [N, T, d], got: {z.shape}")

        N, T, d = z.shape
        if T <= 0 or N <= 0 or d <= 0:
            raise ValueError(f"All input dimensions must be positive, got: [N={N}, T={T}, d={d}]")

        specs = self.discrete_cwt(z)  # [N, K_len, d]

        spec_real, spec_imag = specs.real, specs.imag  # [N, K, d]
        spec_real_imag = torch.stack([spec_real, spec_imag], dim=-1)  # [N, K, d, 2]
        node_choice = self.node_spec_map(spec_real_imag.view(-1, self.K_len * self.hid_dim * 2))  # [N, K]

        # Dual masks: M_c = sigmoid(L / tau), M_s = sigmoid(-L / tau) = 1 - M_c.
        cmask_ = torch.sigmoid(node_choice / self.temporature)
        smask_ = torch.sigmoid(-node_choice / self.temporature)  # [N, K]

        if len(COLLECTOR.cache.get('cmask0', [])) == len(COLLECTOR.cache.get('loss', [])):
            COLLECTOR.add('cmask0', seq2str(cmask_[0].detach().cpu().numpy()))
            COLLECTOR.add('smask0', seq2str(smask_[0].detach().cpu().numpy()))

        cmask = cmask_.unsqueeze(-1).expand_as(spec_imag)
        smask = smask_.unsqueeze(-1).expand_as(spec_imag)

        # Apply masks in the wavelet coefficient domain.
        c_spec_real = spec_real * cmask  # [N, K, d] * [N, K, d]
        c_spec_imag = spec_imag * cmask  # [N, K, d] * [N, K, d]

        s_spec_real = spec_real * smask  # [N, K, d] * [N, K, d]
        s_spec_imag = spec_imag * smask  # [N, K, d] * [N, K, d]

        c_spec = torch.cat([c_spec_real, c_spec_imag], dim=-1).flatten(-2, -1)
        s_spec = torch.cat([s_spec_real, s_spec_imag], dim=-1).flatten(-2, -1)

        c_spec = self.spec_lin(c_spec)  # [N, d]
        s_spec = self.spec_lin(s_spec)  # [N, d]

        return c_spec, s_spec

    def forward_time_reconstruct(self, z):
        N, T, d = z.shape
        spec_len_eff = max(1, 2 * (self.K_len - 1))
        T_eff = min(T, spec_len_eff)
        z = z[:, T - T_eff:, :]
        _, conv_fft = self.discrete_cwt_full(z)  #[N,K_len,d],[N,num_scales,T_eff,d]
        conv_result = torch.fft.ifft(conv_fft, dim=2, norm='ortho')  # [N,num_scales,T_eff,d]
        W = 3 if T_eff >= 3 else T_eff
        scale_coeffs = conv_result[:, :, T_eff - W:, :].mean(dim=2)  # [N,num_scales,d]
        sc_real, sc_imag = scale_coeffs.real, scale_coeffs.imag  # [N,num_scales,d]
        sc_real_imag = torch.stack([sc_real, sc_imag], dim=-1)  # [N,num_scales,d,2]
        node_choice_scale = self.scale_node_spec_map(sc_real_imag.view(-1, self.num_scales * self.hid_dim * 2))  # [N,num_scales]
        cmask_ = torch.sigmoid(node_choice_scale / self.temporature)
        smask_ = torch.sigmoid(-node_choice_scale / self.temporature)
        cmask = cmask_.unsqueeze(-1).expand_as(sc_real)  # [N,num_scales,d]
        smask = smask_.unsqueeze(-1).expand_as(sc_real)  # [N,num_scales,d]
        cmask_exp = cmask.unsqueeze(2)  # [N,num_scales,1,d]
        smask_exp = smask.unsqueeze(2)  # [N,num_scales,1,d]
        c_conv_fft = conv_fft * cmask_exp  #[N,num_scales,T_eff,d]
        s_conv_fft = conv_fft * smask_exp
        c_out_time = torch.fft.ifft(c_conv_fft, dim=2, norm='ortho').real.mean(dim=1)  #[N,T_eff,d]
        s_out_time = torch.fft.ifft(s_conv_fft, dim=2, norm='ortho').real.mean(dim=1)
        return c_out_time, s_out_time

    def forward_scale_compress(self, z):
        N, T, d = z.shape
        spec_len_eff = max(1, 2 * (self.K_len - 1))
        T_eff = min(T, spec_len_eff)
        z = z[:, T - T_eff:, :]
        _, conv_fft = self.discrete_cwt_full(z)
        conv_result = torch.fft.ifft(conv_fft, dim=2, norm='ortho')  # [N,num_scales,T_eff,d]
        W = 3 if T_eff >= 3 else T_eff
        scale_coeffs = conv_result[:, :, T_eff - W:, :].mean(dim=2)  # [N,num_scales,d]
        sc_real, sc_imag = scale_coeffs.real, scale_coeffs.imag  # [N,num_scales,d]
        sc_real_imag = torch.stack([sc_real, sc_imag], dim=-1)  # [N,num_scales,d,2]
        node_choice_scale = self.scale_node_spec_map(sc_real_imag.view(-1, self.num_scales * self.hid_dim * 2))  # [N,num_scales]
        cmask_ = torch.sigmoid(node_choice_scale / self.temporature)
        smask_ = torch.sigmoid(-node_choice_scale / self.temporature)
        cmask = cmask_.unsqueeze(-1).expand_as(sc_real)  # [N,num_scales,d]
        smask = smask_.unsqueeze(-1).expand_as(sc_real)  # [N,num_scales,d]
        c_real = sc_real * cmask
        c_imag = sc_imag * cmask
        s_real = sc_real * smask
        s_imag = sc_imag * smask
        c_feat = torch.cat([c_real, c_imag], dim=-1).flatten(-2, -1)  # [N, num_scales * 2 * d]
        s_feat = torch.cat([s_real, s_imag], dim=-1).flatten(-2, -1)
        c_out = self.scale_spec_lin(c_feat)  # [N,d]
        s_out = self.scale_spec_lin(s_feat)  # [N,d]
        return c_out, s_out


class SpecMask(nn.Module):
    def __init__(self, hid_dim, temporature, K_len) -> None:
        super().__init__()
        self.K_len = K_len
        self.node_spec_map = nn.Sequential(nn.Linear(self.K_len * hid_dim * 2, hid_dim * 2), nn.ReLU(),
                                           nn.Linear(hid_dim * 2, self.K_len))  # N, K x d x 2 - > N x f
        self.temporature = temporature
        self.K_len = K_len
        self.hid_dim = hid_dim

    def forward(self, specs):
        # specs [N, T, d]
        # learn causal and spurious masks
        spec_real, spec_imag = specs.real, specs.imag  # [N, K, d]
        spec_real_imag = torch.stack([spec_real, spec_imag], dim=-1)  # [N, K, d, 2]
        node_choice = self.node_spec_map(
            spec_real_imag.view(-1, self.K_len * self.hid_dim * 2))  # [N, K * d * 2] -> [N, K]

        cmask_ = torch.sigmoid(node_choice / self.temporature)
        smask_ = torch.sigmoid(- node_choice / self.temporature)  # [N, K]
        if len(COLLECTOR.cache.get('cmask0', [])) == len(COLLECTOR.cache.get('loss', [])):
            COLLECTOR.add('cmask0', seq2str(cmask_[0].detach().cpu().numpy()))
            COLLECTOR.add('smask0', seq2str(smask_[0].detach().cpu().numpy()))

        cmask = cmask_.unsqueeze(-1).expand_as(spec_imag)
        smask = smask_.unsqueeze(-1).expand_as(spec_imag)

        # filter in the spectral domain
        c_spec_real = spec_real * cmask  # [N, K, d] * [N, K, d]
        c_spec_imag = spec_imag * cmask  # [N, K, d] * [N, K, d]
        c_spec = torch.complex(c_spec_real, c_spec_imag)  # [N, K, d]

        s_spec_real = spec_real * smask  # [N, K, d] * [N, K, d]
        s_spec_imag = spec_imag * smask  # [N, K, d] * [N, K, d]
        s_spec = torch.complex(s_spec_real, s_spec_imag)  # [N, K, d]

        return c_spec, s_spec


class SpecAttn(nn.Module):
    def __init__(self, hid_dim, args) -> None:
        super().__init__()
        self.spec_attn = SpecAttentionLayer(hid_dim, 1, args.attn_drop, args.temp_attn_res, use_RTE=args.spec_RTE,
                                            temporature=args.temporature, norm=True)
        self.hid_dim = hid_dim

    def forward(self, specs):
        # specs [N, T, d]
        c_spec, s_spec = self.spec_attn(specs)  # [N, T, d]
        return c_spec, s_spec


class SGCNNetIter(torch.nn.Module):
    def __init__(self, args, data):
        super(SGCNNetIter, self).__init__()
        self.args = args

        # hyperparams
        in_dim = args.nfeat
        hid_dim = 2 * args.nhid
        out_dim = 2 * args.nhid
        num_layers = args.n_layers
        time_length = args.length
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.window_size = args.window_size
        self.temporature = args.temporature
        self.norm = args.norm

        if args.spec_len == -1:
            self.spec_len = args.length
        elif args.spec_len == -2:
            self.spec_len = args.window_size
        else:
            self.spec_len = args.spec_len
        self.K_len = 1 + self.spec_len // 2

        # dataset
        self.time_length = time_length
        self.len = args.length
        self.len_train = self.len - args.testlength - args.vallength
        self.len_val = args.vallength
        self.len_test = args.testlength
        x = data['x']
        self.x = [x for _ in range(self.len)] if len(x.shape) <= 2 else x
        print('total length: {}, test length: {}, window_size : {}'.format(self.len, args.testlength, self.window_size))

        # spatio model
        if args.backbone in ("wavcil", "dida"):
            self.backbone = DGNN(args)
        elif args.backbone in 'GCN GIN GAT'.split():
            args.static_conv = args.backbone
            self.backbone = SpaSeqNet(args)
        else:
            raise NotImplementedError()

        # spectral model
        if args.spec_filt == "mask":
            self.spec_filt = SpecMask(hid_dim, args.temporature, self.K_len)
        elif args.spec_filt == 'attn':
            self.spec_filt = SpecAttn(hid_dim, args)
        elif args.spec_filt == 'dcwt':
            # DCWT: map node trajectories to a complex wavelet coefficient domain with time-frequency localization.
            num_scales = getattr(args, 'dcwt_scales', 8)
            omega0 = getattr(args, 'dcwt_omega0', 6.0)
            self.spec_filt = DiscreteComplexWaveletMask(hid_dim, args.temporature, self.K_len, num_scales, omega0)
        else:
            raise NotImplementedError()

        # post temporal
        if args.post_temporal:
            self.temporal_attn = TemporalAttentionLayer(hid_dim, args.heads, args.attn_drop, args.temp_attn_res,
                                                        use_RTE=args.use_RTE, only_last=True)

        # clf
        self.ctype = args.ctype
        self.ltype = args.ltype

        # measure and link prediction
        self.cs_decoder = self.backbone.cs_decoder
        self.ss_decoder = self.backbone.ss_decoder
        self.measure = EnvLoss(args)

        # optim
        self.optim = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.criterion = torch.nn.BCELoss()

    def cal_loss(self, zs, edge_list, epoch):
        cs, ss = zs  # [T, N, F], [T, N, F] z starts from 0 to T
        args = self.args
        device = args.device
        intervene_times, la = args.intervene_times, args.intervene_lambda
        len_train = self.len_train

        edge_index, edge_label = get_edges_all(edge_list[1:],
                                               self.args)  # [T-1, E]edges start from 1 to T for prediction

        def cal_y(embeddings, decoder):
            preds = torch.tensor([]).to(device)
            for t in range(len_train - 1):  # [T-1]
                z = embeddings[t]
                pred = decoder(z, edge_index[t])
                preds = torch.cat([preds, pred])
            return preds

        criterion = torch.nn.BCELoss()

        def cal_loss(y, label):
            return criterion(y, label)

        cy = cal_y(cs, self.cs_decoder)
        sy = cal_y(ss, self.ss_decoder)
        causal_loss = cal_loss(cy, edge_label)

        if self.args.intervene_times <= 0 or self.args.intervene_lambda <= 0:
            loss = causal_loss
        else:
            env_loss = torch.tensor([]).to(device)
            for i in range(intervene_times):
                s1 = np.random.randint(len(sy))
                s = torch.sigmoid(sy[s1]).detach()
                conf = s * cy
                env_loss = torch.cat([env_loss, cal_loss(conf, edge_label).unsqueeze(0)])
            # Intervention-based invariance penalty: mean + variance of losses across random interventions.
            env_mean = env_loss.mean()
            env_var = torch.var(env_loss * intervene_times)
            penalty = env_mean + env_var
            loss = causal_loss + la * penalty

            COLLECTOR.add('env_mean', env_mean.item())
            COLLECTOR.add('env_var', env_var.item())

        COLLECTOR.add('causal_loss', causal_loss.item())
        return loss

    def spectral_filter(self, z):
        # z [T, N, d]
        if not self.args.use_filt:
            return z[-1], z[-1]
        ctype = self.ctype
        time_len = z.shape[0]

        # Branch between DCWT and FFT-based spectral filtering.
        if self.args.spec_filt == 'dcwt':
            z = torch.permute(z, (1, 0, 2))
            if getattr(self.args, 'dcwt_reconstruct_time', 0):
                c_out, s_out = self.spec_filt.forward_time_reconstruct(z)
                if self.args.post_temporal:
                    c_out = self.temporal_attn(c_out)
                    s_out = self.temporal_attn(s_out)
                c_out = torch.permute(c_out, (1, 0, 2))
                s_out = torch.permute(s_out, (1, 0, 2))
                c_out = c_out[-1]
                s_out = s_out[-1]
                return c_out, s_out
            else:
                c_out, s_out = self.spec_filt.forward_scale_compress(z)
                return c_out, s_out
        else:
            # transform into spectral domain
            z = torch.permute(z, (1, 0, 2))  # [N, T, d]
            specs = rfft(z, n=self.spec_len, norm="ortho", dim=1)  # [N, K, d]

            # learn causal and spurious masks
            c_spec, s_spec = self.spec_filt(specs)  # [N, T, d]

            # transform back to time domain
            cout = irfft(c_spec, n=self.spec_len, norm="ortho", dim=1)[:, :time_len, :]  # [N, T, d]
            if self.args.post_temporal:
                cout = self.temporal_attn(cout)  # [N, T, d]
            cout = torch.permute(cout, (1, 0, 2))  # [T, N, d]

            sout = irfft(s_spec, n=self.spec_len, norm="ortho", dim=1)[:, :time_len, :]  # [N, T, d]
            if self.args.post_temporal:
                sout = self.temporal_attn(sout)  # [N, T, d]
            sout = torch.permute(sout, (1, 0, 2))  # [T, N, d]

            cout = cout[-1]
            sout = sout[-1]
            out = [cout, sout]
            return out

    def get_final_emb(self, edge_list):
        x_list, cs, ss = self.backbone(edge_list, self.x[:len(edge_list)])
        x_list = torch.stack(x_list, dim=0)
        time_len = len(edge_list)
        czs, szs = [], []
        for end in range(1, time_len):
            cz, sz = self.spectral_filter(x_list[:end])  # [T,N,F]
            czs.append(cz)
            szs.append(sz)
        czs = torch.stack(czs, dim=0)  # [T,N,F]
        szs = torch.stack(szs, dim=0)  # [T,N,F]
        return czs, szs

    def train_epoch(self, data, epoch):
        args, optimizer, model, window_size = self.args, self.optim, self, self.window_size

        model.train()
        edge_list = [data['edge_index_list'][ix] for ix in range(self.len_train)]
        cs, ss = self.get_final_emb(edge_list)

        loss = self.cal_loss([cs, ss], edge_list, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def test_epoch(self, data, type='train'):
        args, model, window_size = self.args, self, self.window_size
        train_auc_list, val_auc_list, test_auc_list = [], [], []

        model.eval()
        edge_list = [data['edge_index_list'][ix] for ix in range(self.len)]
        cs, ss = self.get_final_emb(edge_list)

        for t in range(self.len - 1):
            z = cs[t]
            edge_index, pos_edge, neg_edge = prepare(data, t + 1)[:3]
            if is_empty_edges(neg_edge):
                continue
            auc, ap = self.measure.predict(z, pos_edge, neg_edge, self.cs_decoder)
            if t < self.len_train - 1:
                train_auc_list.append(auc)
            elif t < self.len_train + self.len_val - 1:
                val_auc_list.append(auc)
            else:
                test_auc_list.append(auc)

        train, val, test = np.mean(train_auc_list), np.mean(val_auc_list), np.mean(test_auc_list)
        COLLECTOR.add(key=f'{type}_train_auc_list', value=train_auc_list)
        COLLECTOR.add(key=f'{type}_val_auc_list', value=val_auc_list)
        COLLECTOR.add(key=f'{type}_test_auc_list', value=test_auc_list)

        return train, val, test
