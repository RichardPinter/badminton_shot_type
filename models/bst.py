# Writen by Jing-Yuan Chang

import torch
from torch import nn, Tensor
from positional_encodings.torch_encodings import PositionalEncoding1D
from torchinfo import summary

from models.tempose import MLP_Head, TransformerEncoder, TCN, MLP
from models.shuttlepose import CrossTransformerLayer


class BST(nn.Module):
    '''BadmintonStrokeTransformer
    - ShuttlePose_3_3 + cross attention on facing(player positions)
    '''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=1,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
    ):
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)
        tcn_channels = [d_model // 2, d_model]
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_positions = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)
        
        # MLP Head
        self.mlp_head = MLP_Head(d_model * 4, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem)
        self.embedding_tem.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_cross)
        self.embedding_cross.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        # Same as TemPose
        if isinstance(m, nn.Linear):
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,      # JnB: (b, t, n, input_dim)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        pos: Tensor,      # pos: (b, t, n, 2)
        video_len: Tensor  # video_len: (b)
    ):
        b, t, n, in_dim = JnB.shape
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)
        JnB = self.tcn_pose(JnB)
        JnB = JnB.view(b, n, -1, t).transpose(-2, -1)

        shuttle = shuttle.transpose(1, 2).contiguous()
        shuttle = self.tcn_shuttle(shuttle)
        shuttle = shuttle.unsqueeze(1).transpose(-2, -1)
        
        pos = pos.permute(2, 0, 1, 3).contiguous()
        facing: Tensor = pos[0] - pos[1]
        # facing: (b, t, 2)
        facing = facing.transpose(1, 2).contiguous()
        facing = self.tcn_positions(facing)
        facing = facing.unsqueeze(1).transpose(-2, -1)

        x = torch.cat((JnB, shuttle, facing), dim=1)
        _, n, _, d = x.shape

        class_token_tem = self.learned_token_tem.view(1, 1, -1).expand(b*n, -1, -1)
        x = x.view(b*n, t, d)
        x = torch.cat((class_token_tem, x), dim=1) + self.embedding_tem

        range_t = torch.arange(0, 1+t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, 1+t)
        mask_n = mask.repeat_interleave(n, dim=0)
        # mask_n: (b*n, 1+t)

        x: Tensor = self.pre_dropout(x)
        x = self.encoder_tem(x, mask_n)
        x = x.view(b, n, 1+t, d)

        p1_cls = x[:, 0, 0, :].contiguous()
        p2_cls = x[:, 1, 0, :].contiguous()
        shuttle_cls = x[:, 2, 0, :].contiguous()
        facing_cls = x[:, 3, 0, :].contiguous()

        p1 = x[:, 0, 1:, :].contiguous() + self.embedding_cross
        p2 = x[:, 1, 1:, :].contiguous() + self.embedding_cross
        shuttle = x[:, 2, 1:, :].contiguous() + self.embedding_cross
        facing = x[:, 3, 1:, :].contiguous() + self.embedding_cross

        cross_mask = mask[:, 1:].contiguous()
        p1_shuttle = self.cross_trans(p1, shuttle, cross_mask)
        p2_shuttle = self.cross_trans(p2, shuttle, cross_mask)

        p1_facing = self.cross_trans(p1, facing, cross_mask)
        p2_facing = self.cross_trans(p2, facing, cross_mask)

        p1_situation = p1_shuttle + p1_facing
        p2_situation = p2_shuttle + p2_facing

        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        p1_situation = torch.cat((class_token_inter, p1_situation), dim=1) + self.embedding_inter
        p2_situation = torch.cat((class_token_inter, p2_situation), dim=1) + self.embedding_inter

        p1_situation: Tensor = self.encoder_inter(p1_situation, mask)
        p2_situation: Tensor = self.encoder_inter(p2_situation, mask)

        p1_situation_cls = p1_situation[:, 0, :].contiguous()
        p2_situation_cls = p2_situation[:, 0, :].contiguous()

        p1_conclusion = p1_cls + p1_situation_cls
        p2_conclusion = p2_cls + p2_situation_cls

        x = torch.cat((p1_conclusion, p2_conclusion, shuttle_cls, facing_cls), dim=1)
        x = self.mlp_head(x)
        return x


class BST_2(nn.Module):
    '''BadmintonStrokeTransformer
    - ShuttlePose_3_3 + cross attention on facing(players positions)
    - No residual connection on facing
    '''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=1,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
    ):
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)
        tcn_channels = [d_model // 2, d_model]
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_positions = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)
        
        # MLP Head
        self.mlp_head = MLP_Head(d_model * 3, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem)
        self.embedding_tem.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_cross)
        self.embedding_cross.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        # Same as TemPose
        if isinstance(m, nn.Linear):
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,      # JnB: (b, t, n, input_dim)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        pos: Tensor,      # pos: (b, t, n, 2)
        video_len: Tensor  # video_len: (b)
    ):
        b, t, n, in_dim = JnB.shape
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)
        JnB = self.tcn_pose(JnB)
        JnB = JnB.view(b, n, -1, t).transpose(-2, -1)

        shuttle = shuttle.transpose(1, 2).contiguous()
        shuttle = self.tcn_shuttle(shuttle)
        shuttle = shuttle.unsqueeze(1).transpose(-2, -1)
        
        pos = pos.permute(2, 0, 1, 3).contiguous()
        facing: Tensor = pos[0] - pos[1]
        # facing: (b, t, 2)
        facing = facing.transpose(1, 2).contiguous()
        facing = self.tcn_positions(facing)
        facing = facing.unsqueeze(1).transpose(-2, -1)

        x = torch.cat((JnB, shuttle, facing), dim=1)
        _, n, _, d = x.shape

        class_token_tem = self.learned_token_tem.view(1, 1, -1).expand(b*n, -1, -1)
        x = x.view(b*n, t, d)
        x = torch.cat((class_token_tem, x), dim=1) + self.embedding_tem

        range_t = torch.arange(0, 1+t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, 1+t)
        mask_n = mask.repeat_interleave(n, dim=0)
        # mask_n: (b*n, 1+t)

        x: Tensor = self.pre_dropout(x)
        x = self.encoder_tem(x, mask_n)
        x = x.view(b, n, 1+t, d)

        p1_cls = x[:, 0, 0, :].contiguous()
        p2_cls = x[:, 1, 0, :].contiguous()
        shuttle_cls = x[:, 2, 0, :].contiguous()

        p1 = x[:, 0, 1:, :].contiguous() + self.embedding_cross
        p2 = x[:, 1, 1:, :].contiguous() + self.embedding_cross
        shuttle = x[:, 2, 1:, :].contiguous() + self.embedding_cross
        facing = x[:, 3, 1:, :].contiguous() + self.embedding_cross

        cross_mask = mask[:, 1:].contiguous()
        p1_shuttle = self.cross_trans(p1, shuttle, cross_mask)
        p2_shuttle = self.cross_trans(p2, shuttle, cross_mask)

        p1_facing = self.cross_trans(p1, facing, cross_mask)
        p2_facing = self.cross_trans(p2, facing, cross_mask)

        p1_situation = p1_shuttle + p1_facing
        p2_situation = p2_shuttle + p2_facing

        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        p1_situation = torch.cat((class_token_inter, p1_situation), dim=1) + self.embedding_inter
        p2_situation = torch.cat((class_token_inter, p2_situation), dim=1) + self.embedding_inter

        p1_situation: Tensor = self.encoder_inter(p1_situation, mask)
        p2_situation: Tensor = self.encoder_inter(p2_situation, mask)

        p1_situation_cls = p1_situation[:, 0, :].contiguous()
        p2_situation_cls = p2_situation[:, 0, :].contiguous()

        p1_conclusion = p1_cls + p1_situation_cls
        p2_conclusion = p2_cls + p2_situation_cls

        x = torch.cat((p1_conclusion, p2_conclusion, shuttle_cls), dim=1)
        x = self.mlp_head(x)
        return x


class BST_3(nn.Module):
    '''BadmintonStrokeTransformer
    - ShuttlePose_3_3
    - Players positions embedding (from TCN) before CrossTransformerLayer
    '''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=1,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,
        position_factor=0.1
    ):
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)
        tcn_channels = [d_model // 2, d_model]
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_positions = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)
        
        # MLP Head
        self.mlp_head = MLP_Head(d_model * 3, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model
        self.alpha = position_factor

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem)
        self.embedding_tem.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_cross)
        self.embedding_cross.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        # Same as TemPose
        if isinstance(m, nn.Linear):
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,      # JnB: (b, t, n, input_dim)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        pos: Tensor,      # pos: (b, t, n, 2)
        video_len: Tensor  # video_len: (b)
    ):
        b, t, n, in_dim = JnB.shape
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)
        JnB = self.tcn_pose(JnB)
        JnB = JnB.view(b, n, -1, t).transpose(-2, -1)

        shuttle = shuttle.transpose(1, 2).contiguous()
        shuttle = self.tcn_shuttle(shuttle)
        shuttle = shuttle.unsqueeze(1).transpose(-2, -1)
        
        pos = pos.permute(2, 0, 1, 3).contiguous()
        pos = pos.view(n*b, t, 2)
        pos = pos.transpose(1, 2).contiguous()
        pos = self.tcn_positions(pos)
        pos = pos.transpose(1, 2).reshape(n, b, t, -1)

        x = torch.cat((JnB, shuttle), dim=1)
        _, n, _, d = x.shape

        class_token_tem = self.learned_token_tem.view(1, 1, -1).expand(b*n, -1, -1)
        x = x.view(b*n, t, d)
        x = torch.cat((class_token_tem, x), dim=1) + self.embedding_tem

        range_t = torch.arange(0, 1+t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, 1+t)
        mask_n = mask.repeat_interleave(n, dim=0)
        # mask_n: (b*n, 1+t)

        x: Tensor = self.pre_dropout(x)
        x = self.encoder_tem(x, mask_n)
        x = x.view(b, n, 1+t, d)

        p1_cls = x[:, 0, 0, :].contiguous()
        p2_cls = x[:, 1, 0, :].contiguous()
        shuttle_cls = x[:, 2, 0, :].contiguous()

        p1_embedding = self.alpha * pos[0] + self.embedding_cross
        p2_embedding = self.alpha * pos[1] + self.embedding_cross

        p1 = x[:, 0, 1:, :].contiguous() + p1_embedding
        p2 = x[:, 1, 1:, :].contiguous() + p2_embedding
        shuttle = x[:, 2, 1:, :].contiguous() + self.embedding_cross

        cross_mask = mask[:, 1:].contiguous()
        p1_shuttle = self.cross_trans(p1, shuttle, cross_mask)
        p2_shuttle = self.cross_trans(p2, shuttle, cross_mask)

        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        p1_shuttle = torch.cat((class_token_inter, p1_shuttle), dim=1) + self.embedding_inter
        p2_shuttle = torch.cat((class_token_inter, p2_shuttle), dim=1) + self.embedding_inter

        p1_shuttle: Tensor = self.encoder_inter(p1_shuttle, mask)
        p2_shuttle: Tensor = self.encoder_inter(p2_shuttle, mask)

        p1_shuttle_cls = p1_shuttle[:, 0, :].contiguous()
        p2_shuttle_cls = p2_shuttle[:, 0, :].contiguous()

        p1_conclusion = p1_cls + p1_shuttle_cls
        p2_conclusion = p2_cls + p2_shuttle_cls

        x = torch.cat((p1_conclusion, p2_conclusion, shuttle_cls), dim=1)
        x = self.mlp_head(x)
        return x


class BST_3_2(nn.Module):
    '''BadmintonStrokeTransformer
    - ShuttlePose_4
    - Players positions embedding (from TCN) before CrossTransformerLayer
    '''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=1,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,
        position_factor=0.1
    ):
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)
        tcn_channels = [d_model // 2, d_model]
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_positions = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)
        
        # MLP Head
        self.mlp_head = MLP_Head(d_model * 3, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model
        self.alpha = position_factor

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem)
        self.embedding_tem.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_cross)
        self.embedding_cross.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        # Same as TemPose
        if isinstance(m, nn.Linear):
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,      # JnB: (b, t, n, input_dim)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        pos: Tensor,      # pos: (b, t, n, 2)
        video_len: Tensor  # video_len: (b)
    ):
        b, t, n, in_dim = JnB.shape
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)
        JnB = self.tcn_pose(JnB)
        JnB = JnB.view(b, n, -1, t).transpose(-2, -1)

        shuttle = shuttle.transpose(1, 2).contiguous()
        shuttle = self.tcn_shuttle(shuttle)
        shuttle = shuttle.unsqueeze(1).transpose(-2, -1)
        
        pos = pos.permute(2, 0, 1, 3).contiguous()
        pos = pos.view(n*b, t, 2)
        pos = pos.transpose(1, 2).contiguous()
        pos = self.tcn_positions(pos)
        pos = pos.transpose(1, 2).reshape(n, b, t, -1)

        x = torch.cat((JnB, shuttle), dim=1)
        _, n, _, d = x.shape

        class_token_tem = self.learned_token_tem.view(1, 1, -1).expand(b*n, -1, -1)
        x = x.view(b*n, t, d)
        x = torch.cat((class_token_tem, x), dim=1) + self.embedding_tem

        range_t = torch.arange(0, 1+t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, 1+t)
        mask_n = mask.repeat_interleave(n, dim=0)
        # mask_n: (b*n, 1+t)

        x: Tensor = self.pre_dropout(x)
        x = self.encoder_tem(x, mask_n)
        x = x.view(b, n, 1+t, d)

        p1_cls = x[:, 0, 0, :].contiguous()
        p2_cls = x[:, 1, 0, :].contiguous()
        shuttle_cls = x[:, 2, 0, :].contiguous()

        p1_embedding = self.alpha * pos[0] + self.embedding_cross
        p2_embedding = self.alpha * pos[1] + self.embedding_cross

        p1 = x[:, 0, 1:, :].contiguous() + p1_embedding
        p2 = x[:, 1, 1:, :].contiguous() + p2_embedding
        shuttle = x[:, 2, 1:, :].contiguous() + self.embedding_cross

        cross_mask = mask[:, 1:].contiguous()
        p1_shuttle = self.cross_trans(p1, shuttle, cross_mask)
        p2_shuttle = self.cross_trans(p2, shuttle, cross_mask)
        shuttle_p1 = self.cross_trans(shuttle, p1, cross_mask)
        shuttle_p2 = self.cross_trans(shuttle, p2, cross_mask)

        p1_behavior, _ = torch.stack([p1_shuttle, shuttle_p1]).max(0)
        p2_behavior, _ = torch.stack([p2_shuttle, shuttle_p2]).max(0)

        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        p1_behavior = torch.cat((class_token_inter, p1_behavior), dim=1) + self.embedding_inter
        p2_behavior = torch.cat((class_token_inter, p2_behavior), dim=1) + self.embedding_inter

        p1_behavior: Tensor = self.encoder_inter(p1_behavior, mask)
        p2_behavior: Tensor = self.encoder_inter(p2_behavior, mask)

        p1_behavior_cls = p1_behavior[:, 0, :].contiguous()
        p2_behavior_cls = p2_behavior[:, 0, :].contiguous()

        p1_conclusion = p1_cls + p1_behavior_cls
        p2_conclusion = p2_cls + p2_behavior_cls

        x = torch.cat((p1_conclusion, p2_conclusion, shuttle_cls), dim=1)
        x = self.mlp_head(x)
        return x


class BST_4(nn.Module):
    '''BadmintonStrokeTransformer
    - ShuttlePose_3_3
    - Players positions embedding (from projection) before CrossTransformerLayer
    '''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=1,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,
        position_factor=0.1
    ):
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, [d_model // 2, d_model], tcn_kernel_size, drop_p)
        self.project_positions = nn.Linear(2, d_model)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)
        
        # MLP Head
        self.mlp_head = MLP_Head(d_model * 3, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model
        self.alpha = position_factor

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem)
        self.embedding_tem.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_cross)
        self.embedding_cross.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        # Same as TemPose
        if isinstance(m, nn.Linear):
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,      # JnB: (b, t, n, input_dim)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        pos: Tensor,      # pos: (b, t, n, 2)
        video_len: Tensor  # video_len: (b)
    ):
        b, t, n, in_dim = JnB.shape
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)
        JnB = self.tcn_pose(JnB)
        JnB = JnB.view(b, n, -1, t).transpose(-2, -1)

        shuttle = shuttle.transpose(1, 2).contiguous()
        shuttle = self.tcn_shuttle(shuttle)
        shuttle = shuttle.unsqueeze(1).transpose(-2, -1)
        
        pos = pos.permute(2, 0, 1, 3).contiguous()
        pos = pos.view(n*b, t, 2)
        pos = self.project_positions(pos)
        pos = pos.view(n, b, t, -1)

        x = torch.cat((JnB, shuttle), dim=1)
        _, n, _, d = x.shape

        class_token_tem = self.learned_token_tem.view(1, 1, -1).expand(b*n, -1, -1)
        x = x.view(b*n, t, d)
        x = torch.cat((class_token_tem, x), dim=1) + self.embedding_tem

        range_t = torch.arange(0, 1+t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, 1+t)
        mask_n = mask.repeat_interleave(n, dim=0)
        # mask_n: (b*n, 1+t)

        x: Tensor = self.pre_dropout(x)
        x = self.encoder_tem(x, mask_n)
        x = x.view(b, n, 1+t, d)

        p1_cls = x[:, 0, 0, :].contiguous()
        p2_cls = x[:, 1, 0, :].contiguous()
        shuttle_cls = x[:, 2, 0, :].contiguous()

        p1_embedding = self.alpha * pos[0] + self.embedding_cross
        p2_embedding = self.alpha * pos[1] + self.embedding_cross

        p1 = x[:, 0, 1:, :].contiguous() + p1_embedding
        p2 = x[:, 1, 1:, :].contiguous() + p2_embedding
        shuttle = x[:, 2, 1:, :].contiguous() + self.embedding_cross

        cross_mask = mask[:, 1:].contiguous()
        p1_shuttle = self.cross_trans(p1, shuttle, cross_mask)
        p2_shuttle = self.cross_trans(p2, shuttle, cross_mask)

        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        p1_shuttle = torch.cat((class_token_inter, p1_shuttle), dim=1) + self.embedding_inter
        p2_shuttle = torch.cat((class_token_inter, p2_shuttle), dim=1) + self.embedding_inter

        p1_shuttle: Tensor = self.encoder_inter(p1_shuttle, mask)
        p2_shuttle: Tensor = self.encoder_inter(p2_shuttle, mask)

        p1_shuttle_cls = p1_shuttle[:, 0, :].contiguous()
        p2_shuttle_cls = p2_shuttle[:, 0, :].contiguous()

        p1_conclusion = p1_cls + p1_shuttle_cls
        p2_conclusion = p2_cls + p2_shuttle_cls

        x = torch.cat((p1_conclusion, p2_conclusion, shuttle_cls), dim=1)
        x = self.mlp_head(x)
        return x


class BST_5(nn.Module):
    '''BadmintonStrokeTransformer
    - ShuttlePose_3_3
    - Players positions added to poses
    '''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=1,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,
        position_factor=0.1
    ):
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.tcn_pose = TCN(in_dim+2, [d_model, d_model], tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, [d_model // 2, d_model], tcn_kernel_size, drop_p)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)
        
        # MLP Head
        self.mlp_head = MLP_Head(d_model * 3, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model
        self.alpha = position_factor

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem)
        self.embedding_tem.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_cross)
        self.embedding_cross.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        # Same as TemPose
        if isinstance(m, nn.Linear):
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,      # JnB: (b, t, n, input_dim)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        pos: Tensor,      # pos: (b, t, n, 2)
        video_len: Tensor  # video_len: (b)
    ):
        b, t, n, in_dim = JnB.shape
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)
        pos = pos.permute(0, 2, 3, 1).reshape(b*n, 2, t)
        JnB = torch.cat((JnB, pos), dim=-2)
        JnB = self.tcn_pose(JnB)
        JnB = JnB.view(b, n, -1, t).transpose(-2, -1)

        shuttle = shuttle.transpose(1, 2).contiguous()
        shuttle = self.tcn_shuttle(shuttle)
        shuttle = shuttle.unsqueeze(1).transpose(-2, -1)
        
        x = torch.cat((JnB, shuttle), dim=1)
        _, n, _, d = x.shape

        class_token_tem = self.learned_token_tem.view(1, 1, -1).expand(b*n, -1, -1)
        x = x.view(b*n, t, d)
        x = torch.cat((class_token_tem, x), dim=1) + self.embedding_tem

        range_t = torch.arange(0, 1+t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, 1+t)
        mask_n = mask.repeat_interleave(n, dim=0)
        # mask_n: (b*n, 1+t)

        x: Tensor = self.pre_dropout(x)
        x = self.encoder_tem(x, mask_n)
        x = x.view(b, n, 1+t, d)

        p1_cls = x[:, 0, 0, :].contiguous()
        p2_cls = x[:, 1, 0, :].contiguous()
        shuttle_cls = x[:, 2, 0, :].contiguous()

        p1 = x[:, 0, 1:, :].contiguous() + self.embedding_cross
        p2 = x[:, 1, 1:, :].contiguous() + self.embedding_cross
        shuttle = x[:, 2, 1:, :].contiguous() + self.embedding_cross

        cross_mask = mask[:, 1:].contiguous()
        p1_shuttle = self.cross_trans(p1, shuttle, cross_mask)
        p2_shuttle = self.cross_trans(p2, shuttle, cross_mask)

        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        p1_shuttle = torch.cat((class_token_inter, p1_shuttle), dim=1) + self.embedding_inter
        p2_shuttle = torch.cat((class_token_inter, p2_shuttle), dim=1) + self.embedding_inter

        p1_shuttle: Tensor = self.encoder_inter(p1_shuttle, mask)
        p2_shuttle: Tensor = self.encoder_inter(p2_shuttle, mask)

        p1_shuttle_cls = p1_shuttle[:, 0, :].contiguous()
        p2_shuttle_cls = p2_shuttle[:, 0, :].contiguous()

        p1_conclusion = p1_cls + p1_shuttle_cls
        p2_conclusion = p2_cls + p2_shuttle_cls

        x = torch.cat((p1_conclusion, p2_conclusion, shuttle_cls), dim=1)
        x = self.mlp_head(x)
        return x


class BST_5_2(nn.Module):
    '''BadmintonStrokeTransformer
    - ShuttlePose_3_3
    - Players positions added to poses after a MLP layer
    '''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=1,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
    ):
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.mlp_positions = MLP(2, out_dim=in_dim, hd_dim=256, drop_p=0.25)

        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, [d_model // 2, d_model], tcn_kernel_size, drop_p)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)
        
        # MLP Head
        self.mlp_head = MLP_Head(d_model * 3, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem)
        self.embedding_tem.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_cross)
        self.embedding_cross.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        # Same as TemPose
        if isinstance(m, nn.Linear):
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,      # JnB: (b, t, n, input_dim)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        pos: Tensor,      # pos: (b, t, n, 2)
        video_len: Tensor  # video_len: (b)
    ):
        b, t, n, in_dim = JnB.shape
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)
        
        pos = self.mlp_positions(pos)
        pos_impact = pos.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)

        JnB = JnB * pos_impact + JnB

        JnB = self.tcn_pose(JnB)
        JnB = JnB.view(b, n, -1, t).transpose(-2, -1)

        shuttle = shuttle.transpose(1, 2).contiguous()
        shuttle = self.tcn_shuttle(shuttle)
        shuttle = shuttle.unsqueeze(1).transpose(-2, -1)
        
        x = torch.cat((JnB, shuttle), dim=1)
        _, n, _, d = x.shape

        class_token_tem = self.learned_token_tem.view(1, 1, -1).expand(b*n, -1, -1)
        x = x.view(b*n, t, d)
        x = torch.cat((class_token_tem, x), dim=1) + self.embedding_tem

        range_t = torch.arange(0, 1+t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, 1+t)
        mask_n = mask.repeat_interleave(n, dim=0)
        # mask_n: (b*n, 1+t)

        x: Tensor = self.pre_dropout(x)
        x = self.encoder_tem(x, mask_n)
        x = x.view(b, n, 1+t, d)

        p1_cls = x[:, 0, 0, :].contiguous()
        p2_cls = x[:, 1, 0, :].contiguous()
        shuttle_cls = x[:, 2, 0, :].contiguous()

        p1 = x[:, 0, 1:, :].contiguous() + self.embedding_cross
        p2 = x[:, 1, 1:, :].contiguous() + self.embedding_cross
        shuttle = x[:, 2, 1:, :].contiguous() + self.embedding_cross

        cross_mask = mask[:, 1:].contiguous()
        p1_shuttle = self.cross_trans(p1, shuttle, cross_mask)
        p2_shuttle = self.cross_trans(p2, shuttle, cross_mask)

        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        p1_shuttle = torch.cat((class_token_inter, p1_shuttle), dim=1) + self.embedding_inter
        p2_shuttle = torch.cat((class_token_inter, p2_shuttle), dim=1) + self.embedding_inter

        p1_shuttle: Tensor = self.encoder_inter(p1_shuttle, mask)
        p2_shuttle: Tensor = self.encoder_inter(p2_shuttle, mask)

        p1_shuttle_cls = p1_shuttle[:, 0, :].contiguous()
        p2_shuttle_cls = p2_shuttle[:, 0, :].contiguous()

        p1_conclusion = p1_cls + p1_shuttle_cls
        p2_conclusion = p2_cls + p2_shuttle_cls

        x = torch.cat((p1_conclusion, p2_conclusion, shuttle_cls), dim=1)
        x = self.mlp_head(x)
        return x


class BST_5_3(nn.Module):
    '''BadmintonStrokeTransformer
    - BST_5_2
    - MLP_positions hd_dim from 256 changed to (in_dim * 2)
    '''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=1,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
    ):
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.mlp_positions = MLP(2, out_dim=in_dim, hd_dim=(in_dim * 2), drop_p=0.25)

        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, [d_model // 2, d_model], tcn_kernel_size, drop_p)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)
        
        # MLP Head
        self.mlp_head = MLP_Head(d_model * 3, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem)
        self.embedding_tem.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_cross)
        self.embedding_cross.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        # Same as TemPose
        if isinstance(m, nn.Linear):
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,      # JnB: (b, t, n, input_dim)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        pos: Tensor,      # pos: (b, t, n, 2)
        video_len: Tensor  # video_len: (b)
    ):
        b, t, n, in_dim = JnB.shape
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)
        
        pos = self.mlp_positions(pos)
        pos_impact = pos.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)

        JnB = JnB * pos_impact + JnB

        JnB = self.tcn_pose(JnB)
        JnB = JnB.view(b, n, -1, t).transpose(-2, -1)

        shuttle = shuttle.transpose(1, 2).contiguous()
        shuttle = self.tcn_shuttle(shuttle)
        shuttle = shuttle.unsqueeze(1).transpose(-2, -1)
        
        x = torch.cat((JnB, shuttle), dim=1)
        _, n, _, d = x.shape

        class_token_tem = self.learned_token_tem.view(1, 1, -1).expand(b*n, -1, -1)
        x = x.view(b*n, t, d)
        x = torch.cat((class_token_tem, x), dim=1) + self.embedding_tem

        range_t = torch.arange(0, 1+t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, 1+t)
        mask_n = mask.repeat_interleave(n, dim=0)
        # mask_n: (b*n, 1+t)

        x: Tensor = self.pre_dropout(x)
        x = self.encoder_tem(x, mask_n)
        x = x.view(b, n, 1+t, d)

        p1_cls = x[:, 0, 0, :].contiguous()
        p2_cls = x[:, 1, 0, :].contiguous()
        shuttle_cls = x[:, 2, 0, :].contiguous()

        p1 = x[:, 0, 1:, :].contiguous() + self.embedding_cross
        p2 = x[:, 1, 1:, :].contiguous() + self.embedding_cross
        shuttle = x[:, 2, 1:, :].contiguous() + self.embedding_cross

        cross_mask = mask[:, 1:].contiguous()
        p1_shuttle = self.cross_trans(p1, shuttle, cross_mask)
        p2_shuttle = self.cross_trans(p2, shuttle, cross_mask)

        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        p1_shuttle = torch.cat((class_token_inter, p1_shuttle), dim=1) + self.embedding_inter
        p2_shuttle = torch.cat((class_token_inter, p2_shuttle), dim=1) + self.embedding_inter

        p1_shuttle: Tensor = self.encoder_inter(p1_shuttle, mask)
        p2_shuttle: Tensor = self.encoder_inter(p2_shuttle, mask)

        p1_shuttle_cls = p1_shuttle[:, 0, :].contiguous()
        p2_shuttle_cls = p2_shuttle[:, 0, :].contiguous()

        p1_conclusion = p1_cls + p1_shuttle_cls
        p2_conclusion = p2_cls + p2_shuttle_cls

        x = torch.cat((p1_conclusion, p2_conclusion, shuttle_cls), dim=1)
        x = self.mlp_head(x)
        return x


class BST_5_4(nn.Module):
    '''BadmintonStrokeTransformer
    - BST_5_2
    - MLP_positions hd_dim from 256 changed to (in_dim * (mlp_d_scale=4))
    '''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=1,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
    ):
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.mlp_positions = MLP(2, out_dim=in_dim, hd_dim=(in_dim * mlp_d_scale), drop_p=0.25)

        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, [d_model // 2, d_model], tcn_kernel_size, drop_p)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)
        
        # MLP Head
        self.mlp_head = MLP_Head(d_model * 3, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem)
        self.embedding_tem.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_cross)
        self.embedding_cross.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        # Same as TemPose
        if isinstance(m, nn.Linear):
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,      # JnB: (b, t, n, input_dim)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        pos: Tensor,      # pos: (b, t, n, 2)
        video_len: Tensor  # video_len: (b)
    ):
        b, t, n, in_dim = JnB.shape
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)
        
        pos = self.mlp_positions(pos)
        pos_impact = pos.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)

        JnB = JnB * pos_impact + JnB

        JnB = self.tcn_pose(JnB)
        JnB = JnB.view(b, n, -1, t).transpose(-2, -1)

        shuttle = shuttle.transpose(1, 2).contiguous()
        shuttle = self.tcn_shuttle(shuttle)
        shuttle = shuttle.unsqueeze(1).transpose(-2, -1)
        
        x = torch.cat((JnB, shuttle), dim=1)
        _, n, _, d = x.shape

        class_token_tem = self.learned_token_tem.view(1, 1, -1).expand(b*n, -1, -1)
        x = x.view(b*n, t, d)
        x = torch.cat((class_token_tem, x), dim=1) + self.embedding_tem

        range_t = torch.arange(0, 1+t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, 1+t)
        mask_n = mask.repeat_interleave(n, dim=0)
        # mask_n: (b*n, 1+t)

        x: Tensor = self.pre_dropout(x)
        x = self.encoder_tem(x, mask_n)
        x = x.view(b, n, 1+t, d)

        p1_cls = x[:, 0, 0, :].contiguous()
        p2_cls = x[:, 1, 0, :].contiguous()
        shuttle_cls = x[:, 2, 0, :].contiguous()

        p1 = x[:, 0, 1:, :].contiguous() + self.embedding_cross
        p2 = x[:, 1, 1:, :].contiguous() + self.embedding_cross
        shuttle = x[:, 2, 1:, :].contiguous() + self.embedding_cross

        cross_mask = mask[:, 1:].contiguous()
        p1_shuttle = self.cross_trans(p1, shuttle, cross_mask)
        p2_shuttle = self.cross_trans(p2, shuttle, cross_mask)

        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        p1_shuttle = torch.cat((class_token_inter, p1_shuttle), dim=1) + self.embedding_inter
        p2_shuttle = torch.cat((class_token_inter, p2_shuttle), dim=1) + self.embedding_inter

        p1_shuttle: Tensor = self.encoder_inter(p1_shuttle, mask)
        p2_shuttle: Tensor = self.encoder_inter(p2_shuttle, mask)

        p1_shuttle_cls = p1_shuttle[:, 0, :].contiguous()
        p2_shuttle_cls = p2_shuttle[:, 0, :].contiguous()

        p1_conclusion = p1_cls + p1_shuttle_cls
        p2_conclusion = p2_cls + p2_shuttle_cls

        x = torch.cat((p1_conclusion, p2_conclusion, shuttle_cls), dim=1)
        x = self.mlp_head(x)
        return x


class BST_5_5(nn.Module):
    '''BadmintonStrokeTransformer
    - BST_5_2
    - MLP_positions hd_dim = in_dim // 2
    - MLP_positions drop_p = drop_p = 0.3
    '''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=1,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
    ):
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.mlp_positions = MLP(2, out_dim=in_dim, hd_dim=(in_dim // 2), drop_p=drop_p)

        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, [d_model // 2, d_model], tcn_kernel_size, drop_p)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)
        
        # MLP Head
        self.mlp_head = MLP_Head(d_model * 3, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem)
        self.embedding_tem.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_cross)
        self.embedding_cross.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        # Same as TemPose
        if isinstance(m, nn.Linear):
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,      # JnB: (b, t, n, input_dim)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        pos: Tensor,      # pos: (b, t, n, 2)
        video_len: Tensor  # video_len: (b)
    ):
        b, t, n, in_dim = JnB.shape
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)
        
        pos = self.mlp_positions(pos)
        pos_impact = pos.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)

        JnB = JnB * pos_impact + JnB

        JnB = self.tcn_pose(JnB)
        JnB = JnB.view(b, n, -1, t).transpose(-2, -1)

        shuttle = shuttle.transpose(1, 2).contiguous()
        shuttle = self.tcn_shuttle(shuttle)
        shuttle = shuttle.unsqueeze(1).transpose(-2, -1)
        
        x = torch.cat((JnB, shuttle), dim=1)
        _, n, _, d = x.shape

        class_token_tem = self.learned_token_tem.view(1, 1, -1).expand(b*n, -1, -1)
        x = x.view(b*n, t, d)
        x = torch.cat((class_token_tem, x), dim=1) + self.embedding_tem

        range_t = torch.arange(0, 1+t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, 1+t)
        mask_n = mask.repeat_interleave(n, dim=0)
        # mask_n: (b*n, 1+t)

        x: Tensor = self.pre_dropout(x)
        x = self.encoder_tem(x, mask_n)
        x = x.view(b, n, 1+t, d)

        p1_cls = x[:, 0, 0, :].contiguous()
        p2_cls = x[:, 1, 0, :].contiguous()
        shuttle_cls = x[:, 2, 0, :].contiguous()

        p1 = x[:, 0, 1:, :].contiguous() + self.embedding_cross
        p2 = x[:, 1, 1:, :].contiguous() + self.embedding_cross
        shuttle = x[:, 2, 1:, :].contiguous() + self.embedding_cross

        cross_mask = mask[:, 1:].contiguous()
        p1_shuttle = self.cross_trans(p1, shuttle, cross_mask)
        p2_shuttle = self.cross_trans(p2, shuttle, cross_mask)

        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        p1_shuttle = torch.cat((class_token_inter, p1_shuttle), dim=1) + self.embedding_inter
        p2_shuttle = torch.cat((class_token_inter, p2_shuttle), dim=1) + self.embedding_inter

        p1_shuttle: Tensor = self.encoder_inter(p1_shuttle, mask)
        p2_shuttle: Tensor = self.encoder_inter(p2_shuttle, mask)

        p1_shuttle_cls = p1_shuttle[:, 0, :].contiguous()
        p2_shuttle_cls = p2_shuttle[:, 0, :].contiguous()

        p1_conclusion = p1_cls + p1_shuttle_cls
        p2_conclusion = p2_cls + p2_shuttle_cls

        x = torch.cat((p1_conclusion, p2_conclusion, shuttle_cls), dim=1)
        x = self.mlp_head(x)
        return x


class MLP_with_Tanh(nn.Module):
    def __init__(self, in_dim, out_dim, hd_dim, drop_p=0.0) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hd_dim),
            nn.Tanh(),
            nn.Dropout(drop_p),
            nn.Linear(hd_dim, out_dim)
        )

    def forward(self, x: Tensor):
        return self.mlp(x)


class MLP_with_ReLU(nn.Module):
    def __init__(self, in_dim, out_dim, hd_dim, drop_p=0.0) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hd_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_p),
            nn.Linear(hd_dim, out_dim)
        )

    def forward(self, x: Tensor):
        return self.mlp(x)


class BST_5_6(nn.Module):
    '''BadmintonStrokeTransformer
    - BST_5_2
    - MLP_positions using Tanh as an activation function
    - MLP_positions drop_p = drop_p = 0.3
    '''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=1,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
    ):
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.mlp_positions = MLP_with_Tanh(2, out_dim=in_dim, hd_dim=256, drop_p=drop_p)

        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, [d_model // 2, d_model], tcn_kernel_size, drop_p)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)
        
        # MLP Head
        self.mlp_head = MLP_Head(d_model * 3, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem)
        self.embedding_tem.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_cross)
        self.embedding_cross.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        # Same as TemPose
        if isinstance(m, nn.Linear):
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,      # JnB: (b, t, n, input_dim)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        pos: Tensor,      # pos: (b, t, n, 2)
        video_len: Tensor  # video_len: (b)
    ):
        b, t, n, in_dim = JnB.shape
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)
        
        pos = self.mlp_positions(pos)
        pos_impact = pos.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)

        JnB = JnB * pos_impact + JnB

        JnB = self.tcn_pose(JnB)
        JnB = JnB.view(b, n, -1, t).transpose(-2, -1)

        shuttle = shuttle.transpose(1, 2).contiguous()
        shuttle = self.tcn_shuttle(shuttle)
        shuttle = shuttle.unsqueeze(1).transpose(-2, -1)
        
        x = torch.cat((JnB, shuttle), dim=1)
        _, n, _, d = x.shape

        class_token_tem = self.learned_token_tem.view(1, 1, -1).expand(b*n, -1, -1)
        x = x.view(b*n, t, d)
        x = torch.cat((class_token_tem, x), dim=1) + self.embedding_tem

        range_t = torch.arange(0, 1+t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, 1+t)
        mask_n = mask.repeat_interleave(n, dim=0)
        # mask_n: (b*n, 1+t)

        x: Tensor = self.pre_dropout(x)
        x = self.encoder_tem(x, mask_n)
        x = x.view(b, n, 1+t, d)

        p1_cls = x[:, 0, 0, :].contiguous()
        p2_cls = x[:, 1, 0, :].contiguous()
        shuttle_cls = x[:, 2, 0, :].contiguous()

        p1 = x[:, 0, 1:, :].contiguous() + self.embedding_cross
        p2 = x[:, 1, 1:, :].contiguous() + self.embedding_cross
        shuttle = x[:, 2, 1:, :].contiguous() + self.embedding_cross

        cross_mask = mask[:, 1:].contiguous()
        p1_shuttle = self.cross_trans(p1, shuttle, cross_mask)
        p2_shuttle = self.cross_trans(p2, shuttle, cross_mask)

        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        p1_shuttle = torch.cat((class_token_inter, p1_shuttle), dim=1) + self.embedding_inter
        p2_shuttle = torch.cat((class_token_inter, p2_shuttle), dim=1) + self.embedding_inter

        p1_shuttle: Tensor = self.encoder_inter(p1_shuttle, mask)
        p2_shuttle: Tensor = self.encoder_inter(p2_shuttle, mask)

        p1_shuttle_cls = p1_shuttle[:, 0, :].contiguous()
        p2_shuttle_cls = p2_shuttle[:, 0, :].contiguous()

        p1_conclusion = p1_cls + p1_shuttle_cls
        p2_conclusion = p2_cls + p2_shuttle_cls

        x = torch.cat((p1_conclusion, p2_conclusion, shuttle_cls), dim=1)
        x = self.mlp_head(x)
        return x


class BST_5_7(nn.Module):
    '''BadmintonStrokeTransformer
    - BST_5_2
    - MLP_positions using ReLU as an activation function
    - MLP_positions drop_p = drop_p = 0.3
    '''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=1,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
    ):
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.mlp_positions = MLP_with_ReLU(2, out_dim=in_dim, hd_dim=256, drop_p=drop_p)

        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, [d_model // 2, d_model], tcn_kernel_size, drop_p)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)
        
        # MLP Head
        self.mlp_head = MLP_Head(d_model * 3, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem)
        self.embedding_tem.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_cross)
        self.embedding_cross.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        # Same as TemPose
        if isinstance(m, nn.Linear):
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,      # JnB: (b, t, n, input_dim)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        pos: Tensor,      # pos: (b, t, n, 2)
        video_len: Tensor  # video_len: (b)
    ):
        b, t, n, in_dim = JnB.shape
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)
        
        pos = self.mlp_positions(pos)
        pos_impact = pos.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)

        JnB = JnB * pos_impact + JnB

        JnB = self.tcn_pose(JnB)
        JnB = JnB.view(b, n, -1, t).transpose(-2, -1)

        shuttle = shuttle.transpose(1, 2).contiguous()
        shuttle = self.tcn_shuttle(shuttle)
        shuttle = shuttle.unsqueeze(1).transpose(-2, -1)
        
        x = torch.cat((JnB, shuttle), dim=1)
        _, n, _, d = x.shape

        class_token_tem = self.learned_token_tem.view(1, 1, -1).expand(b*n, -1, -1)
        x = x.view(b*n, t, d)
        x = torch.cat((class_token_tem, x), dim=1) + self.embedding_tem

        range_t = torch.arange(0, 1+t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, 1+t)
        mask_n = mask.repeat_interleave(n, dim=0)
        # mask_n: (b*n, 1+t)

        x: Tensor = self.pre_dropout(x)
        x = self.encoder_tem(x, mask_n)
        x = x.view(b, n, 1+t, d)

        p1_cls = x[:, 0, 0, :].contiguous()
        p2_cls = x[:, 1, 0, :].contiguous()
        shuttle_cls = x[:, 2, 0, :].contiguous()

        p1 = x[:, 0, 1:, :].contiguous() + self.embedding_cross
        p2 = x[:, 1, 1:, :].contiguous() + self.embedding_cross
        shuttle = x[:, 2, 1:, :].contiguous() + self.embedding_cross

        cross_mask = mask[:, 1:].contiguous()
        p1_shuttle = self.cross_trans(p1, shuttle, cross_mask)
        p2_shuttle = self.cross_trans(p2, shuttle, cross_mask)

        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        p1_shuttle = torch.cat((class_token_inter, p1_shuttle), dim=1) + self.embedding_inter
        p2_shuttle = torch.cat((class_token_inter, p2_shuttle), dim=1) + self.embedding_inter

        p1_shuttle: Tensor = self.encoder_inter(p1_shuttle, mask)
        p2_shuttle: Tensor = self.encoder_inter(p2_shuttle, mask)

        p1_shuttle_cls = p1_shuttle[:, 0, :].contiguous()
        p2_shuttle_cls = p2_shuttle[:, 0, :].contiguous()

        p1_conclusion = p1_cls + p1_shuttle_cls
        p2_conclusion = p2_cls + p2_shuttle_cls

        x = torch.cat((p1_conclusion, p2_conclusion, shuttle_cls), dim=1)
        x = self.mlp_head(x)
        return x


class BST_5_8(nn.Module):
    '''BadmintonStrokeTransformer
    - BST_5_2
        - Without concatenate shuttle_cls before MLP_Head
    - Also equal to BST_7
        - Without calculating cosine simularities
        - MLP_positions drop_p = drop_p = 0.3
    '''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=1,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
    ):
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.mlp_positions = MLP(2, out_dim=in_dim, hd_dim=256, drop_p=0.25)

        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, [d_model // 2, d_model], tcn_kernel_size, drop_p)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)
        
        # MLP Head
        self.mlp_head = MLP_Head(d_model * 2, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem)
        self.embedding_tem.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_cross)
        self.embedding_cross.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        # Same as TemPose
        if isinstance(m, nn.Linear):
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,      # JnB: (b, t, n, input_dim)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        pos: Tensor,      # pos: (b, t, n, 2)
        video_len: Tensor  # video_len: (b)
    ):
        b, t, n, in_dim = JnB.shape
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)
        
        pos = self.mlp_positions(pos)
        pos_impact = pos.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)

        JnB = JnB * pos_impact + JnB

        JnB = self.tcn_pose(JnB)
        JnB = JnB.view(b, n, -1, t).transpose(-2, -1)

        shuttle = shuttle.transpose(1, 2).contiguous()
        shuttle = self.tcn_shuttle(shuttle)
        shuttle = shuttle.unsqueeze(1).transpose(-2, -1)
        
        x = torch.cat((JnB, shuttle), dim=1)
        _, n, _, d = x.shape

        class_token_tem = self.learned_token_tem.view(1, 1, -1).expand(b*n, -1, -1)
        x = x.view(b*n, t, d)
        x = torch.cat((class_token_tem, x), dim=1) + self.embedding_tem

        range_t = torch.arange(0, 1+t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, 1+t)
        mask_n = mask.repeat_interleave(n, dim=0)
        # mask_n: (b*n, 1+t)

        x: Tensor = self.pre_dropout(x)
        x = self.encoder_tem(x, mask_n)
        x = x.view(b, n, 1+t, d)

        p1_cls = x[:, 0, 0, :].contiguous()
        p2_cls = x[:, 1, 0, :].contiguous()

        p1 = x[:, 0, 1:, :].contiguous() + self.embedding_cross
        p2 = x[:, 1, 1:, :].contiguous() + self.embedding_cross
        shuttle = x[:, 2, 1:, :].contiguous() + self.embedding_cross

        cross_mask = mask[:, 1:].contiguous()
        p1_shuttle = self.cross_trans(p1, shuttle, cross_mask)
        p2_shuttle = self.cross_trans(p2, shuttle, cross_mask)

        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        p1_shuttle = torch.cat((class_token_inter, p1_shuttle), dim=1) + self.embedding_inter
        p2_shuttle = torch.cat((class_token_inter, p2_shuttle), dim=1) + self.embedding_inter

        p1_shuttle: Tensor = self.encoder_inter(p1_shuttle, mask)
        p2_shuttle: Tensor = self.encoder_inter(p2_shuttle, mask)

        p1_shuttle_cls = p1_shuttle[:, 0, :].contiguous()
        p2_shuttle_cls = p2_shuttle[:, 0, :].contiguous()

        p1_conclusion = p1_cls + p1_shuttle_cls
        p2_conclusion = p2_cls + p2_shuttle_cls

        x = torch.cat((p1_conclusion, p2_conclusion), dim=1)
        x = self.mlp_head(x)
        return x


class BST_5_9(nn.Module):
    '''BadmintonStrokeTransformer
    - BST_5_8
        - Without p1_cls, p2_cls
    '''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=1,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
    ):
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.mlp_positions = MLP(2, out_dim=in_dim, hd_dim=256, drop_p=0.25)

        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, [d_model // 2, d_model], tcn_kernel_size, drop_p)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)
        
        # MLP Head
        self.mlp_head = MLP_Head(d_model * 2, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem)
        self.embedding_tem.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_cross)
        self.embedding_cross.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        # Same as TemPose
        if isinstance(m, nn.Linear):
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,      # JnB: (b, t, n, input_dim)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        pos: Tensor,      # pos: (b, t, n, 2)
        video_len: Tensor  # video_len: (b)
    ):
        b, t, n, in_dim = JnB.shape
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)
        
        pos = self.mlp_positions(pos)
        pos_impact = pos.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)

        JnB = JnB * pos_impact + JnB

        JnB = self.tcn_pose(JnB)
        JnB = JnB.view(b, n, -1, t).transpose(-2, -1)

        shuttle = shuttle.transpose(1, 2).contiguous()
        shuttle = self.tcn_shuttle(shuttle)
        shuttle = shuttle.unsqueeze(1).transpose(-2, -1)
        
        x = torch.cat((JnB, shuttle), dim=1)
        _, n, _, d = x.shape

        class_token_tem = self.learned_token_tem.view(1, 1, -1).expand(b*n, -1, -1)
        x = x.view(b*n, t, d)
        x = torch.cat((class_token_tem, x), dim=1) + self.embedding_tem

        range_t = torch.arange(0, 1+t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, 1+t)
        mask_n = mask.repeat_interleave(n, dim=0)
        # mask_n: (b*n, 1+t)

        x: Tensor = self.pre_dropout(x)
        x = self.encoder_tem(x, mask_n)
        x = x.view(b, n, 1+t, d)

        p1 = x[:, 0, 1:, :].contiguous() + self.embedding_cross
        p2 = x[:, 1, 1:, :].contiguous() + self.embedding_cross
        shuttle = x[:, 2, 1:, :].contiguous() + self.embedding_cross

        cross_mask = mask[:, 1:].contiguous()
        p1_shuttle = self.cross_trans(p1, shuttle, cross_mask)
        p2_shuttle = self.cross_trans(p2, shuttle, cross_mask)

        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        p1_shuttle = torch.cat((class_token_inter, p1_shuttle), dim=1) + self.embedding_inter
        p2_shuttle = torch.cat((class_token_inter, p2_shuttle), dim=1) + self.embedding_inter

        p1_shuttle: Tensor = self.encoder_inter(p1_shuttle, mask)
        p2_shuttle: Tensor = self.encoder_inter(p2_shuttle, mask)

        p1_shuttle_cls = p1_shuttle[:, 0, :].contiguous()
        p2_shuttle_cls = p2_shuttle[:, 0, :].contiguous()

        x = torch.cat((p1_shuttle_cls, p2_shuttle_cls), dim=1)
        x = self.mlp_head(x)
        return x


class BST_6(nn.Module):
    '''BadmintonStrokeTransformer
    - BST_5_5
    - Adding Clean Gate
    '''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=1,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
    ):
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.mlp_positions = MLP(2, out_dim=in_dim, hd_dim=(in_dim // 2), drop_p=drop_p)

        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, [d_model // 2, d_model], tcn_kernel_size, drop_p)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)
        
        # Clean Gate
        self.mlp_clean = MLP(d_model, d_model, d_model, drop_p)

        # MLP Head
        self.mlp_head = MLP_Head(d_model * 3, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem)
        self.embedding_tem.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_cross)
        self.embedding_cross.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        # Same as TemPose
        if isinstance(m, nn.Linear):
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,      # JnB: (b, t, n, input_dim)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        pos: Tensor,      # pos: (b, t, n, 2)
        video_len: Tensor  # video_len: (b)
    ):
        b, t, n, in_dim = JnB.shape
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)
        
        pos = self.mlp_positions(pos)
        pos_impact = pos.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)

        JnB = JnB * pos_impact + JnB

        JnB = self.tcn_pose(JnB)
        JnB = JnB.view(b, n, -1, t).transpose(-2, -1)

        shuttle = shuttle.transpose(1, 2).contiguous()
        shuttle = self.tcn_shuttle(shuttle)
        shuttle = shuttle.unsqueeze(1).transpose(-2, -1)
        
        x = torch.cat((JnB, shuttle), dim=1)
        _, n, _, d = x.shape

        class_token_tem = self.learned_token_tem.view(1, 1, -1).expand(b*n, -1, -1)
        x = x.view(b*n, t, d)
        x = torch.cat((class_token_tem, x), dim=1) + self.embedding_tem

        range_t = torch.arange(0, 1+t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, 1+t)
        mask_n = mask.repeat_interleave(n, dim=0)
        # mask_n: (b*n, 1+t)

        x: Tensor = self.pre_dropout(x)
        x = self.encoder_tem(x, mask_n)
        x = x.view(b, n, 1+t, d)

        p1_cls = x[:, 0, 0, :].contiguous()
        p2_cls = x[:, 1, 0, :].contiguous()
        shuttle_cls = x[:, 2, 0, :].contiguous()

        p1 = x[:, 0, 1:, :].contiguous() + self.embedding_cross
        p2 = x[:, 1, 1:, :].contiguous() + self.embedding_cross
        shuttle = x[:, 2, 1:, :].contiguous() + self.embedding_cross

        cross_mask = mask[:, 1:].contiguous()
        p1_shuttle = self.cross_trans(p1, shuttle, cross_mask)
        p2_shuttle = self.cross_trans(p2, shuttle, cross_mask)

        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        p1_shuttle = torch.cat((class_token_inter, p1_shuttle), dim=1) + self.embedding_inter
        p2_shuttle = torch.cat((class_token_inter, p2_shuttle), dim=1) + self.embedding_inter

        p1_shuttle: Tensor = self.encoder_inter(p1_shuttle, mask)
        p2_shuttle: Tensor = self.encoder_inter(p2_shuttle, mask)

        p1_shuttle_cls = p1_shuttle[:, 0, :].contiguous()
        p2_shuttle_cls = p2_shuttle[:, 0, :].contiguous()

        # Clean Gate
        info_need_clean = torch.minimum(p1_shuttle_cls, p2_shuttle_cls)
        dirt = self.mlp_clean(info_need_clean)
        shuttle_cls = shuttle_cls - dirt

        p1_conclusion = p1_cls + p1_shuttle_cls
        p2_conclusion = p2_cls + p2_shuttle_cls

        x = torch.cat((p1_conclusion, p2_conclusion, shuttle_cls), dim=1)
        x = self.mlp_head(x)
        return x


class BST_6_2(nn.Module):
    '''BadmintonStrokeTransformer
    - BST_5_2
    - MLP_positions drop_p = drop_p = 0.3
    - Adding Clean Gate
    '''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=1,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
    ):
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.mlp_positions = MLP(2, out_dim=in_dim, hd_dim=256, drop_p=drop_p)

        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, [d_model // 2, d_model], tcn_kernel_size, drop_p)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)
        
        # Clean Gate
        self.mlp_clean = MLP(d_model, d_model, d_model, drop_p)

        # MLP Head
        self.mlp_head = MLP_Head(d_model * 3, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem)
        self.embedding_tem.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_cross)
        self.embedding_cross.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        # Same as TemPose
        if isinstance(m, nn.Linear):
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,      # JnB: (b, t, n, input_dim)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        pos: Tensor,      # pos: (b, t, n, 2)
        video_len: Tensor  # video_len: (b)
    ):
        b, t, n, in_dim = JnB.shape
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)
        
        pos = self.mlp_positions(pos)
        pos_impact = pos.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)

        JnB = JnB * pos_impact + JnB

        JnB = self.tcn_pose(JnB)
        JnB = JnB.view(b, n, -1, t).transpose(-2, -1)

        shuttle = shuttle.transpose(1, 2).contiguous()
        shuttle = self.tcn_shuttle(shuttle)
        shuttle = shuttle.unsqueeze(1).transpose(-2, -1)
        
        x = torch.cat((JnB, shuttle), dim=1)
        _, n, _, d = x.shape

        class_token_tem = self.learned_token_tem.view(1, 1, -1).expand(b*n, -1, -1)
        x = x.view(b*n, t, d)
        x = torch.cat((class_token_tem, x), dim=1) + self.embedding_tem

        range_t = torch.arange(0, 1+t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, 1+t)
        mask_n = mask.repeat_interleave(n, dim=0)
        # mask_n: (b*n, 1+t)

        x: Tensor = self.pre_dropout(x)
        x = self.encoder_tem(x, mask_n)
        x = x.view(b, n, 1+t, d)

        p1_cls = x[:, 0, 0, :].contiguous()
        p2_cls = x[:, 1, 0, :].contiguous()
        shuttle_cls = x[:, 2, 0, :].contiguous()

        p1 = x[:, 0, 1:, :].contiguous() + self.embedding_cross
        p2 = x[:, 1, 1:, :].contiguous() + self.embedding_cross
        shuttle = x[:, 2, 1:, :].contiguous() + self.embedding_cross

        cross_mask = mask[:, 1:].contiguous()
        p1_shuttle = self.cross_trans(p1, shuttle, cross_mask)
        p2_shuttle = self.cross_trans(p2, shuttle, cross_mask)

        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        p1_shuttle = torch.cat((class_token_inter, p1_shuttle), dim=1) + self.embedding_inter
        p2_shuttle = torch.cat((class_token_inter, p2_shuttle), dim=1) + self.embedding_inter

        p1_shuttle: Tensor = self.encoder_inter(p1_shuttle, mask)
        p2_shuttle: Tensor = self.encoder_inter(p2_shuttle, mask)

        p1_shuttle_cls = p1_shuttle[:, 0, :].contiguous()
        p2_shuttle_cls = p2_shuttle[:, 0, :].contiguous()

        # Clean Gate
        info_need_clean = torch.minimum(p1_shuttle_cls, p2_shuttle_cls)
        dirt = self.mlp_clean(info_need_clean)
        shuttle_cls = shuttle_cls - dirt

        p1_conclusion = p1_cls + p1_shuttle_cls
        p2_conclusion = p2_cls + p2_shuttle_cls

        x = torch.cat((p1_conclusion, p2_conclusion, shuttle_cls), dim=1)
        x = self.mlp_head(x)
        return x


class BST_7(nn.Module):
    '''BadmintonStrokeTransformer
    - ShuttlePose_6 (cosine simularity)
    - Players positions added to poses after a MLP layer
    '''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=1,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
    ):
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.mlp_positions = MLP(2, out_dim=in_dim, hd_dim=256, drop_p=drop_p)

        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, [d_model // 2, d_model], tcn_kernel_size, drop_p)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)
        
        # Cosine Simularity
        self.cos_sim = nn.CosineSimilarity()

        # MLP Head
        self.mlp_head = MLP_Head(d_model * 2, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem)
        self.embedding_tem.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_cross)
        self.embedding_cross.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        # Same as TemPose
        if isinstance(m, nn.Linear):
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,      # JnB: (b, t, n, input_dim)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        pos: Tensor,      # pos: (b, t, n, 2)
        video_len: Tensor  # video_len: (b)
    ):
        b, t, n, in_dim = JnB.shape
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)
        
        pos = self.mlp_positions(pos)
        pos_impact = pos.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)

        JnB = JnB * pos_impact + JnB

        JnB = self.tcn_pose(JnB)
        JnB = JnB.view(b, n, -1, t).transpose(-2, -1)

        shuttle = shuttle.transpose(1, 2).contiguous()
        shuttle = self.tcn_shuttle(shuttle)
        shuttle = shuttle.unsqueeze(1).transpose(-2, -1)
        
        x = torch.cat((JnB, shuttle), dim=1)
        _, n, _, d = x.shape

        class_token_tem = self.learned_token_tem.view(1, 1, -1).expand(b*n, -1, -1)
        x = x.view(b*n, t, d)
        x = torch.cat((class_token_tem, x), dim=1) + self.embedding_tem

        range_t = torch.arange(0, 1+t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, 1+t)
        mask_n = mask.repeat_interleave(n, dim=0)
        # mask_n: (b*n, 1+t)

        x: Tensor = self.pre_dropout(x)
        x = self.encoder_tem(x, mask_n)
        x = x.view(b, n, 1+t, d)

        p1_cls = x[:, 0, 0, :].contiguous()
        p2_cls = x[:, 1, 0, :].contiguous()
        shuttle_cls = x[:, 2, 0, :].contiguous()

        p1 = x[:, 0, 1:, :].contiguous() + self.embedding_cross
        p2 = x[:, 1, 1:, :].contiguous() + self.embedding_cross
        shuttle = x[:, 2, 1:, :].contiguous() + self.embedding_cross

        cross_mask = mask[:, 1:].contiguous()
        p1_shuttle = self.cross_trans(p1, shuttle, cross_mask)
        p2_shuttle = self.cross_trans(p2, shuttle, cross_mask)

        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        p1_shuttle = torch.cat((class_token_inter, p1_shuttle), dim=1) + self.embedding_inter
        p2_shuttle = torch.cat((class_token_inter, p2_shuttle), dim=1) + self.embedding_inter

        p1_shuttle: Tensor = self.encoder_inter(p1_shuttle, mask)
        p2_shuttle: Tensor = self.encoder_inter(p2_shuttle, mask)

        p1_shuttle_cls = p1_shuttle[:, 0, :].contiguous()
        p2_shuttle_cls = p2_shuttle[:, 0, :].contiguous()

        p1_conclusion = p1_cls + p1_shuttle_cls
        p2_conclusion = p2_cls + p2_shuttle_cls

        # Compute Cosine Simularities
        p1_shuttle_sim = self.cos_sim(p1_shuttle_cls, shuttle_cls)
        p2_shuttle_sim = self.cos_sim(p2_shuttle_cls, shuttle_cls)
        alpha: Tensor = (p1_shuttle_sim - p2_shuttle_sim + 2) / 4
        alpha = alpha.unsqueeze(1)

        p1_conclusion = alpha * p1_conclusion
        p2_conclusion = (1-alpha) * p2_conclusion

        x = torch.cat((p1_conclusion, p2_conclusion), dim=1)
        x = self.mlp_head(x)
        return x


class BST_7_2(nn.Module):
    '''BadmintonStrokeTransformer
    - BST_7
    - Using adding instead of concatenation before MLP_Head
    '''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=1,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
    ):
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.mlp_positions = MLP(2, out_dim=in_dim, hd_dim=256, drop_p=drop_p)

        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, [d_model // 2, d_model], tcn_kernel_size, drop_p)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)
        
        # Cosine Simularity
        self.cos_sim = nn.CosineSimilarity()

        # MLP Head
        self.mlp_head = MLP_Head(d_model, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem)
        self.embedding_tem.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_cross)
        self.embedding_cross.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        # Same as TemPose
        if isinstance(m, nn.Linear):
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,      # JnB: (b, t, n, input_dim)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        pos: Tensor,      # pos: (b, t, n, 2)
        video_len: Tensor  # video_len: (b)
    ):
        b, t, n, in_dim = JnB.shape
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)
        
        pos = self.mlp_positions(pos)
        pos_impact = pos.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)

        JnB = JnB * pos_impact + JnB

        JnB = self.tcn_pose(JnB)
        JnB = JnB.view(b, n, -1, t).transpose(-2, -1)

        shuttle = shuttle.transpose(1, 2).contiguous()
        shuttle = self.tcn_shuttle(shuttle)
        shuttle = shuttle.unsqueeze(1).transpose(-2, -1)
        
        x = torch.cat((JnB, shuttle), dim=1)
        _, n, _, d = x.shape

        class_token_tem = self.learned_token_tem.view(1, 1, -1).expand(b*n, -1, -1)
        x = x.view(b*n, t, d)
        x = torch.cat((class_token_tem, x), dim=1) + self.embedding_tem

        range_t = torch.arange(0, 1+t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, 1+t)
        mask_n = mask.repeat_interleave(n, dim=0)
        # mask_n: (b*n, 1+t)

        x: Tensor = self.pre_dropout(x)
        x = self.encoder_tem(x, mask_n)
        x = x.view(b, n, 1+t, d)

        p1_cls = x[:, 0, 0, :].contiguous()
        p2_cls = x[:, 1, 0, :].contiguous()
        shuttle_cls = x[:, 2, 0, :].contiguous()

        p1 = x[:, 0, 1:, :].contiguous() + self.embedding_cross
        p2 = x[:, 1, 1:, :].contiguous() + self.embedding_cross
        shuttle = x[:, 2, 1:, :].contiguous() + self.embedding_cross

        cross_mask = mask[:, 1:].contiguous()
        p1_shuttle = self.cross_trans(p1, shuttle, cross_mask)
        p2_shuttle = self.cross_trans(p2, shuttle, cross_mask)

        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        p1_shuttle = torch.cat((class_token_inter, p1_shuttle), dim=1) + self.embedding_inter
        p2_shuttle = torch.cat((class_token_inter, p2_shuttle), dim=1) + self.embedding_inter

        p1_shuttle: Tensor = self.encoder_inter(p1_shuttle, mask)
        p2_shuttle: Tensor = self.encoder_inter(p2_shuttle, mask)

        p1_shuttle_cls = p1_shuttle[:, 0, :].contiguous()
        p2_shuttle_cls = p2_shuttle[:, 0, :].contiguous()

        p1_conclusion = p1_cls + p1_shuttle_cls
        p2_conclusion = p2_cls + p2_shuttle_cls

        # Compute Cosine Simularities
        p1_shuttle_sim = self.cos_sim(p1_shuttle_cls, shuttle_cls)
        p2_shuttle_sim = self.cos_sim(p2_shuttle_cls, shuttle_cls)
        alpha: Tensor = (p1_shuttle_sim - p2_shuttle_sim + 2) / 4
        alpha = alpha.unsqueeze(1)

        x = alpha * p1_conclusion + (1-alpha) * p2_conclusion
        x = self.mlp_head(x)
        return x


class BST_7_3(nn.Module):
    '''BadmintonStrokeTransformer
    - BST_7
    - Concatenate shuttle_cls before MLP_Head
    '''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=1,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
    ):
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.mlp_positions = MLP(2, out_dim=in_dim, hd_dim=256, drop_p=drop_p)

        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, [d_model // 2, d_model], tcn_kernel_size, drop_p)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)
        
        # Cosine Simularity
        self.cos_sim = nn.CosineSimilarity()

        # MLP Head
        self.mlp_head = MLP_Head(d_model * 3, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem)
        self.embedding_tem.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_cross)
        self.embedding_cross.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        # Same as TemPose
        if isinstance(m, nn.Linear):
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,      # JnB: (b, t, n, input_dim)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        pos: Tensor,      # pos: (b, t, n, 2)
        video_len: Tensor  # video_len: (b)
    ):
        b, t, n, in_dim = JnB.shape
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)
        
        pos = self.mlp_positions(pos)
        pos_impact = pos.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)

        JnB = JnB * pos_impact + JnB

        JnB = self.tcn_pose(JnB)
        JnB = JnB.view(b, n, -1, t).transpose(-2, -1)

        shuttle = shuttle.transpose(1, 2).contiguous()
        shuttle = self.tcn_shuttle(shuttle)
        shuttle = shuttle.unsqueeze(1).transpose(-2, -1)
        
        x = torch.cat((JnB, shuttle), dim=1)
        _, n, _, d = x.shape

        class_token_tem = self.learned_token_tem.view(1, 1, -1).expand(b*n, -1, -1)
        x = x.view(b*n, t, d)
        x = torch.cat((class_token_tem, x), dim=1) + self.embedding_tem

        range_t = torch.arange(0, 1+t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, 1+t)
        mask_n = mask.repeat_interleave(n, dim=0)
        # mask_n: (b*n, 1+t)

        x: Tensor = self.pre_dropout(x)
        x = self.encoder_tem(x, mask_n)
        x = x.view(b, n, 1+t, d)

        p1_cls = x[:, 0, 0, :].contiguous()
        p2_cls = x[:, 1, 0, :].contiguous()
        shuttle_cls = x[:, 2, 0, :].contiguous()

        p1 = x[:, 0, 1:, :].contiguous() + self.embedding_cross
        p2 = x[:, 1, 1:, :].contiguous() + self.embedding_cross
        shuttle = x[:, 2, 1:, :].contiguous() + self.embedding_cross

        cross_mask = mask[:, 1:].contiguous()
        p1_shuttle = self.cross_trans(p1, shuttle, cross_mask)
        p2_shuttle = self.cross_trans(p2, shuttle, cross_mask)

        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        p1_shuttle = torch.cat((class_token_inter, p1_shuttle), dim=1) + self.embedding_inter
        p2_shuttle = torch.cat((class_token_inter, p2_shuttle), dim=1) + self.embedding_inter

        p1_shuttle: Tensor = self.encoder_inter(p1_shuttle, mask)
        p2_shuttle: Tensor = self.encoder_inter(p2_shuttle, mask)

        p1_shuttle_cls = p1_shuttle[:, 0, :].contiguous()
        p2_shuttle_cls = p2_shuttle[:, 0, :].contiguous()

        p1_conclusion = p1_cls + p1_shuttle_cls
        p2_conclusion = p2_cls + p2_shuttle_cls

        # Compute Cosine Simularities
        p1_shuttle_sim = self.cos_sim(p1_shuttle_cls, shuttle_cls)
        p2_shuttle_sim = self.cos_sim(p2_shuttle_cls, shuttle_cls)
        alpha: Tensor = (p1_shuttle_sim - p2_shuttle_sim + 2) / 4
        alpha = alpha.unsqueeze(1)

        p1_conclusion = alpha * p1_conclusion
        p2_conclusion = (1-alpha) * p2_conclusion

        x = torch.cat((p1_conclusion, p2_conclusion, shuttle_cls), dim=1)
        x = self.mlp_head(x)
        return x


class BST_8(nn.Module):
    '''BadmintonStrokeTransformer
    - Cosine Simularity and Clean Gate
    '''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=1,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
    ):
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.mlp_positions = MLP(2, out_dim=in_dim, hd_dim=256, drop_p=drop_p)

        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, [d_model // 2, d_model], tcn_kernel_size, drop_p)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)
        
        # Cosine Simularity
        self.cos_sim = nn.CosineSimilarity()

        # Clean Gate
        self.mlp_clean = MLP(d_model, d_model, d_model, drop_p)

        # MLP Head
        self.mlp_head = MLP_Head(d_model * 3, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem)
        self.embedding_tem.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_cross)
        self.embedding_cross.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        # Same as TemPose
        if isinstance(m, nn.Linear):
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,      # JnB: (b, t, n, input_dim)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        pos: Tensor,      # pos: (b, t, n, 2)
        video_len: Tensor  # video_len: (b)
    ):
        b, t, n, in_dim = JnB.shape
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)
        
        pos = self.mlp_positions(pos)
        pos_impact = pos.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)

        JnB = JnB * pos_impact + JnB

        JnB = self.tcn_pose(JnB)
        JnB = JnB.view(b, n, -1, t).transpose(-2, -1)

        shuttle = shuttle.transpose(1, 2).contiguous()
        shuttle = self.tcn_shuttle(shuttle)
        shuttle = shuttle.unsqueeze(1).transpose(-2, -1)
        
        x = torch.cat((JnB, shuttle), dim=1)
        _, n, _, d = x.shape

        class_token_tem = self.learned_token_tem.view(1, 1, -1).expand(b*n, -1, -1)
        x = x.view(b*n, t, d)
        x = torch.cat((class_token_tem, x), dim=1) + self.embedding_tem

        range_t = torch.arange(0, 1+t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, 1+t)
        mask_n = mask.repeat_interleave(n, dim=0)
        # mask_n: (b*n, 1+t)

        x: Tensor = self.pre_dropout(x)
        x = self.encoder_tem(x, mask_n)
        x = x.view(b, n, 1+t, d)

        p1_cls = x[:, 0, 0, :].contiguous()
        p2_cls = x[:, 1, 0, :].contiguous()
        shuttle_cls = x[:, 2, 0, :].contiguous()

        p1 = x[:, 0, 1:, :].contiguous() + self.embedding_cross
        p2 = x[:, 1, 1:, :].contiguous() + self.embedding_cross
        shuttle = x[:, 2, 1:, :].contiguous() + self.embedding_cross

        cross_mask = mask[:, 1:].contiguous()
        p1_shuttle = self.cross_trans(p1, shuttle, cross_mask)
        p2_shuttle = self.cross_trans(p2, shuttle, cross_mask)

        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        p1_shuttle = torch.cat((class_token_inter, p1_shuttle), dim=1) + self.embedding_inter
        p2_shuttle = torch.cat((class_token_inter, p2_shuttle), dim=1) + self.embedding_inter

        p1_shuttle: Tensor = self.encoder_inter(p1_shuttle, mask)
        p2_shuttle: Tensor = self.encoder_inter(p2_shuttle, mask)

        p1_shuttle_cls = p1_shuttle[:, 0, :].contiguous()
        p2_shuttle_cls = p2_shuttle[:, 0, :].contiguous()

        # Compute Cosine Simularities
        p1_shuttle_sim = self.cos_sim(p1_shuttle_cls, shuttle_cls)
        p2_shuttle_sim = self.cos_sim(p2_shuttle_cls, shuttle_cls)
        alpha: Tensor = (p1_shuttle_sim - p2_shuttle_sim + 2) / 4
        alpha = alpha.unsqueeze(1)

        p1_conclusion = p1_cls + p1_shuttle_cls
        p2_conclusion = p2_cls + p2_shuttle_cls

        p1_conclusion = alpha * p1_conclusion
        p2_conclusion = (1-alpha) * p2_conclusion

        # Clean Gate
        info_need_clean = torch.minimum(p1_shuttle_cls, p2_shuttle_cls)
        dirt = self.mlp_clean(info_need_clean)
        shuttle_cls = shuttle_cls - dirt

        x = torch.cat((p1_conclusion, p2_conclusion, shuttle_cls), dim=1)
        x = self.mlp_head(x)
        return x


class BST_8_2(nn.Module):
    '''BadmintonStrokeTransformer
    - Clean Gate before Cosine Simularity
    '''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=1,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
    ):
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.mlp_positions = MLP(2, out_dim=in_dim, hd_dim=256, drop_p=drop_p)

        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, [d_model // 2, d_model], tcn_kernel_size, drop_p)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)
        
        # Cosine Simularity
        self.cos_sim = nn.CosineSimilarity()

        # Clean Gate
        self.mlp_clean = MLP(d_model, d_model, d_model, drop_p)

        # MLP Head
        self.mlp_head = MLP_Head(d_model * 3, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem)
        self.embedding_tem.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_cross)
        self.embedding_cross.copy_(pos_encoding)

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        # Same as TemPose
        if isinstance(m, nn.Linear):
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,      # JnB: (b, t, n, input_dim)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        pos: Tensor,      # pos: (b, t, n, 2)
        video_len: Tensor  # video_len: (b)
    ):
        b, t, n, in_dim = JnB.shape
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)
        
        pos = self.mlp_positions(pos)
        pos_impact = pos.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)

        JnB = JnB * pos_impact + JnB

        JnB = self.tcn_pose(JnB)
        JnB = JnB.view(b, n, -1, t).transpose(-2, -1)

        shuttle = shuttle.transpose(1, 2).contiguous()
        shuttle = self.tcn_shuttle(shuttle)
        shuttle = shuttle.unsqueeze(1).transpose(-2, -1)
        
        x = torch.cat((JnB, shuttle), dim=1)
        _, n, _, d = x.shape

        class_token_tem = self.learned_token_tem.view(1, 1, -1).expand(b*n, -1, -1)
        x = x.view(b*n, t, d)
        x = torch.cat((class_token_tem, x), dim=1) + self.embedding_tem

        range_t = torch.arange(0, 1+t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, 1+t)
        mask_n = mask.repeat_interleave(n, dim=0)
        # mask_n: (b*n, 1+t)

        x: Tensor = self.pre_dropout(x)
        x = self.encoder_tem(x, mask_n)
        x = x.view(b, n, 1+t, d)

        p1_cls = x[:, 0, 0, :].contiguous()
        p2_cls = x[:, 1, 0, :].contiguous()
        shuttle_cls = x[:, 2, 0, :].contiguous()

        p1 = x[:, 0, 1:, :].contiguous() + self.embedding_cross
        p2 = x[:, 1, 1:, :].contiguous() + self.embedding_cross
        shuttle = x[:, 2, 1:, :].contiguous() + self.embedding_cross

        cross_mask = mask[:, 1:].contiguous()
        p1_shuttle = self.cross_trans(p1, shuttle, cross_mask)
        p2_shuttle = self.cross_trans(p2, shuttle, cross_mask)

        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        p1_shuttle = torch.cat((class_token_inter, p1_shuttle), dim=1) + self.embedding_inter
        p2_shuttle = torch.cat((class_token_inter, p2_shuttle), dim=1) + self.embedding_inter

        p1_shuttle: Tensor = self.encoder_inter(p1_shuttle, mask)
        p2_shuttle: Tensor = self.encoder_inter(p2_shuttle, mask)

        p1_shuttle_cls = p1_shuttle[:, 0, :].contiguous()
        p2_shuttle_cls = p2_shuttle[:, 0, :].contiguous()

        # Clean Gate
        info_need_clean = torch.minimum(p1_shuttle_cls, p2_shuttle_cls)
        dirt = self.mlp_clean(info_need_clean)
        shuttle_cls = shuttle_cls - dirt

        # Compute Cosine Simularities
        p1_shuttle_sim = self.cos_sim(p1_shuttle_cls, shuttle_cls)
        p2_shuttle_sim = self.cos_sim(p2_shuttle_cls, shuttle_cls)
        alpha: Tensor = (p1_shuttle_sim - p2_shuttle_sim + 2) / 4
        alpha = alpha.unsqueeze(1)

        p1_conclusion = p1_cls + p1_shuttle_cls
        p2_conclusion = p2_cls + p2_shuttle_cls

        p1_conclusion = alpha * p1_conclusion
        p2_conclusion = (1-alpha) * p2_conclusion

        x = torch.cat((p1_conclusion, p2_conclusion, shuttle_cls), dim=1)
        x = self.mlp_head(x)
        return x


if __name__ == '__main__':
    n_features = (17 + 19 * 2) * 2
    pose = torch.randn((1, 30, 2, n_features), dtype=torch.float)
    shuttle = torch.randn((1, 30, 2), dtype=torch.float)
    pos = torch.randn((1, 30, 2, 2), dtype=torch.float)
    videos_len = torch.tensor([30])
    input_data = [pose, shuttle, pos, videos_len]
    model = BST_8_2(
        in_dim=n_features,
        seq_len=30,
        n_class=35,
        d_model=100
    )
    summary(model, input_data=input_data, depth=4, device='cpu')
