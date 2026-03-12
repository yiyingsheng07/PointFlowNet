import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, trans_dim=60):
        super(LearnablePositionalEncoding, self).__init__()
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, trans_dim),
        )
    
    def forward(self, x):
        pos_encodings = self.pos_embed(x) 
        return pos_encodings

class FarthestPointSampling(nn.Module):
    """Samples points using farthest point sampling"""
    def __init__(self, npoints):
        super(FarthestPointSampling, self).__init__()
        self.npoints = npoints
        
    def forward(self, xyz):
        """
        Input:
            xyz: pointcloud data, [B, N, 3]
        Output:
            centroids: sampled pointcloud index, [B, npoints]
        """
        device = xyz.device
        B, N, _ = xyz.shape
        
        centroids = torch.zeros(B, self.npoints, dtype=torch.long, device=device)
        distance = torch.ones(B, N, device=device) * 1e10
        
        # Initialize with a random point
        batch_indices = torch.arange(B, dtype=torch.long, device=device)
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
        
        for i in range(self.npoints):
            centroids[:, i] = farthest
            
            # Get distance to current farthest point
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, dim=-1)
            mask = dist < distance
            distance[mask] = dist[mask]
            
            # Find the farthest point
            farthest = torch.max(distance, dim=1)[1]
            
        return centroids

def knn_group(points, centroids_idx, xyz, k=48):
    """
    Group points around centroids using KNN
    Input:
        points: input points data, [B, N, C]
        centroids_idx: indices of centroids, [B, S]
        xyz: coordinates, [B, N, 3]
        k: number of neighbors
    Output:
        grouped_points: grouped points, [B, S, k, C]
        grouped_xyz: grouped coordinates, [B, S, k, 3]
    """
    batch_size, num_points, _ = points.shape
    num_centroids = centroids_idx.shape[1]
    device = points.device
    
    batch_indices = torch.arange(batch_size, device=device).view(-1, 1).repeat(1, num_centroids)
    centroids_xyz = xyz[batch_indices.flatten(), centroids_idx.flatten()].view(batch_size, num_centroids, 3)
    
    # Get distance from all points to all centroids
    dist = torch.cdist(centroids_xyz, xyz)  # [B, S, N]
    
    # Get the k nearest neighbors for each centroid
    _, knn_idx = torch.topk(dist, k, dim=2, largest=False, sorted=True)  # [B, S, k]
    
    # Reshape indices for gather operation
    batch_indices = torch.arange(batch_size, device=device).view(-1, 1, 1).repeat(1, num_centroids, k)
    
    # Gather the grouped points and coordinates
    grouped_points = points[batch_indices, knn_idx]  # [B, S, k, C]
    grouped_xyz = xyz[batch_indices, knn_idx]  # [B, S, k, 3]
    
    return grouped_points, grouped_xyz, centroids_xyz

class GroupedPointEncoder(nn.Module):
    def __init__(self, encoder_channel=512):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Linear(5, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128)
        )
        self.second_conv = nn.Sequential(
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.encoder_channel)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, c = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, c)
        # encoder
        feature = self.first_conv(point_groups)  # B*G N 128
        feature_global = torch.max(feature, dim=1, keepdim=True)[0]  # B*G 1 128
        feature = torch.cat([feature_global.expand(-1, n, -1), feature], dim=2)  # B*G n 256
        feature = self.second_conv(feature)  # B*G n encoder_channel
        feature_global = torch.max(feature, dim=1, keepdim=False)[0]  # B*G encoder_channel
        return feature_global.reshape(bs, g, self.encoder_channel)

class GlobalEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.first_layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128)
        )
        self.second_layer = nn.Sequential(
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512)
        )
        
    def forward(self, x):
        _, num_points, _ = x.shape
        feature1 = self.first_layer(x) # B N 128
        feature1_global = torch.max(feature1, dim=1, keepdim=False)[0] # B 128
        feature1_global = feature1_global.unsqueeze(1).expand(-1, num_points, -1)  # B N 128
        feature1_cat = torch.cat([feature1_global, feature1], dim=2) # B N 256
        feature2 = self.second_layer(feature1_cat) # B N 512

        return feature2

class GatedLocalFusion(nn.Module):
    def __init__(self, global_dim, local_dim):
        super().__init__()
        # Project local to match global dimension
        self.local_proj = nn.Sequential(
            nn.Linear(local_dim, global_dim),
            nn.LayerNorm(global_dim),
            nn.ReLU()
        )
        
        # The Gate: Takes Global + Local -> Outputs a weight between 0 and 1
        self.gate = nn.Sequential(
            nn.Linear(global_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, global_dim),
            nn.Sigmoid()  # Critical: outputs 0.0 to 1.0
        )

    def forward(self, global_feat, local_feat):
        # global_feat: [B, N, 1024] (combined global + point features)
        # local_feat:  [B, N, 512]  (the grouping output)
        
        # 1. Align dimensions
        local_feat_aligned = self.local_proj(local_feat)
        
        # 2. Calculate the Gate
        # "Based on what I know globally and locally, how much should I trust local?"
        gate_input = torch.cat([global_feat, local_feat_aligned], dim=-1)
        alpha = self.gate(gate_input)
        
        # 3. Fuse
        # Base feature + (Weight * Local Correction)
        return global_feat + (alpha * local_feat_aligned)

class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()

        # Dimensions
        self.trans_dim = 60
        self.in_dim = 3
        self.in_dim_pv = 4  # xyz + distance
        self.encoder_channel = 512
        
        # Positional encoding
        self.positional_encoding = LearnablePositionalEncoding(trans_dim=self.trans_dim)
        
        # Group configuration for pvdata
        self.num_centroids = 1024  # Number of centroids to sample
        self.fps = FarthestPointSampling(npoints=self.num_centroids)
        self.k_neighbors = 48  # Neighbors per centroid for 60k points

        self.num_nearest_centroids = 6            # 6–8 is a good range
        self.attn_proj = nn.Linear(
                self.encoder_channel,             # W in  Σ αᵢ W gᵢ
                self.encoder_channel,
                bias=False)
        
        # Encoder for pvdata groups
        self.pvdata_encoder = GlobalEncoder(self.in_dim_pv + self.trans_dim)
        self.pvgroup_encoder = GroupedPointEncoder(encoder_channel=self.encoder_channel)

        self.fusion = GatedLocalFusion(global_dim=1024, local_dim=512)
        
        # Decoder layers (maintain original structure)
        self.layer4 = nn.Sequential(
            nn.Linear(self.encoder_channel * 2, 1024),  # mdata features + pvdata features
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True)
        )
        
        self.layer5 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.01)
        )
        
        self.layer6 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01)
        )
        
        self.layer7 = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(0.01)
        )
        
        self.logits = nn.Linear(64, 3)  # Output for velocity vectors (u,v,w)
    
    @staticmethod
    def gather_by_index(feat, idx):
        """
        feat : [B, C_centroids, C_feat]
        idx  : [B, N_pts, k]         (values ∈ [0, C_centroids-1])
        return: [B, N_pts, k, C_feat]
        """
        B, N, k = idx.shape
        C_feat  = feat.size(2)

        # 1) flatten the (N, k) block so idx ranks == feat ranks (3)
        idx_flat = idx.reshape(B, -1)                     # [B, N*k]

        # 2) expand last dim so gather keeps C_feat
        idx_exp  = idx_flat.unsqueeze(-1).expand(-1, -1, C_feat)   # [B, N*k, C_feat]

        # 3) gather along dim=1 (centroid dimension)
        gathered = torch.gather(feat, 1, idx_exp)         # [B, N*k, C_feat]

        # 4) restore (N, k) grid
        return gathered.reshape(B, N, k, C_feat)

    def get_features_from_k_nearest_centroids(
            self, pvdata, centroids_xyz, pvgroup_features, k=8):
        """
        Attention-pool the k (≈8) closest centroids for every interior point.
        Args:
        pvdata           : [B, N, 3]
        centroids_xyz    : [B, C, 3]
        pvgroup_features : [B, C, E]
        Returns:
        point_features   : [B, N, E]
        """
        # 1. pair-wise distances
        dist, idx = torch.topk(          # [B, N, k]
            torch.cdist(pvdata[:, :, :3], centroids_xyz), k, dim=2, largest=False)

        # 2. αᵢ  =  softmax( -dᵢ )
        attn = torch.softmax(-dist, dim=2)                          # [B, N, k]

        # 3. gather centroid features → project → weighted sum
        cent_feat = self.gather_by_index(pvgroup_features, idx)          # [B, N, k, E]
        cent_feat = self.attn_proj(cent_feat)                       # [B, N, k, E]
        out = (attn.unsqueeze(-1) * cent_feat).sum(2)               # [B, N, E]
        return out

    def forward(self, mdata, pvdata):
        batch_size, num_points, _ = pvdata.shape

        # Process pvdata with global featues
        pe_pvdata = self.positional_encoding(pvdata[:, :, :3])
        pvdata_with_pe = torch.cat([pvdata[:, :, :self.in_dim_pv], pe_pvdata], dim=-1)
        pvdata_features = self.pvdata_encoder(pvdata_with_pe)  # [B, N, encoder_channel]

        # Global pooling for pvdata features
        pvdata_features2 = pvdata_features.transpose(1, 2)  # [B, encoder_channel, N]
        pvdata_global = torch.max(pvdata_features2, dim=2, keepdim=False)[0]  # [B, encoder_channel]
        pvdata_global = pvdata_global.unsqueeze(1).repeat(1, num_points, 1)  # [B, N, encoder_channel]
        
        # Process pvdata with FPS and KNN grouping
        # Sample centroids using FPS
        centroids_idx = self.fps(pvdata[:, :, :3])  # [B, 1024]
        
        # Group points around centroids
        grouped_pvdata, grouped_xyz, centroids_xyz = knn_group(
            pvdata, centroids_idx, pvdata[:, :, :3], k=self.k_neighbors
        )  # [B, 1024, 64, 3]

        # ---------- add distance‑to‑centroid ----------
        # |grouped_xyz - centroid|_2  → (B,S,k,1)
        d_centroid = torch.norm(grouped_xyz - centroids_xyz.unsqueeze(2), dim=-1, keepdim=True)

        grouped_pvdata = torch.cat([grouped_pvdata, d_centroid], dim=-1)   # (B,S,k,5)
        
        # Extract features from grouped points using Conv1d encoder
        pvgroup_features = self.pvgroup_encoder(grouped_pvdata)  # [B, 1024, encoder_channel]
        
        # Get features from k nearest centroids for each point
        pvdata_point_features = self.get_features_from_k_nearest_centroids(
            pvdata[:, :, :3], centroids_xyz, pvgroup_features, k=self.num_nearest_centroids
        )  # [B, N, encoder_channel]
        
        # Concatenate mdata and pvdata features
        global_context = torch.cat([pvdata_global, pvdata_features], dim=-1) # [B, N, 1024]
        fused_features = self.fusion(global_context, pvdata_point_features)
        
        # Decoder layers
        out = self.layer4(fused_features)  # [B, N, 1024]
        out = self.layer5(out)  # [B, N, 512]
        out = self.layer6(out)  # [B, N, 256]
        out = self.layer7(out)  # [B, N, 64]
        out = self.logits(out)  # [B, N, 3]
        
        return out  # [B, N, 3] - Output velocity vectors