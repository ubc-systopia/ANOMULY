from torch_geometric.nn.norm.batch_norm import BatchNorm
from torch_geometric.nn.conv import GCNConv, SAGEConv
from torch_geometric.nn import GAT, GraphSAGE, GATConv, ResGatedGraphConv
import torch.nn.functional as F
import torch.nn as nn
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class ANOMULY(nn.Module):
    def __init__(self, n_views, num_features, hidden_dims, gru_hidden_dims, attns_in_channels, attns_out_channels, n_layers=2) -> None:
        super(ANOMULY, self).__init__()  

        self.n_views = n_views
        self.n_layers = n_layers
        self.hidden_dims = hidden_dims
        self.gru_hidden_dims = gru_hidden_dims
        self.num_features = num_features
        self.graph_gcn_layers = [[] for _ in range(self.n_views)]
        self.graph_gru_layers = [[] for _ in range(self.n_views)]
        self.batch_norm = [[] for _ in range(self.n_views)]
        self.attns = []
        self.attns_in_channels = attns_in_channels
        self.attns_out_channels = attns_out_channels

        for layer in range(self.n_layers):     
            for view in range(self.n_views):
                    if layer == 0:
                        self.graph_gcn_layers[view].append(GCNConv(self.num_features, hidden_dims[layer]))
                    
                    else:
                        self.graph_gcn_layers[view][layer].append(GCNConv(gru_hidden_dims[layer - 1], hidden_dims[layer]))
                    
                    
                    self.graph_gru_layers[view].append(nn.GRU(input_size=hidden_dims[layer], hidden_size=self.gru_hidden_dims[layer]))
                    self.batch_norm[view].append(BatchNorm(hidden_dims[layer]))
            

            self.attns.append(MultiplexAttention(self.n_views, self.attns_in_channels[layer], self.attns_out_channels[layer]))

        self.act = F.relu()

        self.reset_parameters()      
                
    def forward(self, data, hiddens=None):
        x = data.x
        edge_index = data.edge_index
    
        if hiddens is None:
            for view in range(self.n_views):
                hiddens[view] = self.init_hidden(self.hidden_dim)
        
        for layer in range(self.n_layers):
            for view in range(self.n_views):
                x[view] = self.graph_gcn_layers[view][layer](x[view], edge_index[view])
                x[view] = self.batch_norm[view][layer](x[view])
                x[view] = self.act(x[view])
                x[view] = F.dropout(x[view], training=self.training)
            
                x[view], hiddens[view][layer] = self.graph_gru_layers[view][layer](x[view], hiddens[view][layer])

            attn_out = self.attns[layer](x)

            for view in range(self.n_views):
                x[view] = x[view] + attn_out
        

        x = torch.sigmoid(x)
        
        return x.squeeze(), hiddens






class SingleANOMULY(nn.Module):
    def __init__(self, num_features, hidden_dims, gru_hidden_dims, n_layers=2) -> None:
        super(SingleANOMULY, self).__init__()  

        self.n_layers = n_layers
        self.hidden_dims = hidden_dims
        self.gru_hidden_dims = gru_hidden_dims
        self.num_features = num_features
        self.graph_gcn_layers = []
        self.graph_gru_layers = []
        self.batch_norm = []

        for layer in range(self.n_layers):     
            if layer == 0:
                self.graph_gcn_layers.append(GCNConv(self.num_features, hidden_dims[layer]))
            
            else:
                self.graph_gcn_layers[layer].append(GCNConv(gru_hidden_dims[layer - 1], hidden_dims[layer]))
            
            
            self.graph_gru_layers.append(nn.GRU(input_size=hidden_dims[layer], hidden_size=self.gru_hidden_dims[layer]))
            self.batch_norm.append(BatchNorm(hidden_dims[layer]))
            

        self.act = F.relu()

        self.reset_parameters()      
                
    def forward(self, data, hiddens=None):
        x = data.x
        edge_index = data.edge_index
    
        if hiddens is None:
            hiddens = self.init_hidden(self.hidden_dim)
        
        for layer in range(self.n_layers):
            x = self.graph_gcn_layers[layer](x, edge_index)
            x = self.batch_norm[layer](x)
            x = self.act(x)
            x = F.dropout(x, training=self.training)
        
            x, hiddens[layer] = self.graph_gru_layers[layer](x, hiddens[layer])
        

        x = torch.sigmoid(x)
        
        return x.squeeze(), hiddens

        
        
    
    def init_hidden(self, dim):
        return torch.zeros((self.n_layers, dim)).to(device)



class MultiplexAttention(nn.Module):
    def __init__(self, n_views, in_channels, out_channels) -> None:
        super(MultiplexAttention, self).__init__()
        self.n_views = n_views
        self.linear = [nn.Linear(in_channels, out_channels) for _ in range(n_views)]
        self.activation = nn.Tanh()

        self.reset_parameters()

    def forward(self, x):
        # input size: views * batch_size * #nodes * in_channels
        out = torch.zeros_like(x)
        for view in range(self.n_views):
            out[view] = self.linear[view](x[view])
        
        
        s   = x.mean(dim=2) 
        s   = torch.unsqueeze(s, 2)

        out = torch.transpose(out, dim0=2, dim1=3) 

        out = torch.matmul(s, out)
        out = torch.squeeze(out, 2)
        out = self.activation(out)
        
        out = F.softmax(out, dim=0) 

        out = torch.transpose(out, dim0=0, dim1=1)
        out = torch.transpose(out, dim0=1, dim1=2)
        out = torch.unsqueeze(out, 2)
        x   = torch.transpose(x, dim0=0, dim1=1)
        x   = torch.transpose(x, dim0=1, dim1=2)
        out = torch.matmul(out, x)
        out = torch.squeeze(out, 2) # Size: batch_size * #nodes * in_channels

        return out