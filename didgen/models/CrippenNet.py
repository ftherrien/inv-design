import torch
import torch.nn as nn

CRIPPEN_PARAMS = [0.1441, 
                  0.0000 ,
                  -0.2035, 
                  -0.2051, 
                  -0.2783, 
                  0.1551 ,
                  0.00170, 
                  0.08452, 
                  -0.1444, 
                  -0.0516, 
                  0.1193 ,
                  -0.0967, 
                  -0.5443, 
                  0.0000 ,
                  0.2450 ,
                  0.1980 ,
                  0.0000 ,
                  0.1581 ,
                  0.2955 ,
                  0.2713 ,
                  0.1360 ,
                  0.4619 ,
                  0.5437 ,
                  0.1893 ,
                  -0.8186, 
                  0.2640 ,
                  0.2148 ,
                  0.08129,                  
                  0.1230 ,
                  -0.2677, 
                  0.2142 ,
                  0.2980 ,
                  0.1125 ,
                  -1.0190, 
                  -0.7096, 
                  -1.0270, 
                  -0.5188, 
                  0.08387, 
                  0.1836 ,
                  -0.3187, 
                  -0.4458, 
                  0.01508, 
                  -1.950 ,
                  -0.3239, 
                  -1.119 ,
                  -0.3396, 
                  0.2887 ,
                  -0.4806,
                  0.1552 ,
                  -0.2893, 
                  -0.0684, 
                  -0.4195, 
                  0.0335 ,
                  -0.3339, 
                  -1.189 ,
                  0.1788 ,
                  -0.1526, 
                  0.1129 ,
                  0.4833 ,
                  -1.326 ,
                  -0.1188,
                  0.4202 ,
                  0.6895 ,
                  0.8456 ,
                  0.8857 ,
                  -2.996 ,
                  0.8612 ,
                  0.6482 ,
                  -0.0024, 
                  0.6237 ,
                  -0.3808,
                  -0.0025]

C_PARAMS = [0.1441, 
            0.0000 ,
            -0.2035, 
            -0.2051, 
            -0.2783, 
            0.1551 ,
            0.00170, 
            0.08452, 
            -0.1444, 
            -0.0516, 
            0.1193 ,
            -0.0967, 
            -0.5443, 
            0.0000 ,
            0.2450 ,
            0.1980 ,
            0.0000 ,
            0.1581 ,
            0.2955 ,
            0.2713 ,
            0.1360 ,
            0.4619 ,
            0.5437 ,
            0.1893 ,
            -0.8186, 
            0.2640 ,
            0.2148 ,
            0.08129]

H_PARAMS = [0.1230 ,
            -0.2677, 
            0.2142 ,
            0.2980 ,
            0.1125]

N_PARAMS = [-1.0190, 
            -0.7096, 
            -1.0270, 
            -0.5188, 
            0.08387, 
            0.1836 ,
            -0.3187, 
            -0.4458, 
            0.01508, 
            -1.950 ,
            -0.3239, 
            -1.119 ,
            -0.3396, 
            0.2887 ,
            -0.4806]

O_PARAMS = [0.1552 ,
            -0.2893, 
            -0.0684, 
            -0.4195, 
            0.0335 ,
            -0.3339, 
            -1.189 ,
            0.1788 ,
            -0.1526, 
            0.1129 ,
            0.4833 ,
            -1.326 ,
            -0.1188]

F_PARAM = 0.4202

Cl_PARAM = 0.6895

Br_PARAM = 0.8456

I_PARAM = 0.8857

Hal_PARAM = -2.996

P_PARAM = 0.8612

S_PARAMS = [0.6482 ,
            -0.0024, 
            0.6237]

Me_PARAMS = [-0.3808,
             -0.0025]

zinc_PARAMS = C_PARAMS + O_PARAMS + N_PARAMS + [F_PARAM] + H_PARAMS + S_PARAMS + [Cl_PARAM] + [Br_PARAM] + [I_PARAM] + [P_PARAM]

class CrippenNet(nn.Module):

    def sequence(self, orig_atom_fea_len, layer_list, out_len):
        seq = (nn.Linear(orig_atom_fea_len, layer_list[0]), self.nonlinear)
        for i in range(len(layer_list)-1):
            seq += (nn.Linear(layer_list[i], layer_list[i+1]), self.nonlinear)
        seq += (nn.Linear(layer_list[-1], out_len),)

        return seq
        
        
    def __init__(self, orig_atom_fea_len, n_conv=6, layer_list=[144], classifier=False):

        super(CrippenNet, self).__init__()

        self.classifier = classifier
        
        self.fea_len = orig_atom_fea_len
        
        self.nonlinear = nn.Sigmoid()

        self.softmax = nn.Softmax(dim=2)
        
        self.n_conv = n_conv
        
        self.C_NN = nn.Sequential(*self.sequence(orig_atom_fea_len*n_conv, layer_list, len(C_PARAMS)))
        self.O_NN = nn.Sequential(*self.sequence(orig_atom_fea_len*n_conv, layer_list, len(O_PARAMS)))
        self.N_NN = nn.Sequential(*self.sequence(orig_atom_fea_len*n_conv, layer_list, len(N_PARAMS)))
        self.H_NN = nn.Sequential(*self.sequence(orig_atom_fea_len*n_conv, layer_list, len(H_PARAMS)))
        self.S_NN = nn.Sequential(*self.sequence(orig_atom_fea_len*n_conv, layer_list, len(S_PARAMS)))

        
    def forward(self, atom_fea, adj):

        atom_fea = atom_fea[:,:,:self.fea_len]

        atom_feas = [atom_fea]
        for _ in range(self.n_conv):
            atom_feas.append(adj.matmul(atom_feas[-1]))

        atom_feas = torch.cat(atom_feas[1:], dim=2)
            
        C_type = self.softmax(self.C_NN(atom_feas)) # 28
        O_type = self.softmax(self.O_NN(atom_feas)) # 13
        N_type = self.softmax(self.N_NN(atom_feas)) # 15
        H_type = self.softmax(self.H_NN(atom_feas)) # 5
        S_type = self.softmax(self.S_NN(atom_feas)) # 3
        ones = torch.ones(atom_fea.shape[:2], device = atom_fea.device).unsqueeze(2)

        if self.classifier:
            return torch.cat([C_type*atom_fea[:,:,0:1].expand(C_type.shape),
                              O_type*atom_fea[:,:,1:2].expand(O_type.shape),
                              N_type*atom_fea[:,:,2:3].expand(N_type.shape),
                              ones*atom_fea[:,:,3:4],
                              H_type*atom_fea[:,:,4:5].expand(H_type.shape),
                              S_type*atom_fea[:,:,5:6].expand(S_type.shape),
                              ones*atom_fea[:,:,6:7],
                              ones*atom_fea[:,:,7:8],
                              ones*atom_fea[:,:,8:9],
                              ones*atom_fea[:,:,9:10]], dim=2)
            
        else:
            
            crippen_values = atom_fea*torch.cat([C_type.matmul(torch.tensor(C_PARAMS, device = atom_fea.device)).unsqueeze(2),
                                                        O_type.matmul(torch.tensor(O_PARAMS, device = atom_fea.device)).unsqueeze(2),
                                                        N_type.matmul(torch.tensor(N_PARAMS, device = atom_fea.device)).unsqueeze(2),
                                                        F_PARAM*torch.ones(atom_fea.shape[:2], device = atom_fea.device).unsqueeze(2),
                                                        H_type.matmul(torch.tensor(H_PARAMS, device = atom_fea.device)).unsqueeze(2),
                                                        S_type.matmul(torch.tensor(S_PARAMS, device = atom_fea.device)).unsqueeze(2),
                                                        Cl_PARAM*torch.ones(atom_fea.shape[:2], device = atom_fea.device).unsqueeze(2),
                                                        Br_PARAM*torch.ones(atom_fea.shape[:2], device = atom_fea.device).unsqueeze(2),
                                                        I_PARAM*torch.ones(atom_fea.shape[:2], device = atom_fea.device).unsqueeze(2),
                                                        P_PARAM*torch.ones(atom_fea.shape[:2], device = atom_fea.device).unsqueeze(2)], dim=2)
                                             
            crippen_values = torch.sum(crippen_values, dim=2)
            
            return torch.sum(crippen_values, dim=1, keepdim=True)
