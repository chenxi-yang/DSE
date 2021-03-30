import torch
import torch.nn.functional as F
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(self.out_channels))
    
    # def forward(self, x_list):
    #     res_list = list()
    #     for x in x_list:
    #         # print(x.c)
    #         # print(x.delta)
    #         new_x = x.matmul(self.weight).add(self.bias)
    #         res_list.append(new_x)
    #     return res_list

    def forward(self, x):
        return x.matmul(self.weight).add(self.bias)


class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    # def forward(self, x_list):
    #     res_list = list()
    #     for x in x_list:
    #         res_list.append(x.sigmoid())
    #     return res_list
    def forward(self, x):
        return x.sigmoid()


# class While(nn.Module):
#     def __init__(self):

class Assign(nn.Module):
    def __init__(self, target_idx, f):
        super().__init__()
        self.target_idx = target_idx
        self.f = f
    
    def forward(self, x_list):
        res_list = calculate_x_list(self.target_idx, self.f, x_list)
        return res_list


class IfElse(nn.Module):
    def __init__(self, target_idx, test, f_test, body, orelse):
        super().__init__()
        self.target_idx = target_idx
        self.test = test
        self.f_test = f_test
        self.body = body
        self.orelse = orelse
    
    def forward(self, x_list):
        res_list = list()
        body_list = calculate_branch_list(self.target_idx, self.f_test(self.test), x_list, '<')
        orelse = calculate_branch_list(self.target_idx, self.f_test(self.test), x_list, '>')

        body_list = self.body(body_list)
        res_list.extend(body_list)

        orelse_list = self.orelse(orelse_list)
        res_list.extend(orelse_list)

        return res_list


class While(nn.Module):
    def __init__(self, target_idx, test, body):
        super().__init__()
        self.target_idx = target_idx
        self.test = test
        self.body = body
    
    def forward(self, x_list):
        res_list = list()

        while(len(x_list) > 0):
            body_list = calculate_branch_list(self.target_idx, self.test, x_list, '<')
            orelse_list= calculate_branch_list(self.target_idx, self.test, x_list, '>')

            res_list.extend(orelse_list)

            if len(body_list) > 0:
                x_list = self.body(body_list)
        
        return x_list
            
            

            

            









