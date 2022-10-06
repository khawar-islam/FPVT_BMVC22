from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import math
from torch.nn import Parameter


class DictArcMarginProduct(nn.Module):
    def __init__(self, in_features=128, out_features=200, out_features_test=74974, s=32.0, m=0.50, label_dict=None,
                 easy_margin=False):
        super(DictArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
        self.weight_test = Parameter(torch.Tensor(out_features_test, in_features), requires_grad=False)
        nn.init.xavier_uniform_(self.weight)
        # init.kaiming_uniform_()
        # self.weight.data.normal_(std=0.001)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.label_dict = label_dict

        self.need_back_label_list = []
        self.first_test = True

    def forward(self, x, label, label_set, testing=False):
        if testing:
            return self.forward_test(x, label, label_set)
        else:
            return self.forward_train(x, label, label_set)

    def forward_train(self, x, label, label_set):
        # import time
        # start_time = time.time()
        for label_id in self.need_back_label_list:
            self.weight.data[label_id] = self.weight_test[label_id].clone()
            self.weight_test[label_id] = 0
        self.need_back_label_list.clear()
        # print('recode time = ', time.time()-start_time)
        # start_time = time.time()

        label_new = label.clone() * 0 - 1
        # import pdb
        # pdb.set_trace()
        used_ind_set = set()
        filter_label_set = set()
        for ind, label_id in enumerate(label_set):
            assert label_id in self.label_dict.keys(), label_id
            assert self.label_dict[label_id] < self.out_features, \
                'self.label_dict[label_id] < self.out_features,{} vs {},label_id={}' \
                    .format(self.label_dict[label_id], self.out_features, label_id)

            if self.label_dict[label_id] not in used_ind_set:
                mask = label == label_id
                # label_new[mask] = self.label_dict[label_id]
                label_new.masked_fill_(mask, self.label_dict[label_id])
                self.weight.data[self.label_dict[label_id]] = torch.sum(x[mask].detach(), dim=0)
                used_ind_set.add(self.label_dict[label_id])
            else:
                filter_label_set.add(label_id)

        # print('mapping time = ', time.time()-start_time)
        # start_time = time.time()
        candidate_set = set([i for i in range(self.out_features)]) - used_ind_set
        candidate_set = list(candidate_set)
        sta = 0
        for label_id in filter_label_set:
            assert sta < len(candidate_set)
            mask = label == label_id
            label_new.masked_fill_(mask, candidate_set[sta])
            # label_new[mask] = candidate_set[sta]
            self.weight_test[candidate_set[sta]] = self.weight[candidate_set[sta]].clone()
            self.weight.data[candidate_set[sta]] = torch.sum(x[mask].detach(), dim=0)
            self.need_back_label_list.append(candidate_set[sta])
            sta += 1

        # print('candidate time = ', time.time()-start_time)
        # start_time = time.time()
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        sine = 1.0 - torch.pow(cosine, 2)
        sine = torch.where(sine > 0, sine, torch.zeros(sine.size(), device='cuda'))
        sine = torch.sqrt(sine)
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label_new.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        # print('end time = ', time.time()-start_time)
        # start_time = time.time()
        return output, label_new

    def forward_test(self, x, label, label_set):
        for label_id in self.need_back_label_list:
            self.weight.data[label_id] = self.weight_test[label_id].clone()
        self.need_back_label_list.clear()

        with torch.no_grad():
            self.weight_test.data = self.weight.clone()
        label_new = label.clone()
        used_ind_set = set()
        filter_label_set = set()
        for ind, label_id in enumerate(label_set):
            assert label_id in self.label_dict.keys(), label_id
            assert self.label_dict[label_id] < self.out_features, \
                'self.label_dict[label_id] < self.out_features,{} vs {},label_id={}' \
                    .format(self.label_dict[label_id], self.out_features, label_id)

            if self.label_dict[label_id] not in used_ind_set:
                mask = label == label_id
                label_new[mask] = self.label_dict[label_id]
                self.weight_test[self.label_dict[label_id]] = torch.sum(x[mask].detach(), dim=0)
                used_ind_set.add(self.label_dict[label_id])
            else:
                filter_label_set.add(label_id)

        candidate_set = set([i for i in range(self.out_features)]) - used_ind_set
        candidate_set = list(candidate_set)
        sta = 0
        for label_id in filter_label_set:
            assert sta < len(candidate_set)
            mask = label == label_id
            label_new[mask] = candidate_set[sta]
            self.weight_test[candidate_set[sta]] = torch.sum(x[mask].detach(), dim=0)
            self.need_back_label_list.append(candidate_set[sta])
            sta += 1

        cosine = F.linear(F.normalize(x), F.normalize(self.weight_test))
        sine = 1.0 - torch.pow(cosine, 2)
        sine = torch.where(sine > 0, sine, torch.zeros(sine.size(), device='cuda'))
        sine = torch.sqrt(sine)
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label_new.view(-1, 1).long(), 1)
        output0 = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output0 *= self.s

        output1 = cosine
        output1 *= self.s
        return (output0, output1), label_new


class NoiseWeighting(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NoiseWeighting, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim * output_dim, 512),
            nn.Linear(512, output_dim),
            nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )
        self.output_dim = output_dim
        self.input_dim = input_dim

    def forward(self, x, y):
        y_np = y.cpu().numpy()
        assert len(set(y_np)) == 1
        indices = []
        for y_c in set(y_np):
            indices_c = np.where(y_np == y_c)[0]
            # assert len(indices_c) % self.output_dim ==0, '{} vs {}'.format(len(indices_c), self.output_dim)
            # assert len(indices_c) % self.output_dim ==0, '{} vs {}'.format(len(indices_c), self.output_dim)
            if len(indices_c) % self.output_dim != 0:
                continue
            indices.extend(indices_c)
        indices = np.array(indices).reshape(-1)
        # import pdb
        # pdb.set_trace()

        x_new = x[indices]
        x_new = x_new.view(-1, self.input_dim * self.output_dim)
        y_new = y[indices]
        y_new = y_new.view(-1, self.output_dim)[:, 0]

        score = self.block(x_new)
        score = score.view(-1, self.output_dim, 1)
        x_new = x_new.view(-1, self.output_dim, self.input_dim)
        return_x = score * x_new  # F.linear(score, x_new)
        return_x = return_x.mean(dim=1)
        return_y = y_new.view(-1)
        # for s, x_n in zip(score, x_new):
        #    x_n = x_n.view(output_dim,-1)
        #    x_list.append(F.linear(s, x_n))

        return return_x.mean(dim=0)


class DictArcMarginProduct_Reweight(nn.Module):
    def __init__(self, in_features=128, out_features=200, out_features_test=74974, s=32.0, m=0.50, label_dict=None,
                 easy_margin=False, n_sample=4):
        super(DictArcMarginProduct_Reweight, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
        self.weight_test = Parameter(torch.Tensor(out_features_test, in_features), requires_grad=False)
        nn.init.xavier_uniform_(self.weight)
        # init.kaiming_uniform_()
        # self.weight.data.normal_(std=0.001)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.label_dict = label_dict

        self.need_back_label_list = []
        self.first_test = True

        self.ensemble_feature = NoiseWeighting(in_features, n_sample)

    def forward(self, x, label, label_set, testing=False):
        if testing:
            return self.forward_test(x, label, label_set)
        else:
            return self.forward_train(x, label, label_set)

    def forward_train(self, x, label, label_set):
        # import time
        # start_time = time.time()
        for label_id in self.need_back_label_list:
            self.weight.data[label_id] = self.weight_test[label_id].clone()
            self.weight_test[label_id] = 0
        self.need_back_label_list.clear()
        # print('recode time = ', time.time()-start_time)
        # start_time = time.time()

        label_new = label.clone() * 0 - 1
        # import pdb
        # pdb.set_trace()
        used_ind_set = set()
        filter_label_set = set()
        for ind, label_id in enumerate(label_set):
            assert label_id in self.label_dict.keys(), label_id
            assert self.label_dict[label_id] < self.out_features, \
                'self.label_dict[label_id] < self.out_features,{} vs {},label_id={}' \
                    .format(self.label_dict[label_id], self.out_features, label_id)

            if self.label_dict[label_id] not in used_ind_set:
                mask = label == label_id
                # label_new[mask] = self.label_dict[label_id]
                label_new.masked_fill_(mask, self.label_dict[label_id])
                # self.weight.data[self.label_dict[label_id]] = torch.sum(x[mask].detach(), dim=0)
                # candidate_w = self.ensemble_feature(x[mask].detach(),label[mask])
                # self.weight.data[self.label_dict[label_id]] = candidate_w
                self.weight.data[self.label_dict[label_id]] = self.ensemble_feature(x[mask].detach(), label[mask])
                used_ind_set.add(self.label_dict[label_id])
            else:
                filter_label_set.add(label_id)

        # print('mapping time = ', time.time()-start_time)
        # start_time = time.time()
        candidate_set = set([i for i in range(self.out_features)]) - used_ind_set
        candidate_set = list(candidate_set)
        sta = 0
        for label_id in filter_label_set:
            assert sta < len(candidate_set)
            mask = label == label_id
            label_new.masked_fill_(mask, candidate_set[sta])
            # label_new[mask] = candidate_set[sta]
            self.weight_test[candidate_set[sta]] = self.weight[candidate_set[sta]].clone()
            # self.weight.data[candidate_set[sta]] = torch.sum(x[mask].detach(), dim=0)
            self.weight.data[candidate_set[sta]] = self.ensemble_feature(x[mask].detach(), label[mask])
            self.need_back_label_list.append(candidate_set[sta])
            sta += 1

        # print('candidate time = ', time.time()-start_time)
        # start_time = time.time()
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        sine = 1.0 - torch.pow(cosine, 2)
        sine = torch.where(sine > 0, sine, torch.zeros(sine.size(), device='cuda'))
        sine = torch.sqrt(sine)
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label_new.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        # print('end time = ', time.time()-start_time)
        # start_time = time.time()
        return output, label_new

    def forward_test(self, x, label, label_set):
        for label_id in self.need_back_label_list:
            self.weight.data[label_id] = self.weight_test[label_id].clone()
        self.need_back_label_list.clear()

        with torch.no_grad():
            self.weight_test.data = self.weight.clone()
        label_new = label.clone()
        used_ind_set = set()
        filter_label_set = set()
        for ind, label_id in enumerate(label_set):
            assert label_id in self.label_dict.keys(), label_id
            assert self.label_dict[label_id] < self.out_features, \
                'self.label_dict[label_id] < self.out_features,{} vs {},label_id={}' \
                    .format(self.label_dict[label_id], self.out_features, label_id)

            if self.label_dict[label_id] not in used_ind_set:
                mask = label == label_id
                label_new[mask] = self.label_dict[label_id]
                # self.weight_test[self.label_dict[label_id]] = torch.sum(x[mask].detach(), dim=0)
                self.weight_test[self.label_dict[label_id]] = self.ensemble_feature(x[mask].detach(), label[mask])
                used_ind_set.add(self.label_dict[label_id])
            else:
                filter_label_set.add(label_id)

        candidate_set = set([i for i in range(self.out_features)]) - used_ind_set
        candidate_set = list(candidate_set)
        sta = 0
        for label_id in filter_label_set:
            assert sta < len(candidate_set)
            mask = label == label_id
            label_new[mask] = candidate_set[sta]
            self.weight_test[candidate_set[sta]] = torch.sum(x[mask].detach(), dim=0)
            self.weight_test[candidate_set[sta]] = self.ensemble_feature(x[mask].detach(), label[mask])
            self.need_back_label_list.append(candidate_set[sta])
            sta += 1

        cosine = F.linear(F.normalize(x), F.normalize(self.weight_test))
        sine = 1.0 - torch.pow(cosine, 2)
        sine = torch.where(sine > 0, sine, torch.zeros(sine.size(), device='cuda'))
        sine = torch.sqrt(sine)
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label_new.view(-1, 1).long(), 1)
        output0 = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output0 *= self.s

        output1 = cosine
        output1 *= self.s
        return (output0, output1), label_new