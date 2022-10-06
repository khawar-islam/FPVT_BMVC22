import torch
from .arcface import ArcFaceHeader


class MagFaceHeader(ArcFaceHeader):
    """ MagFaceHeader class"""

    def __init__(self, in_features, out_features, s=64.0, l_a=10, u_a=110, l_m=0.45, u_m=0.8, lambda_g=20):
        super(MagFaceHeader, self).__init__(in_features=in_features, out_features=out_features, s=s, m=None)

        self.l_a = l_a
        self.u_a = u_a
        self.l_m = l_m
        self.u_m = u_m

        self.lambda_g = lambda_g

    def compute_m(self, a):
        return (self.u_m - self.l_m) / (self.u_a - self.l_a) * (a - self.l_a) + self.l_m

    def compute_g(self, a):
        return torch.mean((1 / self.u_a ** 2) * a + 1 / a)

    def forward(self, input, label):
        # multiply normed features (input) and normed weights to obtain cosine of theta (logits)
        self.linear.weight = torch.nn.Parameter(self.normalize(self.linear.weight))
        logits = self.linear(self.normalize(input)).clamp(-1 + self.epsilon, 1 - self.epsilon)

        # difference compared to arcface
        a = torch.norm(input, dim=1, keepdim=True).clamp(self.l_a, self.u_a)
        m = self.compute_m(a)
        g = self.compute_g(a)

        # apply arccos to get theta
        theta = torch.acos(logits).clamp(-1, 1)

        # add angular margin (m) to theta and transform back by cos
        target_logits = torch.cos(theta + m)

        # derive one-hot encoding for label
        one_hot = torch.zeros(logits.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)

        # build the output logits
        output = one_hot * target_logits + (1.0 - one_hot) * logits
        # feature re-scaling
        output *= self.s

        return output + self.lambda_g * g
