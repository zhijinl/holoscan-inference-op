#! /usr/bin/env python3
## ---------------------------------------------------------------------------
##
## File: model.py for HoloInfer
##
## Created by Zhijin Li
## E-mail:   <zhijinl@nvidia.com>
##
## Started on  Wed Apr 24 12:04:15 2024 Zhijin Li
## Last update Wed Apr 24 13:18:42 2024 Zhijin Li
## ---------------------------------------------------------------------------


import torch


class Model(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.weights = torch.randn(size=(640, 4))

  def forward(self, inp):

    # Assume input shape (1, 3, 640, 640)
    out = torch.matmul(inp, self.weights)
    out = torch.mean(out, dim=2)
    out = torch.reshape(out, shape=(-1, 4))
    return out


if __name__ == '__main__':

  torch.onnx.export(
    Model(),
    torch.ones((1, 3, 640, 640)),
    'dummy.onnx',
    input_names=['inp'],
    output_names=['out']
  )
