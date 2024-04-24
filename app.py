#! /usr/bin/env python3
## ---------------------------------------------------------------------------
##
## File: app.py for HoloInfer
##
## Created by Zhijin Li
## E-mail:   <zhijinl@nvidia.com>
##
## Started on  Wed Apr 24 13:22:28 2024 Zhijin Li
## Last update Wed Apr 24 15:48:33 2024 Zhijin Li
## ---------------------------------------------------------------------------


import os

from holoscan.operators import InferenceOp
from holoscan.resources import UnboundedAllocator
from holoscan.core import Application, Operator, OperatorSpec

try:
  import cupy as cp
except ImportError:
    raise ImportError(
      'CuPy must be installed to run this example. See '
      'https://docs.cupy.dev/en/stable/install.html'
    )


class InputOp(Operator):

  def __init__(self, fragment, *args, **kwargs):
    super().__init__(fragment, *args, **kwargs)

  def setup(self, spec: OperatorSpec):
    spec.output('out')

  def compute(self, op_input, op_output, context):

    # inp = cp.random.randn(1, 3, 640, 640, dtype=cp.float32)
    inp = cp.array([0], dtype=cp.float32)

    op_output.emit(
      {
        'inp': inp
      },
      'out'
    )


class PrintOp(Operator):

  def __init__(self, fragment, *args, **kwargs):
    super().__init__(fragment, *args, **kwargs)

  def setup(self, spec: OperatorSpec):
    spec.input('in')

  def compute(self, op_input, op_output, context):

    in_message = op_input.receive('in')
    for key, value in in_message.items():
      print('key:', key)
      print('value:', cp.asarray(value, dtype=cp.float32))


class App(Application):

  def compose(self):

    input_op = InputOp(self, name='input')

    inference_op = InferenceOp(
      self,
      name='inference',
      allocator=UnboundedAllocator(self, name='pool'),
      **self.kwargs('inference'),
    )

    print_op = PrintOp(
      self,
      name='print'
    )

    self.add_flow(input_op, inference_op, {('out', 'receivers')})
    self.add_flow(inference_op, print_op, {('transmitter', 'in')})


if __name__ == '__main__':

  config_file = os.path.join(os.path.dirname(__file__), 'app.yaml')

  app = App()
  app.config(config_file)
  app.run()
