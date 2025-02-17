import re
import matplotlib.pyplot as plt

output_dvs="""SEWResNet(
  (conv1): Conv2d(3.136 k, 0.028% Params, 0.0 Ops, 0.000% ACs, 5.7803 G Ops, 98.237% MACs, 100.000% Spike Rate)
  (bn1): BatchNorm2d(128, 0.001% Params, 0.0 Ops, 0.000% ACs, 33.1776 M Ops, 0.564% MACs, 100.000% Spike Rate)
  (sn1): ParametricLIFNode(
    1, 0.000% Params, 16.5888 M Ops, 0.025% ACs, 0.0 Ops, 0.000% MACs, 4.823% Spike Rate
    (surrogate_function): Sigmoid(0, 0.000% Params, 836.0163 K Ops, 0.001% ACs, 0.0 Ops, 0.000% MACs, 143.630% Spike Rate)
  )
  (maxpool): MaxPool2d(0, 0.000% Params, 823.8823 K Ops, 0.001% ACs, 0.0 Ops, 0.000% MACs, 5.035% Spike Rate)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(36.864 k, 0.330% Params, 2.8342 G Ops, 4.242% ACs, 0.0 Ops, 0.000% MACs, 16.862% Spike Rate)
      (bn1): BatchNorm2d(128, 0.001% Params, 0.0 Ops, 0.000% ACs, 8.3313 M Ops, 0.142% MACs, 100.000% Spike Rate)
      (sn1): ParametricLIFNode(
        1, 0.000% Params, 4.1656 M Ops, 0.006% ACs, 0.0 Ops, 0.000% MACs, 4.044% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 218.666 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 149.605% Spike Rate)
      )
      (conv2): Conv2d(36.864 k, 0.330% Params, 879.34 M Ops, 1.316% ACs, 0.0 Ops, 0.000% MACs, 5.242% Spike Rate)
      (bn2): BatchNorm2d(128, 0.001% Params, 0.0 Ops, 0.000% ACs, 8.3313 M Ops, 0.142% MACs, 100.000% Spike Rate)
      (sn2): ParametricLIFNode(
        1, 0.000% Params, 4.1656 M Ops, 0.006% ACs, 0.0 Ops, 0.000% MACs, 4.700% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 252.391 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 172.678% Spike Rate)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(36.864 k, 0.330% Params, 2.3461 G Ops, 3.511% ACs, 0.0 Ops, 0.000% MACs, 13.953% Spike Rate)
      (bn1): BatchNorm2d(128, 0.001% Params, 0.0 Ops, 0.000% ACs, 8.3313 M Ops, 0.142% MACs, 100.000% Spike Rate)
      (sn1): ParametricLIFNode(
        1, 0.000% Params, 4.1656 M Ops, 0.006% ACs, 0.0 Ops, 0.000% MACs, 3.178% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 170.7632 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 116.831% Spike Rate)
      )
      (conv2): Conv2d(36.864 k, 0.330% Params, 688.2176 M Ops, 1.030% ACs, 0.0 Ops, 0.000% MACs, 4.112% Spike Rate)
      (bn2): BatchNorm2d(128, 0.001% Params, 0.0 Ops, 0.000% ACs, 8.3313 M Ops, 0.142% MACs, 100.000% Spike Rate)
      (sn2): ParametricLIFNode(
        1, 0.000% Params, 4.1656 M Ops, 0.006% ACs, 0.0 Ops, 0.000% MACs, 3.395% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 191.0102 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 130.683% Spike Rate)
      )
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(73.728 k, 0.660% Params, 2.1236 G Ops, 3.178% ACs, 0.0 Ops, 0.000% MACs, 12.507% Spike Rate)
      (bn1): BatchNorm2d(256, 0.002% Params, 0.0 Ops, 0.000% ACs, 4.2025 M Ops, 0.071% MACs, 100.000% Spike Rate)
      (sn1): ParametricLIFNode(
        1, 0.000% Params, 2.1012 M Ops, 0.003% ACs, 0.0 Ops, 0.000% MACs, 3.412% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 98.3091 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 133.340% Spike Rate)
      )
      (conv2): Conv2d(147.456 k, 1.320% Params, 1.5981 G Ops, 2.392% ACs, 0.0 Ops, 0.000% MACs, 4.685% Spike Rate)
      (bn2): BatchNorm2d(256, 0.002% Params, 0.0 Ops, 0.000% ACs, 4.2025 M Ops, 0.071% MACs, 100.000% Spike Rate)
      (sn2): ParametricLIFNode(
        1, 0.000% Params, 2.1012 M Ops, 0.003% ACs, 0.0 Ops, 0.000% MACs, 2.188% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 73.4766 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 99.659% Spike Rate)
      )
      (downsample): Sequential(
        (0): Conv2d(8.192 k, 0.073% Params, 235.9538 M Ops, 0.353% ACs, 0.0 Ops, 0.000% MACs, 12.507% Spike Rate)
        (1): BatchNorm2d(256, 0.002% Params, 0.0 Ops, 0.000% ACs, 4.2025 M Ops, 0.071% MACs, 100.000% Spike Rate)
      )
      (downsample_sn): ParametricLIFNode(
        1, 0.000% Params, 2.1012 M Ops, 0.003% ACs, 0.0 Ops, 0.000% MACs, 4.327% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 121.4735 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 164.759% Spike Rate)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(147.456 k, 1.320% Params, 1.8411 G Ops, 2.756% ACs, 0.0 Ops, 0.000% MACs, 5.414% Spike Rate)
      (bn1): BatchNorm2d(256, 0.002% Params, 0.0 Ops, 0.000% ACs, 4.2025 M Ops, 0.071% MACs, 100.000% Spike Rate)
      (sn1): ParametricLIFNode(
        1, 0.000% Params, 2.1012 M Ops, 0.003% ACs, 0.0 Ops, 0.000% MACs, 3.757% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 119.9784 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 162.731% Spike Rate)
      )
      (conv2): Conv2d(147.456 k, 1.320% Params, 1.9483 G Ops, 2.916% ACs, 0.0 Ops, 0.000% MACs, 5.755% Spike Rate)
      (bn2): BatchNorm2d(256, 0.002% Params, 0.0 Ops, 0.000% ACs, 4.2025 M Ops, 0.071% MACs, 100.000% Spike Rate)
      (sn2): ParametricLIFNode(
        1, 0.000% Params, 2.1012 M Ops, 0.003% ACs, 0.0 Ops, 0.000% MACs, 3.913% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 141.2548 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 191.589% Spike Rate)
      )
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(294.912 k, 2.640% Params, 1.5944 G Ops, 2.386% ACs, 0.0 Ops, 0.000% MACs, 4.593% Spike Rate)
      (bn1): BatchNorm2d(512, 0.005% Params, 0.0 Ops, 0.000% ACs, 2.1381 M Ops, 0.036% MACs, 100.000% Spike Rate)
      (sn1): ParametricLIFNode(
        1, 0.000% Params, 1.0691 M Ops, 0.002% ACs, 0.0 Ops, 0.000% MACs, 3.585% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 70.7612 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 188.643% Spike Rate)
      )
      (conv2): Conv2d(589.824 k, 5.280% Params, 4.5776 G Ops, 6.851% ACs, 0.0 Ops, 0.000% MACs, 6.557% Spike Rate)
      (bn2): BatchNorm2d(512, 0.005% Params, 0.0 Ops, 0.000% ACs, 2.1381 M Ops, 0.036% MACs, 100.000% Spike Rate)
      (sn2): ParametricLIFNode(
        1, 0.000% Params, 1.0691 M Ops, 0.002% ACs, 0.0 Ops, 0.000% MACs, 7.792% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 176.915 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 471.638% Spike Rate)
      )
      (downsample): Sequential(
        (0): Conv2d(32.768 k, 0.293% Params, 177.1525 M Ops, 0.265% ACs, 0.0 Ops, 0.000% MACs, 4.593% Spike Rate)
        (1): BatchNorm2d(512, 0.005% Params, 0.0 Ops, 0.000% ACs, 2.1381 M Ops, 0.036% MACs, 100.000% Spike Rate)
      )
      (downsample_sn): ParametricLIFNode(
        1, 0.000% Params, 1.0691 M Ops, 0.002% ACs, 0.0 Ops, 0.000% MACs, 6.230% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 100.1934 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 267.106% Spike Rate)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(589.824 k, 5.280% Params, 4.7191 G Ops, 7.063% ACs, 0.0 Ops, 0.000% MACs, 6.765% Spike Rate)
      (bn1): BatchNorm2d(512, 0.005% Params, 0.0 Ops, 0.000% ACs, 2.1381 M Ops, 0.036% MACs, 100.000% Spike Rate)
      (sn1): ParametricLIFNode(
        1, 0.000% Params, 1.0691 M Ops, 0.002% ACs, 0.0 Ops, 0.000% MACs, 2.651% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 37.8176 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 100.818% Spike Rate)
      )
      (conv2): Conv2d(589.824 k, 5.280% Params, 2.4638 G Ops, 3.688% ACs, 0.0 Ops, 0.000% MACs, 3.560% Spike Rate)
      (bn2): BatchNorm2d(512, 0.005% Params, 0.0 Ops, 0.000% ACs, 2.1381 M Ops, 0.036% MACs, 100.000% Spike Rate)
      (sn2): ParametricLIFNode(
        1, 0.000% Params, 1.0691 M Ops, 0.002% ACs, 0.0 Ops, 0.000% MACs, 3.813% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 57.5781 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 153.498% Spike Rate)
      )
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(1.1796 M, 10.560% Params, 4.3174 G Ops, 6.462% ACs, 0.0 Ops, 0.000% MACs, 5.972% Spike Rate)
      (bn1): BatchNorm2d(1.024 k, 0.009% Params, 0.0 Ops, 0.000% ACs, 1.1059 M Ops, 0.019% MACs, 100.000% Spike Rate)
      (sn1): ParametricLIFNode(
        1, 0.000% Params, 552.96 K Ops, 0.001% ACs, 0.0 Ops, 0.000% MACs, 4.467% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 49.1032 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 253.082% Spike Rate)
      )
      (conv2): Conv2d(2.3593 M, 21.120% Params, 12.7089 G Ops, 19.022% ACs, 0.0 Ops, 0.000% MACs, 8.773% Spike Rate)
      (bn2): BatchNorm2d(1.024 k, 0.009% Params, 0.0 Ops, 0.000% ACs, 1.1059 M Ops, 0.019% MACs, 100.000% Spike Rate)
      (sn2): ParametricLIFNode(
        1, 0.000% Params, 552.96 K Ops, 0.001% ACs, 0.0 Ops, 0.000% MACs, 4.828% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 51.5085 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 265.479% Spike Rate)
      )
      (downsample): Sequential(
        (0): Conv2d(131.072 k, 1.173% Params, 479.7104 M Ops, 0.718% ACs, 0.0 Ops, 0.000% MACs, 5.972% Spike Rate)
        (1): BatchNorm2d(1.024 k, 0.009% Params, 0.0 Ops, 0.000% ACs, 1.1059 M Ops, 0.019% MACs, 100.000% Spike Rate)
      )
      (downsample_sn): ParametricLIFNode(
        1, 0.000% Params, 552.96 K Ops, 0.001% ACs, 0.0 Ops, 0.000% MACs, 7.421% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 51.6152 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 266.029% Spike Rate)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(2.3593 M, 21.120% Params, 12.7204 G Ops, 19.039% ACs, 0.0 Ops, 0.000% MACs, 8.858% Spike Rate)
      (bn1): BatchNorm2d(1.024 k, 0.009% Params, 0.0 Ops, 0.000% ACs, 1.1059 M Ops, 0.019% MACs, 100.000% Spike Rate)
      (sn1): ParametricLIFNode(
        1, 0.000% Params, 552.96 K Ops, 0.001% ACs, 0.0 Ops, 0.000% MACs, 3.528% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 31.1873 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 160.742% Spike Rate)
      )
      (conv2): Conv2d(2.3593 M, 21.120% Params, 8.5073 G Ops, 12.733% ACs, 0.0 Ops, 0.000% MACs, 5.903% Spike Rate)
      (bn2): BatchNorm2d(1.024 k, 0.009% Params, 0.0 Ops, 0.000% ACs, 1.1059 M Ops, 0.019% MACs, 100.000% Spike Rate)
      (sn2): ParametricLIFNode(
        1, 0.000% Params, 552.96 K Ops, 0.001% ACs, 0.0 Ops, 0.000% MACs, 3.868% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 30.3736 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 156.548% Spike Rate)
      )
    )
  )
  (avgpool): AdaptiveAvgPool2d(0, 0.000% Params, 46.8223 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 8.549% Spike Rate)
  (fc): Linear(513, 0.005% Params, 0.0 Ops, 0.000% ACs, 4.608 K Ops, 0.000% MACs, 100.000% Spike Rate)
)
"""

output_rgb = """SEWResNet(
  (conv1): Conv2d(9.408 k, 0.084% Params, 0.0 Ops, 0.000% ACs, 17.3408 G Ops, 99.405% MACs, 100.000% Spike Rate)
  (bn1): BatchNorm2d(128, 0.001% Params, 0.0 Ops, 0.000% ACs, 33.1776 M Ops, 0.190% MACs, 100.000% Spike Rate)
  (sn1): ParametricLIFNode(
    1, 0.000% Params, 16.5888 M Ops, 0.029% ACs, 0.0 Ops, 0.000% MACs, 2.527% Spike Rate
    (surrogate_function): Sigmoid(0, 0.000% Params, 1.5402 M Ops, 0.003% ACs, 0.0 Ops, 0.000% MACs, 264.616% Spike Rate)
  )
  (maxpool): MaxPool2d(0, 0.000% Params, 1.5431 M Ops, 0.003% ACs, 0.0 Ops, 0.000% MACs, 9.653% Spike Rate)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(36.864 k, 0.330% Params, 2.739 G Ops, 4.709% ACs, 0.0 Ops, 0.000% MACs, 16.410% Spike Rate)
      (bn1): BatchNorm2d(128, 0.001% Params, 0.0 Ops, 0.000% ACs, 8.3313 M Ops, 0.048% MACs, 100.000% Spike Rate)
      (sn1): ParametricLIFNode(
        1, 0.000% Params, 4.1656 M Ops, 0.007% ACs, 0.0 Ops, 0.000% MACs, 0.734% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 333.2657 K Ops, 0.001% ACs, 0.0 Ops, 0.000% MACs, 228.010% Spike Rate)
      )
      (conv2): Conv2d(36.864 k, 0.330% Params, 1.3731 G Ops, 2.361% ACs, 0.0 Ops, 0.000% MACs, 8.275% Spike Rate)
      (bn2): BatchNorm2d(128, 0.001% Params, 0.0 Ops, 0.000% ACs, 8.3313 M Ops, 0.048% MACs, 100.000% Spike Rate)
      (sn2): ParametricLIFNode(
        1, 0.000% Params, 4.1656 M Ops, 0.007% ACs, 0.0 Ops, 0.000% MACs, 0.000% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 265.0471 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 181.337% Spike Rate)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(36.864 k, 0.330% Params, 1.9571 G Ops, 3.365% ACs, 0.0 Ops, 0.000% MACs, 11.612% Spike Rate)
      (bn1): BatchNorm2d(128, 0.001% Params, 0.0 Ops, 0.000% ACs, 8.3313 M Ops, 0.048% MACs, 100.000% Spike Rate)
      (sn1): ParametricLIFNode(
        1, 0.000% Params, 4.1656 M Ops, 0.007% ACs, 0.0 Ops, 0.000% MACs, 1.262% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 366.1172 K Ops, 0.001% ACs, 0.0 Ops, 0.000% MACs, 250.486% Spike Rate)
      )
      (conv2): Conv2d(36.864 k, 0.330% Params, 1.5096 G Ops, 2.595% ACs, 0.0 Ops, 0.000% MACs, 9.080% Spike Rate)
      (bn2): BatchNorm2d(128, 0.001% Params, 0.0 Ops, 0.000% ACs, 8.3313 M Ops, 0.048% MACs, 100.000% Spike Rate)
      (sn2): ParametricLIFNode(
        1, 0.000% Params, 4.1656 M Ops, 0.007% ACs, 0.0 Ops, 0.000% MACs, 0.160% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 334.707 K Ops, 0.001% ACs, 0.0 Ops, 0.000% MACs, 228.996% Spike Rate)
      )
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(73.728 k, 0.660% Params, 1.4467 G Ops, 2.487% ACs, 0.0 Ops, 0.000% MACs, 8.430% Spike Rate)
      (bn1): BatchNorm2d(256, 0.002% Params, 0.0 Ops, 0.000% ACs, 4.2025 M Ops, 0.024% MACs, 100.000% Spike Rate)
      (sn1): ParametricLIFNode(
        1, 0.000% Params, 2.1012 M Ops, 0.004% ACs, 0.0 Ops, 0.000% MACs, 2.022% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 147.0855 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 199.497% Spike Rate)
      )
      (conv2): Conv2d(147.456 k, 1.319% Params, 2.4126 G Ops, 4.148% ACs, 0.0 Ops, 0.000% MACs, 6.992% Spike Rate)
      (bn2): BatchNorm2d(256, 0.002% Params, 0.0 Ops, 0.000% ACs, 4.2025 M Ops, 0.024% MACs, 100.000% Spike Rate)
      (sn2): ParametricLIFNode(
        1, 0.000% Params, 2.1012 M Ops, 0.004% ACs, 0.0 Ops, 0.000% MACs, 1.408% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 213.8357 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 290.033% Spike Rate)
      )
      (downsample): Sequential(
        (0): Conv2d(8.192 k, 0.073% Params, 160.7487 M Ops, 0.276% ACs, 0.0 Ops, 0.000% MACs, 8.430% Spike Rate)
        (1): BatchNorm2d(256, 0.002% Params, 0.0 Ops, 0.000% ACs, 4.2025 M Ops, 0.024% MACs, 100.000% Spike Rate)
      )
      (downsample_sn): ParametricLIFNode(
        1, 0.000% Params, 2.1012 M Ops, 0.004% ACs, 0.0 Ops, 0.000% MACs, 3.504% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 159.0601 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 215.739% Spike Rate)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(147.456 k, 1.319% Params, 1.9502 G Ops, 3.353% ACs, 0.0 Ops, 0.000% MACs, 5.614% Spike Rate)
      (bn1): BatchNorm2d(256, 0.002% Params, 0.0 Ops, 0.000% ACs, 4.2025 M Ops, 0.024% MACs, 100.000% Spike Rate)
      (sn1): ParametricLIFNode(
        1, 0.000% Params, 2.1012 M Ops, 0.004% ACs, 0.0 Ops, 0.000% MACs, 3.137% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 113.1306 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 153.443% Spike Rate)
      )
      (conv2): Conv2d(147.456 k, 1.319% Params, 1.8569 G Ops, 3.193% ACs, 0.0 Ops, 0.000% MACs, 5.347% Spike Rate)
      (bn2): BatchNorm2d(256, 0.002% Params, 0.0 Ops, 0.000% ACs, 4.2025 M Ops, 0.024% MACs, 100.000% Spike Rate)
      (sn2): ParametricLIFNode(
        1, 0.000% Params, 2.1012 M Ops, 0.004% ACs, 0.0 Ops, 0.000% MACs, 5.105% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 159.5325 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 216.380% Spike Rate)
      )
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(294.912 k, 2.639% Params, 1.5133 G Ops, 2.602% ACs, 0.0 Ops, 0.000% MACs, 4.272% Spike Rate)
      (bn1): BatchNorm2d(512, 0.005% Params, 0.0 Ops, 0.000% ACs, 2.1381 M Ops, 0.012% MACs, 100.000% Spike Rate)
      (sn1): ParametricLIFNode(
        1, 0.000% Params, 1.0691 M Ops, 0.002% ACs, 0.0 Ops, 0.000% MACs, 1.569% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 40.7214 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 108.559% Spike Rate)
      )
      (conv2): Conv2d(589.824 k, 5.277% Params, 2.6682 G Ops, 4.587% ACs, 0.0 Ops, 0.000% MACs, 3.720% Spike Rate)
      (bn2): BatchNorm2d(512, 0.005% Params, 0.0 Ops, 0.000% ACs, 2.1381 M Ops, 0.012% MACs, 100.000% Spike Rate)
      (sn2): ParametricLIFNode(
        1, 0.000% Params, 1.0691 M Ops, 0.002% ACs, 0.0 Ops, 0.000% MACs, 4.328% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 115.725 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 308.512% Spike Rate)
      )
      (downsample): Sequential(
        (0): Conv2d(32.768 k, 0.293% Params, 168.1405 M Ops, 0.289% ACs, 0.0 Ops, 0.000% MACs, 4.272% Spike Rate)
        (1): BatchNorm2d(512, 0.005% Params, 0.0 Ops, 0.000% ACs, 2.1381 M Ops, 0.012% MACs, 100.000% Spike Rate)
      )
      (downsample_sn): ParametricLIFNode(
        1, 0.000% Params, 1.0691 M Ops, 0.002% ACs, 0.0 Ops, 0.000% MACs, 3.933% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 83.5835 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 222.825% Spike Rate)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(589.824 k, 5.277% Params, 4.1531 G Ops, 7.140% ACs, 0.0 Ops, 0.000% MACs, 5.889% Spike Rate)
      (bn1): BatchNorm2d(512, 0.005% Params, 0.0 Ops, 0.000% ACs, 2.1381 M Ops, 0.012% MACs, 100.000% Spike Rate)
      (sn1): ParametricLIFNode(
        1, 0.000% Params, 1.0691 M Ops, 0.002% ACs, 0.0 Ops, 0.000% MACs, 0.981% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 35.8723 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 95.632% Spike Rate)
      )
      (conv2): Conv2d(589.824 k, 5.277% Params, 2.3473 G Ops, 4.036% ACs, 0.0 Ops, 0.000% MACs, 3.342% Spike Rate)
      (bn2): BatchNorm2d(512, 0.005% Params, 0.0 Ops, 0.000% ACs, 2.1381 M Ops, 0.012% MACs, 100.000% Spike Rate)
      (sn2): ParametricLIFNode(
        1, 0.000% Params, 1.0691 M Ops, 0.002% ACs, 0.0 Ops, 0.000% MACs, 2.485% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 107.3449 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 286.171% Spike Rate)
      )
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(1.1796 M, 10.554% Params, 3.0604 G Ops, 5.262% ACs, 0.0 Ops, 0.000% MACs, 4.190% Spike Rate)
      (bn1): BatchNorm2d(1.024 k, 0.009% Params, 0.0 Ops, 0.000% ACs, 1.1059 M Ops, 0.006% MACs, 100.000% Spike Rate)
      (sn1): ParametricLIFNode(
        1, 0.000% Params, 552.96 K Ops, 0.001% ACs, 0.0 Ops, 0.000% MACs, 0.712% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 9.5141 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 49.036% Spike Rate)
      )
      (conv2): Conv2d(2.3593 M, 21.108% Params, 2.5213 G Ops, 4.335% ACs, 0.0 Ops, 0.000% MACs, 1.717% Spike Rate)
      (bn2): BatchNorm2d(1.024 k, 0.009% Params, 0.0 Ops, 0.000% ACs, 1.1059 M Ops, 0.006% MACs, 100.000% Spike Rate)
      (sn2): ParametricLIFNode(
        1, 0.000% Params, 552.96 K Ops, 0.001% ACs, 0.0 Ops, 0.000% MACs, 2.555% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 38.7531 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 199.736% Spike Rate)
      )
      (downsample): Sequential(
        (0): Conv2d(131.072 k, 1.173% Params, 340.0475 M Ops, 0.585% ACs, 0.0 Ops, 0.000% MACs, 4.190% Spike Rate)
        (1): BatchNorm2d(1.024 k, 0.009% Params, 0.0 Ops, 0.000% ACs, 1.1059 M Ops, 0.006% MACs, 100.000% Spike Rate)
      )
      (downsample_sn): ParametricLIFNode(
        1, 0.000% Params, 552.96 K Ops, 0.001% ACs, 0.0 Ops, 0.000% MACs, 10.613% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 96.3133 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 496.406% Spike Rate)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(2.3593 M, 21.108% Params, 22.0872 G Ops, 37.974% ACs, 0.0 Ops, 0.000% MACs, 15.429% Spike Rate)
      (bn1): BatchNorm2d(1.024 k, 0.009% Params, 0.0 Ops, 0.000% ACs, 1.1059 M Ops, 0.006% MACs, 100.000% Spike Rate)
      (sn1): ParametricLIFNode(
        1, 0.000% Params, 552.96 K Ops, 0.001% ACs, 0.0 Ops, 0.000% MACs, 1.162% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 14.329 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 73.853% Spike Rate)
      )
      (conv2): Conv2d(2.3593 M, 21.108% Params, 3.8459 G Ops, 6.612% ACs, 0.0 Ops, 0.000% MACs, 2.726% Spike Rate)
      (bn2): BatchNorm2d(1.024 k, 0.009% Params, 0.0 Ops, 0.000% ACs, 1.1059 M Ops, 0.006% MACs, 100.000% Spike Rate)
      (sn2): ParametricLIFNode(
        1, 0.000% Params, 552.96 K Ops, 0.001% ACs, 0.0 Ops, 0.000% MACs, 4.245% Spike Rate
        (surrogate_function): Sigmoid(0, 0.000% Params, 51.1424 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 263.592% Spike Rate)
      )
    )
  )
  (avgpool): AdaptiveAvgPool2d(0, 0.000% Params, 63.7943 K Ops, 0.000% ACs, 0.0 Ops, 0.000% MACs, 11.639% Spike Rate)
  (fc): Linear(513, 0.005% Params, 0.0 Ops, 0.000% ACs, 4.608 K Ops, 0.000% MACs, 100.000% Spike Rate)
)
"""
import re

# def convert_output_to_dict(output):
#     model_dict = {}
#     lines = output.strip().split('\n')
#     current_layer = None

#     for line in lines:
#         line = line.strip()
#         if line.startswith('('):  # Indicates a layer
#             if current_layer:
#                 model_dict[current_layer] = model_dict.get(current_layer, {})
#             current_layer = line.split(':')[0].strip('() ')
#             model_dict[current_layer] = {}
#         elif ':' in line:  # Indicates a parameter
#             key, value = line.split(':', 1)
#             model_dict[current_layer][key.strip()] = value.strip()

#     return model_dict



def preprocess_spike_rates(input_str):
    # Initialize an empty dictionary to store layers and their spike rates
    layers_dict = {}

    # Use a regular expression to capture layer names and spike rates
    pattern = r'(\w+[\w\d]*): \[(.*?)\]'  # This will capture the layer name and spike rates in the brackets
    
    matches = re.findall(pattern, input_str)
    
    for match in matches:
        layer_name = match[0]
        spike_rates = match[1].split(', ')  # Split the spike rates by commas
        
        # Convert spike rates to numeric values (remove '%' and convert to float)
        spike_rates = [float(rate.replace('%', '')) for rate in spike_rates]
        
        # Add to dictionary with layer_name as the key and the list of spike rates as the value
        layers_dict[layer_name] = spike_rates
    
    return layers_dict

# Call the function and store the result in a variable
layer_spike_rates = preprocess_spike_rates(output_dvs)

# Print the resulting dictionary
print(layer_spike_rates)

import re




def parse_model(output, filter_parametric=False):
    # Initialize empty lists to store layer names and their spike rates
    layer_spike_rates = []
    conv2d_spike_rates = []  # New list for Conv2d layers
    # Regex pattern to extract layer details (including ParametricLIFNode)
    pattern = r"\((\w+)\):\s(\w+\(?\w*\)?(?:[\w\s.,%]+)?)"

    # Find all matches
    matches = re.findall(pattern, output)

    # Loop through matches and print based on filter flag
    for match in matches:
        layer_name = match[0]
        operation = match[1]
        
        if filter_parametric and 'ParametricLIFNode' in operation:
            # Print only ParametricLIFNode layers if the flag is set
            print(f"Layer: {layer_name}")
            print(f"Operation: {operation}")
            print("="*40)

            # Regular expression to find the Spike Rate value
            pattern = r'(\d+\.\d+)% Spike Rate'

            # Search for the Spike Rate value in the string
            match = re.search(pattern, operation)

            if match:
                spike_rate = float(match.group(1))  # Convert the matched value to float
                print(f"Spike Rate: {spike_rate}%")
                layer_spike_rates.append([layer_name, spike_rate])
            else:
                print("Spike Rate not found.")
        
        # New condition to handle Conv2d layers
        if 'Conv2d' in operation:
            print(f"Layer: {layer_name}")
            print(f"Operation: {operation}")
            print("="*40)

            # Regular expression to find the Spike Rate value
            pattern = r'(\d+\.\d+)% Spike Rate'

            # Search for the Spike Rate value in the string
            match = re.search(pattern, operation)

            if match:
                spike_rate = float(match.group(1))  # Convert the matched value to float
                print(f"Spike Rate: {spike_rate}%")
                conv2d_spike_rates.append([layer_name, spike_rate])
            else:
                print("Spike Rate not found.")

    return layer_spike_rates, conv2d_spike_rates  # Return both lists



        





def plot_spike_rates(layer_spike_rates, conv2d_spike_rates):
    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))  # Two subfigures in one column

    # Plot for ParametricLIFNode layers
    layers = [item[0] for item in layer_spike_rates]
    spike_rates = [item[1] for item in layer_spike_rates]
    # Check for duplicate layer names and add suffix if necessary
    layer_counts = {}
    for i in range(len(layers)):
        if layers[i] in layer_counts:
            layer_counts[layers[i]] += 1
            layers[i] = f"{layers[i]}_{layer_counts[layers[i]]}"
        else:
            layer_counts[layers[i]] = 0
    axs[0].bar(layers, spike_rates, color='skyblue')
    axs[0].set_ylabel('Average Spike Rate Conv2d (%)')
    axs[0].set_title('Average Spike Rate per ParametricLIFNode Layer')
    axs[0].set_ylim(0, 11)  # Set y axis max value to 11
    axs[0].tick_params(axis='x', rotation=45)  # Rotate x-axis labels by 45 degrees

    # Plot for Conv2d layers
    conv_layers = [item[0] for item in conv2d_spike_rates]
    conv_spike_rates = [item[1] for item in conv2d_spike_rates]
    # Check for duplicate layer names and add suffix if necessary
    conv_layer_counts = {}
    for i in range(len(conv_layers)):
        if conv_layers[i] in conv_layer_counts:
            conv_layer_counts[conv_layers[i]] += 1
            conv_layers[i] = f"{conv_layers[i]}_{conv_layer_counts[conv_layers[i]]}"
        else:
            conv_layer_counts[conv_layers[i]] = 0
    axs[1].bar(conv_layers, conv_spike_rates, color='lightgreen')
    axs[1].set_ylabel('Average Spike Rate (%)')
    axs[1].set_title('Average Spike Rate per Conv2d Layer')
    axs[1].set_ylim(0, 20)  # Set y axis max value to 11
    axs[1].tick_params(axis='x', rotation=45)  # Rotate x-axis labels by 45 degrees

    plt.tight_layout()  # Ensure labels fit within the figure
    plt.savefig('figures/spike_rates_combined_dvs.png')  # Save the figure to a file
    plt.show()


# Call function with filter_parametric=True to print only ParametricLIFNode layers
layer_spike_rates, conv2d_spike_rates = parse_model(output_dvs, filter_parametric=True)
print(layer_spike_rates)
# Call the function to plot
plot_spike_rates(layer_spike_rates, conv2d_spike_rates)