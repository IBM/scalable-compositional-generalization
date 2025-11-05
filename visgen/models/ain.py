import torch 
from torch import nn

from .resnet import ResNet, BasicBlock


class SplitResNet18(ResNet):
    def __init__(
        self,
        split_layers=1,
        exit_reg=10,
        **kwargs,
    ):
        super().__init__(
            block=BasicBlock,
            layers=[2, 2, 2, 2],
            **kwargs,
        )
        assert split_layers >= 0 and split_layers <= 4, "split layers must be between 0 and 4"

        if self.replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        self.split_layers = split_layers
        self.exit_reg = exit_reg
        self.layer_planes = [2**i for i in range(6, 10)]
        self.exit_head_emb_size = [2**i for i in range(14, 9, -1)]

        split_blocks = []
        for _ in self.attribute_sizes:
            if self.maxpool == 0:
                conv1 = nn.Conv2d(
                    self.in_channels,
                    self.inplanes,
                    kernel_size=7,
                    stride=1,
                    padding=3,
                    bias=False,
                )
                maxpool = nn.Identity()
            else:
                conv1 = nn.Conv2d(
                    self.in_channels,
                    self.inplanes,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                )
                maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            
            bn1 = self._norm_layer(self.inplanes)

            split_block = [
                conv1, bn1, self.activation, maxpool
            ]
            for i in range(self.split_layers):
                if i == 0:
                    dilate = False
                    stride = 1
                else:
                    dilate = replace_stride_with_dilation[i-1]
                    stride = 2
                
                li = self._make_layer(
                    self.block, 
                    self.layer_planes[i], 
                    self.layers[i], 
                    self.activation, 
                    stride=stride, 
                    dilate=dilate, 
                    skip_init=self.skip_init
                )
                split_block.append(li)

            if self.split_layers == 4:
                split_block.append(nn.AdaptiveAvgPool2d((1, 1)))
            
            split_blocks.append(nn.Sequential(*split_block))
            
        self.split_block = nn.ModuleList(split_blocks)
        self.exit_head = nn.Linear(self.exit_head_emb_size[self.split_layers], sum(self.attribute_sizes))
        self.exit_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        shared_blocks = []
        for i in range(self.split_layers, 4):
            print(i, self.layer_planes[i], self.layers[i])
            if i == 0:
                dilate = False
                stride = 1
            else:
                dilate = replace_stride_with_dilation[i-1]
                stride = 2
            
            shared_block = self._make_layer(
                self.block, 
                self.layer_planes[i], 
                self.layers[i], 
                self.activation, 
                stride=stride, 
                dilate=dilate, 
                skip_init=self.skip_init
            )
            shared_blocks.append(shared_block)
        
        if len(shared_blocks) > 0:
            shared_blocks.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.shared_blocks = nn.Sequential(*shared_blocks)

        self.conv1 = None
        self.bn1 = None
        self.maxpool = None
        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None
        self.avgpool = None

    def forward(self, x, mode='test'):
        if self.preprocessing is not None:
            with torch.no_grad():
                x = self.preprocessing(x)

        h = []
        for split_block in self.split_block:
            h.append(split_block(x))
        
        h = torch.cat(h, axis=0)
        x = self.shared_blocks(h)
        x = torch.flatten(x, 1)
        
        # early exit embeddings
        h = self.exit_avgpool(h)
        h = torch.flatten(h, 1)

        if self.objective == "classification":
            # split output into separate list per attribute
            x_split = torch.split(x, x.shape[0]//len(self.attribute_sizes), dim=0)
            h_split = torch.split(h, h.shape[0]//len(self.attribute_sizes), dim=0)
            logits_x, logits_h = [], []
            j = 0
            for i, n in enumerate(self.attribute_sizes):
                logits_xi = x_split[i][:, j : j + n] 
                logits_hi = h_split[i][:, j : j + n]
                logits_x.append(logits_xi)
                logits_h.append(logits_hi)
                j += n

        if mode=='train':
            return logits_x, logits_h
        
        return logits_x

    def train_step(self, x, y, optimizer, amp_scaler=None, **kwargs):
        # train step
        optimizer.zero_grad()
        if amp_scaler:
            with torch.cuda.amp.autocast():
                yp, yhp = self(x, 'train')
                # main loss
                yloss, attr_loss = self.loss_fn(yp, y)
                hloss, _ = self.loss_fn(yhp, y)
                loss = yloss + self.exit_reg * hloss
                amp_scaler.scale(loss).backward()
                total_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.parameters(), max_norm=1e3
                )
            # update the model
            if total_grad_norm.isfinite:
                amp_scaler.step(optimizer)
                amp_scaler.update()
        else:
            yp = self(x)
            # main loss
            loss, attr_loss = self.loss(yp, y)
            loss.backward()
            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), max_norm=1e3
            )
            optimizer.step()
        # compute metrics
        metrics, attr_metrics = self._compute_metrics(yp, y)
        # compose log dictionary
        dlog = self._compose_logging_dict(loss, attr_loss, metrics, attr_metrics)
        return dlog
