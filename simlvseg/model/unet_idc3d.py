import os

# 禁用 Albumentations 的自动更新检查
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
# from monai.networks.layers.simplelayers import SkipConnection
# from monai.utils import alias, deprecated_arg, export

from simlvseg.model.utils.kan import KANBlock
from simlvseg.model.utils.idc3d import InceptionDWConv3dBlock
from simlvseg.model.utils.carafe3d import CARAFE3D, CARAFE3DBlock
from simlvseg.model.utils.attention import GlobalAttBlock, AxialAttBlock
# from simlvseg.model.utils.se3d import SE3dBlock


class UNetIDC3D(nn.Module):
    def __init__(self):
        super().__init__()

        # self.encoder1 = self._create_encoder_block(3, [16, 16])
        # self.encoder2 = self._create_encoder_block(16, [32, 32, 32],)
        # self.encoder3 = self._create_encoder_block(32, [64, 64, 64, 64],)
        # self.encoder4 = self._create_encoder_block(64, [128, 128, 128, 128, 128, 128],)
        # self.encoder5 = self._create_encoder_block(128, [256, 256, 256],)

        self.encoder1 = self._create_encoder_block(3, [16, 16])
        self.encoder2 = self._create_encoder_block(16, [32, 32, 32],
                                                   cube_kernel_size=3, spatial_kernel_size=7, temporal_kernel_size=7)
        self.encoder3 = self._create_encoder_block(32, [64, 64, 64, 64],
                                                   cube_kernel_size=3, spatial_kernel_size=5, temporal_kernel_size=5)
        self.encoder4 = self._create_encoder_block(64, [128, 128, 128, 128, 128, 128],
                                                   cube_kernel_size=3, spatial_kernel_size=3, temporal_kernel_size=3)
        # self.encoder5 = self._create_encoder_block(128, [256, 256, 256],
        #                                            cube_kernel_size=1, spatial_kernel_size=1, temporal_kernel_size=1)

        # self.encoder5 = KANBlock(dim=128, num_heads=8)
        # self.encoder5 = GlobalAttBlock(dim=128, num_heads=8)
        self.encoder5 = AxialAttBlock(dim=128, num_heads=8)

        # self.encoder5 = nn.Sequential(
        #     AxialAttBlock(dim=128, num_heads=8),
        #     GlobalAttBlock(dim=128, num_heads=8),
        #     # AxialAttBlock(dim=128, num_heads=8)
        # )
        # self.upconv5 = self._create_up_conv(128, 128)
        self.upconv5 = CARAFE3D(128, 128)

        self.decoder4 = self._create_decoder_block(128 * 2, 128, False)
        self.decoder3 = self._create_decoder_block(64 * 2, 64, False)
        self.decoder2 = self._create_decoder_block(32 * 2, 32, False)
        self.decoder1 = self._create_decoder_block(16 * 2, 1, True)

        # self.decoder4 = SE3dBlock(128, 128, 1, 1, 16)
        # self.decoder3 = SE3dBlock(64, 64, 1, 1, 16)
        # self.decoder2 = SE3dBlock(32, 32, 1, 1, 16)

        # self.upconv5 = CARAFE3D(256, 128)
        self.upconv4 = CARAFE3D(128, 64)
        self.upconv3 = CARAFE3D(64, 32)
        self.upconv2 = CARAFE3D(32, 16)

        # self.upconv5 = self._create_up_conv(256, 128)
        # self.upconv4 = self._create_up_conv(128, 64)
        # self.upconv3 = self._create_up_conv(64, 32)
        # self.upconv2 = self._create_up_conv(32, 16)

        self.pool = nn.MaxPool3d(2, 2)

    def forward(self, x):
        _, _, h, w, d = x.shape

        if (h % 16 != 0) or (w % 16 != 0) or (d % 16 != 0):
            raise ValueError(f"Invalid volume size ({h}, {w}, {d}). The dimension need to be divisible by 16.")

        x1 = self.encoder1(x)
        x = self.pool(x1)

        x2 = self.encoder2(x)  #[1, 16, 56, 56, 16] -> [1, 32, 56, 56, 16]
        x = self.pool(x2)

        x3 = self.encoder3(x)  # [1 ,32 ,28, 28, 8] -> [1, 64, 28, 28, 8]
        x = self.pool(x3)

        x4 = self.encoder4(x)  # [1, 64, 14, 14, 4] -> [1, 128, 14, 14, 4]
        x = self.pool(x4)

        x = self.encoder5(x)  # [1, 128, 7, 7, 2] -> [1, 256, 7 ,7, 2]

        x = self.upconv5(x)
        x = torch.cat([x, x4], dim=1)
        x = self.decoder4(x)

        x = self.upconv4(x)
        x = torch.cat([x, x3], dim=1)
        x = self.decoder3(x)

        x = self.upconv3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.decoder2(x)

        x = self.upconv2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.decoder1(x)

        return x


    # def _create_encoder_block(
    #         self,
    #         in_channel,
    #         channels,
    #         down_sampling=False,
    # ):
    #
    #     def _create_residual_unit(
    #             in_channels, out_channels, strides,
    #     ):
    #         return ResidualUnit(
    #             3,
    #             in_channels,
    #             out_channels,
    #             strides=strides,
    #             kernel_size=3,
    #             subunits=2,
    #             act=Act.PRELU,
    #             norm=Norm.INSTANCE,
    #             dropout=0.0,
    #             bias=True,
    #             adn_ordering="NDA",
    #         )
    #
    #     _channels = [in_channel, *channels]
    #
    #     units = []
    #     for i in range(len(channels) - 1):
    #         units.append(_create_residual_unit(_channels[i], _channels[i + 1], 1))
    #     units.append(
    #         _create_residual_unit(channels[-2], channels[-1], 2 if down_sampling else 1)
    #     )
    #
    #     return nn.Sequential(*units)

    def _create_encoder_block(
            self,
            in_channel,
            channels,
            down_sampling=False, cube_kernel_size=3, spatial_kernel_size=11, temporal_kernel_size=11

    ):

        def _create_residual_unit(
                in_channels, out_channels, strides,
        ):
            return ResidualUnit(
                3,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=3,
                subunits=2,
                act=Act.PRELU,
                norm=Norm.INSTANCE,
                dropout=0.0,
                bias=True,
                adn_ordering="NDA",
            )

        _channels = [in_channel, *channels]

        units = []
        for i in range(len(channels) - 1):
            if in_channel < 8:
                units.append(_create_residual_unit(_channels[i], _channels[i + 1], 1))
            else:
                units.append(InceptionDWConv3dBlock(_channels[i], _channels[i + 1], 1,
                                                    cube_kernel_size=cube_kernel_size,
                                                    spatial_kernel_size=spatial_kernel_size,
                                                    temporal_kernel_size=temporal_kernel_size))
            if in_channel < 8:
                units.append(_create_residual_unit(channels[-2], channels[-1], 2 if down_sampling else 1))
            else:
                units.append(InceptionDWConv3dBlock(channels[-2], channels[-1], 2 if down_sampling else 1,
                                                    cube_kernel_size=cube_kernel_size,
                                                    spatial_kernel_size=spatial_kernel_size,
                                                    temporal_kernel_size=temporal_kernel_size))

        return nn.Sequential(*units)

    def _create_decoder_block(
            self,
            in_channels,
            out_channels,
            is_top,
    ):
        res_unit = ResidualUnit(
            3,
            in_channels,
            out_channels,
            strides=1,
            kernel_size=3,
            subunits=2,
            act=Act.PRELU,
            norm=Norm.INSTANCE,
            dropout=0.0,
            bias=True,
            last_conv_only=is_top,
            adn_ordering="NDA",
        )

        return res_unit

    def _create_up_conv(
            self,
            in_channels,
            out_channels,
    ):
        return Convolution(
            3,
            in_channels,
            out_channels,
            strides=2,
            kernel_size=3,
            act=Act.PRELU,
            norm=Norm.INSTANCE,
            dropout=0.0,
            bias=True,
            is_transposed=True,
            adn_ordering="NDA",
        )


class UNetIDC3DSmall(UNetIDC3D):
    def __init__(self):
        super().__init__()

        self.encoder1 = self._create_encoder_block(3, [16, 16])
        self.encoder2 = self._create_encoder_block(16, [32, 32])
        self.encoder3 = self._create_encoder_block(32, [64, 64])
        self.encoder4 = self._create_encoder_block(64, [128, 128])
        self.encoder5 = self._create_encoder_block(128, [256, 256])

        self.decoder4 = self._create_decoder_block(128 * 2, 128, False)
        self.decoder3 = self._create_decoder_block(64 * 2, 64, False)
        self.decoder2 = self._create_decoder_block(32 * 2, 32, False)
        self.decoder1 = self._create_decoder_block(16 * 2, 1, True)

        self.upconv5 = self._create_up_conv(256, 128)
        self.upconv4 = self._create_up_conv(128, 64)
        self.upconv3 = self._create_up_conv(64, 32)
        self.upconv2 = self._create_up_conv(32, 16)

        self.pool = nn.MaxPool3d(2, 2)


if __name__ == "__main__":
    # 初始化模型
    model = UNetIDC3D().cuda(0)  # 或者使用 UNet3DSmall()
    # model = OnlyUKAN3D().cuda(1)  # 或者使用 UNet3DSmall()
    # 伪造输入数据：假设输入形状为 (batch_size=1, channels=3, depth=128, height=128, width=128)
    # input = torch.randn(1, 3, 112, 112, 32).cuda(1)
    input = torch.randn(1, 3, 112, 112, 32).cuda(0)

    output = model(input)

    from thop import profile

    flops, params = profile(model, inputs=(input,))
    print('Flops: ', flops, ', Params: ', params)
    print('FLOPs&Params: ' + 'GFLOPs: %.2f G, Params: %.2f MB' % (flops / 1e9, params / 1e6))

    print(f"输出形状: {output.shape}")

    # 测试真实数据
    ####################################################################################################################

    # # step1 加载模型权重
    # checkpoint_path = r'/workdir3t/A-Echo/hcc-echo-mae/lightning_logs/version_25/checkpoints/epoch=35-step=12424.ckpt'
    # checkpoint = torch.load(checkpoint_path, map_location='cuda:0', weights_only=True)
    #
    # # 获取 state_dict
    # state_dict = checkpoint['state_dict']
    #
    # # 移除 'model.' 前缀
    # new_state_dict = {}
    # for key in state_dict:
    #     new_key = key.replace("model.", "")  # 移除 'model.' 前缀
    #     new_state_dict[new_key] = state_dict[key]
    #
    # # step2 初始化模型并加载权重
    # model = UNetIDC3D()
    # model.load_state_dict(new_state_dict)
    # model.eval()
    #
    # # 将模型加载到 GPU 1
    # model = model.cuda(0)
    #
    #
    # # step3 加载真实数据
    # from simlvseg.model.utils.img2tensor import video_to_tensor, tensor_to_video_grayscale
    #
    # video_tensor = video_to_tensor(
    #     r'/workdir3t/A-Echo/hcc-echo-mae/data/EchoNet-Dynamic/dev_data/0X1A0A263B22CCD966.avi')
    # input = video_tensor[:, :, :, :, 0:32].cuda(0)
    #
    # # step4 前向传播
    # output = model(input)
    #
    # # from thop import profile
    # #
    # # flops, params = profile(model, inputs=(input,))
    # # print('Flops: ', flops, ', Params: ', params)
    # # print('FLOPs&Params: ' + 'GFLOPs: %.2f G, Params: %.2f MB' % (flops / 1e9, params / 1e6))
    #
    # # 打印输出形状
    # print(f"输出形状: {output.shape}")
    #
    # # 保存输出的mask视频
    # tensor_to_video_grayscale(output, 'output.avi')
