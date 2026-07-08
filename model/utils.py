import numpy as np
import torch
import torch.nn as nn
import pdb

def get_model(args, pretrain=False):
    """Function to get the model based on the arguments.
    Actually the models available, only 3 dimensions, are:
        - UNETR
        - UNet
        - AttentionUNet
        - SegFormer
        - UNet++
        - SwinUNETR
        - UNETR++
        - nnMAMBA
        - segMAMBA
        - UxLSTM
        - MedSAM

    Args:
        args (argparse.Namespace): Arguments from the command line.
        pretrain (bool, optional): Set to true if you use a pretrained model. Defaults to False.

    Raises:
        ValueError: No pretrain model available
        ValueError: Invalid dimension, should be '2d' or '3d'

    Returns:
        Model: The model object.
    """    
    
    if args.dimension == '3d':
        
        if args.model == 'swin_unetr':
            from .dim3 import SwinUNETR
            return SwinUNETR(
                img_size=args.img_size,
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                depths=args.depths,
                num_heads=args.num_heads,
                feature_size=args.feature_size,
                norm_name=args.norm_name,
                drop_rate=args.drop_rate,
                attn_drop_rate=args.attn_drop_rate,
                dropout_path_rate=args.dropout_path_rate,
                normalize=args.normalize,
                use_checkpoint=args.use_checkpoint,
                spatial_dims=args.spatial_dims,
                downsample=args.downsample,
                use_v2=args.use_v2,
            )
        
        elif args.model == 'unet':
            from .dim3 import UNet
            return UNet(
                spatial_dims=args.spatial_dims,
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                channels=args.channels,
                strides=args.strides,
                kernel_size=args.kernel_size,
                up_kernel_size=args.up_kernel_size,
                num_res_units=args.num_res_units,
                act=args.act,
                norm=args.norm,
                dropout=args.dropout,
                bias=args.bias,
                adn_ordering=args.adn_ordering,
            )
        
        elif args.model == 'unetr':
            from .dim3 import UNETR
            return UNETR(
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                img_size=args.roi_size, 
                feature_size=args.feature_size,
                hidden_size=args.hidden_size,
                mlp_dim=args.mlp_dim,
                num_heads=args.num_heads,
                norm_name=args.norm_name,
                conv_block=args.conv_block,
                res_block=args.res_block,
                dropout_rate=args.dropout_rate,
                spatial_dims=args.spatial_dims,
                qkv_bias=args.qkv_bias,
            )

        elif args.model == 'attention_unet':
            from .dim3 import AttentionUnet
            return AttentionUnet(
                spatial_dims=args.spatial_dims,
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                channels=args.channels,
                strides=args.strides,
                kernel_size=args.kernel_size,
                up_kernel_size=args.up_kernel_size,
                dropout=args.dropout,
            )

        elif args.model == 'unet++':
            from .dim3 import BasicUNetPlusPlus
            return BasicUNetPlusPlus(
                spatial_dims=args.spatial_dims,
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                features=args.features,
                deep_supervision=args.deep_supervision,
                act=args.act,
                norm=args.norm,
                bias=args.bias,
                dropout=args.dropout,
                upsample=args.upsample,
            )
        
        elif args.model == 'segformer':
            from .dim3 import SegFormer3D
            return SegFormer3D(
                in_channels=args.in_channels,
                num_classes=args.out_channels,
                sr_ratios=args.sr_ratios,
                embed_dims=args.embed_dims,
                patch_kernel_size=args.patch_kernel_size,
                patch_stride=args.patch_stride,
                patch_padding=args.patch_padding,
                mlp_ratios=args.mlp_ratios,
                num_heads=args.num_heads,
                depths=args.depths,
                decoder_head_embedding_dim=args.decoder_head_embedding_dim,
                decoder_dropout=args.decoder_dropout,
            )
                
        elif args.model == 'uxlstm':
            from .dim3 import UXlstmBot
            norm_cls = getattr(nn, args.norm_op.split('.')[-1]) 
            nonlin_cls = getattr(nn, args.nonlin.split('.')[-1])
            
            return UXlstmBot(
                input_channels=args.input_channels,
                n_stages=args.n_stages,
                features_per_stage=args.features_per_stage,
                conv_op=getattr(nn, args.conv_op.split('.')[-1]),
                kernel_sizes=args.kernel_sizes,
                strides=args.strides,
                n_conv_per_stage=args.n_conv_per_stage,
                num_classes=args.num_classes,
                n_conv_per_stage_decoder=args.n_conv_per_stage_decoder,
                conv_bias=args.conv_bias,
                norm_op=norm_cls, 
                norm_op_kwargs=args.norm_op_kwargs,
                dropout_op=None,
                dropout_op_kwargs=None,
                nonlin=nonlin_cls,
                nonlin_kwargs=args.nonlin_kwargs,
                deep_supervision=args.deep_supervision,
                stem_channels=args.stem_channels,
            )
        
        elif args.model == 'unetr++':
            from .dim3 import UNETR_PP
            return UNETR_PP(
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                img_size=args.img_size,
                feature_size=args.feature_size,
                hidden_size=args.hidden_size,
                num_heads=args.num_heads,
                pos_embed=args.pos_embed,
                norm_name=args.norm_name,
                dropout_rate=args.dropout_rate,
                depths=args.depths,
                dims=args.dims,
                conv_op=args.conv_op,
                do_ds=args.do_ds,
            )
        
        
        elif args.model == 'segmamba':
            from .dim3 import SegMamba
            return SegMamba(
                in_chans=args.in_chans,
                out_chans=args.out_chans,
                depths=args.depths,
                feat_size=args.feat_size,
                drop_path_rate=args.drop_path_rate,
                layer_scale_init_value=args.layer_scale_init_value,
                hidden_size=args.hidden_size,
                norm_name=args.norm_name,
                conv_block=args.conv_block,
                res_block=args.res_block,
                spatial_dims=args.spatial_dims,
            )
        
        elif args.model == 'nnmamba':
            from .dim3 import nnMambaSeg
            return nnMambaSeg(
                in_ch=args.in_ch,
                channels=args.channels,
                blocks=args.blocks,
                number_classes=args.number_classes,
            )
        
    # 2D MODEL BELOW ↓    
    elif args.dimension == '2d':

        if args.model == 'swin_unetr':
            from .dim2 import SwinUNETR
            return SwinUNETR(
                img_size=args.img_size,
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                depths=args.depths,
                num_heads=args.num_heads,
                feature_size=args.feature_size,
                norm_name=args.norm_name,
                drop_rate=args.drop_rate,
                attn_drop_rate=args.attn_drop_rate,
                dropout_path_rate=args.dropout_path_rate,
                normalize=args.normalize,
                use_checkpoint=args.use_checkpoint,
                spatial_dims=args.spatial_dims,
                downsample=args.downsample,
                use_v2=args.use_v2,
            )
        
        elif args.model == 'unet':
            from .dim2 import UNet
            return UNet(
                spatial_dims=args.spatial_dims,
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                channels=args.channels,
                strides=args.strides,
                kernel_size=args.kernel_size,
                up_kernel_size=args.up_kernel_size,
                num_res_units=args.num_res_units,
                act=args.act,
                norm=args.norm,
                dropout=args.dropout,
                bias=args.bias,
                adn_ordering=args.adn_ordering,
            )
        
        elif args.model == 'unetr':
            from .dim2 import UNETR
            return UNETR(
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                img_size=args.roi_size, 
                feature_size=args.feature_size,
                hidden_size=args.hidden_size,
                mlp_dim=args.mlp_dim,
                num_heads=args.num_heads,
                norm_name=args.norm_name,
                conv_block=args.conv_block,
                res_block=args.res_block,
                dropout_rate=args.dropout_rate,
                spatial_dims=args.spatial_dims,
                qkv_bias=args.qkv_bias,
            )

        elif args.model == 'attention_unet':
            from .dim2 import AttentionUnet
            return AttentionUnet(
                spatial_dims=args.spatial_dims,
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                channels=args.channels,
                strides=args.strides,
                kernel_size=args.kernel_size,
                up_kernel_size=args.up_kernel_size,
                dropout=args.dropout,
            )

        elif args.model == 'unet++':
            from .dim2 import BasicUNetPlusPlus
            return BasicUNetPlusPlus(
                spatial_dims=args.spatial_dims,
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                features=args.features,
                deep_supervision=args.deep_supervision,
                act=args.act,
                norm=args.norm,
                bias=args.bias,
                dropout=args.dropout,
                upsample=args.upsample,
            )
        
        elif args.model == 'segformer':
            from .dim2 import SegFormer3D
            return SegFormer3D(
                in_channels=args.in_channels,
                num_classes=args.out_channels,
                sr_ratios=args.sr_ratios,
                embed_dims=args.embed_dims,
                patch_kernel_size=args.patch_kernel_size,
                patch_stride=args.patch_stride,
                patch_padding=args.patch_padding,
                mlp_ratios=args.mlp_ratios,
                num_heads=args.num_heads,
                depths=args.depths,
                decoder_head_embedding_dim=args.decoder_head_embedding_dim,
                decoder_dropout=args.decoder_dropout,
            )
                
        elif args.model == 'uxlstm':
            from .dim2 import UXlstmBot
            norm_cls = getattr(nn, args.norm_op.split('.')[-1]) 
            nonlin_cls = getattr(nn, args.nonlin.split('.')[-1])
            
            return UXlstmBot(
                input_channels=args.input_channels,
                n_stages=args.n_stages,
                features_per_stage=args.features_per_stage,
                conv_op=getattr(nn, args.conv_op.split('.')[-1]),
                kernel_sizes=args.kernel_sizes,
                strides=args.strides,
                n_conv_per_stage=args.n_conv_per_stage,
                num_classes=args.num_classes,
                n_conv_per_stage_decoder=args.n_conv_per_stage_decoder,
                conv_bias=args.conv_bias,
                norm_op=norm_cls, 
                norm_op_kwargs=args.norm_op_kwargs,
                dropout_op=None,
                dropout_op_kwargs=None,
                nonlin=nonlin_cls,
                nonlin_kwargs=args.nonlin_kwargs,
                deep_supervision=args.deep_supervision,
                stem_channels=args.stem_channels,
            )
        
        elif args.model == 'unetr++':
            from .dim2 import UNETR_PP
            return UNETR_PP(
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                img_size=args.img_size,
                feature_size=args.feature_size,
                hidden_size=args.hidden_size,
                num_heads=args.num_heads,
                pos_embed=args.pos_embed,
                norm_name=args.norm_name,
                dropout_rate=args.dropout_rate,
                depths=args.depths,
                dims=args.dims,
                conv_op=args.conv_op,
                do_ds=args.do_ds,
            )
        
        
        elif args.model == 'segmamba':
            from .dim2 import SegMamba
            return SegMamba(
                in_chans=args.in_chans,
                out_chans=args.out_chans,
                depths=args.depths,
                feat_size=args.feat_size,
                drop_path_rate=args.drop_path_rate,
                layer_scale_init_value=args.layer_scale_init_value,
                hidden_size=args.hidden_size,
                norm_name=args.norm_name,
                conv_block=args.conv_block,
                res_block=args.res_block,
                spatial_dims=args.spatial_dims,
            )
        
        elif args.model == 'nnmamba':
            from .dim2 import nnMambaSeg
            return nnMambaSeg(
                in_ch=args.in_ch,
                channels=args.channels,
                blocks=args.blocks,
                number_classes=args.number_classes,
            )

    else:
        raise ValueError('Invalid dimension, should be \'2d\' or \'3d\'')

