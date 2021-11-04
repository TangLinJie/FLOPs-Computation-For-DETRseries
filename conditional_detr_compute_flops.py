# this is the main entrypoint
# as we describe in the paper, we compute the flops over the first 100 images
# on COCO val2017, and report the average result
import torch
import time
import torchvision
import argparse

import numpy as np
import tqdm

from models import build_model
# from datasets import build_dataset

from flop_count import flop_count


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def get_dataset(coco_path):
    """
    Gets the COCO dataset used for computing the flops on
    """
    class DummyArgs:
        pass
    args = DummyArgs()
    args.dataset_file = "coco"
    args.coco_path = coco_path
    args.masks = False
    dataset = build_dataset(image_set='val', args=args)
    return dataset


def warmup(model, inputs, N=10):
    for i in range(N):
        out = model(*inputs)
    torch.cuda.synchronize()


def measure_time(model, inputs, N=10):
    warmup(model, inputs)
    s = time.time()
    for i in range(N):
        out = model(*inputs)
    torch.cuda.synchronize()
    t = (time.time() - s) / N
    return t


def fmt_res(data):
    return data.mean(), data.std(), data.min(), data.max()


def computing_time_flops(model, images, device):
    tmp = []
    tmp2 = []
    for img in tqdm.tqdm(images):
        inputs = [img.to(device)]
        res = flop_count(model, (inputs,))
        t = measure_time(model, [inputs])
        tmp.append(sum(res.values()))
        tmp2.append(t)
    return tmp, tmp2


def computing_time_flops_with_multi_params(model, params):
    tmp = []
    tmp2 = []
    for param in tqdm.tqdm(params):
        res = flop_count(model, param)
        t = measure_time(model, param)
        tmp.append(sum(res.values()))
        tmp2.append(t)
    return tmp, tmp2


def main(args):
    """
    # get the first 100 images of COCO val2017
    PATH_TO_COCO = "/path/to/coco/"
    dataset = get_dataset(PATH_TO_COCO)
    images = []
    for idx in range(100):
        img, t = dataset[idx]
        images.append(img)
    """
    device = torch.device('cuda')
    # using fake data
    images = [torch.randn(3, 640, 640)]
    feat_size = 80
    dim = 256
    num_queries = 100
    # for transformer inputs
    # NOTE keep order
    transformer_src = torch.randn(1, dim, feat_size, feat_size)
    transformer_mask = torch.zeros(1, feat_size, feat_size).int().bool()
    transformer_query_embed = torch.randn(num_queries, dim)
    transformer_pos = torch.randn(1, dim, feat_size, feat_size)
    transformer_input_params = [transformer_src.to(device), transformer_mask.to(device), \
                                transformer_query_embed.to(device), transformer_pos.to(device)]
    # for encoder inputs
    encoder_ratio = 1
    transformer_encoder_src = torch.randn(feat_size*feat_size*encoder_ratio, 1, dim)
    transformer_encoder_src_key_padding_mask = torch.zeros(1, feat_size*feat_size*encoder_ratio).int().bool()
    transformer_encoder_pos = torch.randn(feat_size*feat_size*encoder_ratio, 1, dim)
    transformer_encoder_input_params = [transformer_encoder_src.to(device), None, \
                                transformer_encoder_src_key_padding_mask.to(device), \
                                transformer_encoder_pos.to(device)]
    # for decoder inputs
    decoder_ratio = 1
    transformer_decoder_tgt = torch.zeros(num_queries, 1, dim)
    transformer_decoder_memory = torch.randn(feat_size*feat_size*decoder_ratio, 1, dim)
    transformer_decoder_memory_key_padding_mask = torch.zeros(1, feat_size*feat_size*decoder_ratio).int().bool()
    transformer_decoder_pos = torch.randn(feat_size*feat_size*decoder_ratio, 1, dim)
    transformer_decoder_query_pos = torch.randn(num_queries, 1, dim)
    transformer_decoder_input_params = [transformer_decoder_tgt.to(device), transformer_decoder_memory.to(device), \
                                        None, None, None, transformer_decoder_memory_key_padding_mask.to(device), \
                                        transformer_decoder_pos.to(device), transformer_decoder_query_pos.to(device)]

    results = {}
    for model_name in ['detr_resnet50']:
        # model = torch.hub.load('facebookresearch/detr', model_name, pretrained=True)
        model, criterion, postprocessors = build_model(args)
        model.to(device)

        # submodule
        transformer = model.transformer
        transformer_encoder = transformer.encoder
        transformer_decoder = transformer.decoder

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        transformer_n_parameters = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
        transformer_encoder_n_parameters = sum(p.numel() for p in transformer_encoder.parameters() if p.requires_grad)
        transformer_decoder_n_parameters = sum(p.numel() for p in transformer_decoder.parameters() if p.requires_grad)

        with torch.no_grad():
            tmp, tmp2 = computing_time_flops(model, images, device)
            tmp_transformer, tmp2_transformer = computing_time_flops_with_multi_params(transformer, [tuple(transformer_input_params)])
            tmp_transformer_encoder, tmp2_transformer_encoder = computing_time_flops_with_multi_params(transformer_encoder, \
                                                                                [tuple(transformer_encoder_input_params)])
            tmp_transformer_decoder, tmp2_transformer_decoder = computing_time_flops_with_multi_params(transformer_decoder, \
                                                                                [tuple(transformer_decoder_input_params)])


        results[model_name] = {'flops': fmt_res(np.array(tmp)), 'time': fmt_res(np.array(tmp2)), '#params': n_parameters/1e6, \
                            'transformer flops': fmt_res(np.array(tmp_transformer)), 'transformer time': fmt_res(np.array(tmp2_transformer)), \
                            'transformer #params': transformer_n_parameters/1e6, \
                            'transformer encoder flops': fmt_res(np.array(tmp_transformer_encoder)), \
                            'transformer encoder time': fmt_res(np.array(tmp2_transformer_encoder)), \
                            'transformer encoder #params': transformer_encoder_n_parameters/1e6, \
                            'transformer decoder flops': fmt_res(np.array(tmp_transformer_decoder)), \
                            'transformer decoder time': fmt_res(np.array(tmp2_transformer_decoder)), \
                            'transformer decoder #params': transformer_decoder_n_parameters/1e6}


    print('=============================')
    print('')
    for r in results:
        print(r)
        for k, v in results[r].items():
            print(' ', k, ':', v)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)