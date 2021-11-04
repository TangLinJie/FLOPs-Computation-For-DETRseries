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
    # TODO add defualt params your model needs
    ## NOTE you need to return a argparse.ArgumentParser object
    pass


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
    feat_size = 20
    dim = 256
    num_queries = 100

    # for transformer inputs
    # TODO add transformer input params
    ## NOTE all input param must be given, and put them in a list while keeping order
    ## For example:
    # transformer_src = torch.randn(1, dim, feat_size, feat_size)
    # transformer_mask = torch.zeros(1, feat_size, feat_size).int().bool()
    # transformer_query_embed = torch.randn(num_queries, dim)
    # transformer_pos = torch.randn(1, dim, feat_size, feat_size)
    # transformer_input_params = [transformer_src.to(device), transformer_mask.to(device), \
    #                             transformer_query_embed.to(device), transformer_pos.to(device)]

    # for encoder inputs
    # TODO add encoder input params
    ## For example:
    # encoder_ratio = 16
    # transformer_encoder_src = torch.randn(feat_size*feat_size*encoder_ratio, 1, dim)
    # transformer_encoder_src_key_padding_mask = torch.zeros(1, feat_size*feat_size*encoder_ratio).int().bool()
    # transformer_encoder_pos = torch.randn(feat_size*feat_size*encoder_ratio, 1, dim)
    # transformer_encoder_input_params = [transformer_encoder_src.to(device), None, \
    #                             transformer_encoder_src_key_padding_mask.to(device), \
    #                             transformer_encoder_pos.to(device)]

    # for decoder inputs
    # TODO add decoder input params
    ## For example:
    # decoder_ratio = 16
    # transformer_decoder_tgt = torch.zeros(num_queries, 1, dim)
    # transformer_decoder_memory = torch.randn(feat_size*feat_size*decoder_ratio, 1, dim)
    # transformer_decoder_memory_key_padding_mask = torch.zeros(1, feat_size*feat_size*decoder_ratio).int().bool()
    # transformer_decoder_pos = torch.randn(feat_size*feat_size*decoder_ratio, 1, dim)
    # transformer_decoder_query_pos = torch.randn(num_queries, 1, dim)
    # transformer_decoder_input_params = [transformer_decoder_tgt.to(device), transformer_decoder_memory.to(device), \
    #                                     None, None, None, transformer_decoder_memory_key_padding_mask.to(device), \
    #                                     transformer_decoder_pos.to(device), transformer_decoder_query_pos.to(device)]

    results = {}
    for model_name in ['detr_resnet50']:
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