# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import numpy as np

def download_clip():  # TODO separate to a utility file
    import clip
    clip_model, clip_preprocess = clip.load("ViT-B/32")
    clip_model.cuda().eval()
    # print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in clip_model.parameters()]):,}")
    print('model parameters', f"{np.sum([int(np.prod(p.shape)) for p in clip_model.parameters()]):,}")
    print('input resolution: ', clip_model.visual.input_resolution)
    print('context length: ', clip_model.context_length)
    print('vocab size: ', clip_model.vocab_size)
    return clip_model, clip_preprocess


class ClipModelLoader:
    _instance = None

    @classmethod
    def load_instance(cls):
        if cls._instance is None:
            print('load clip model!!!')
            cls._instance, _ = download_clip()
        return cls._instance


clip_model_loader = ClipModelLoader()



