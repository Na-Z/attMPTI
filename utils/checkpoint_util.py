""" Util functions for loading and saving checkpoints

Author: Zhao Na, 2020
"""
import os
import torch


def load_pretrain_checkpoint(model, pretrain_checkpoint_path):
    # load pretrained model for point cloud encoding
    model_dict = model.state_dict()
    if pretrain_checkpoint_path is not None:
        print('Load encoder module from pretrained checkpoint...')
        pretrained_dict = torch.load(os.path.join(pretrain_checkpoint_path, 'checkpoint.tar'))['params']
        pretrained_dict = {'encoder.' + k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        raise ValueError('Pretrained checkpoint must be given.')

    return model


def load_model_checkpoint(model, model_checkpoint_path, optimizer=None, mode='test'):
    try:
        checkpoint = torch.load(os.path.join(model_checkpoint_path, 'checkpoint.tar'))
        start_iter = checkpoint['iteration']
        start_iou = checkpoint['IoU']
    except:
        raise ValueError('Model checkpoint file must be correctly given (%s).' %model_checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if mode == 'test':
        print('Load model checkpoint at Iteration %d (IoU %f)...' % (start_iter, start_iou))
        return model
    else:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            print('Checkpoint does not include optimizer state dict...')
        print('Resume from checkpoint at Iteration %d (IoU %f)...' % (start_iter, start_iou))
        return model, optimizer

def save_pretrain_checkpoint(model, output_path):
    torch.save(dict(params=model.encoder.state_dict()), os.path.join(output_path, 'checkpoint.tar'))