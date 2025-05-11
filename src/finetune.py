import os
import time

import torch
import copy
from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.eval import evaluate
from src.eval import eval_single_dataset
from src.modeling import ImageEncoder, ImageClassifier, MultiHeadImageClassifier
from src.utils import cosine_lr, LabelSmoothing
from src.heads import get_classification_head
import matplotlib.pyplot as plt

import src.datasets as datasets
import pickle
from src.utils import WORK_DIR



# #WORK_DIR = '/home/group/self_improving/experiments/mixing'
# WORK_DIR = '/groups/gcd50678/mixing'
# #WORK_DIR = 'drive/MyDrive/mixing'


def finetune(args):
    train_dataset = args.train_dataset
    ckpdir = os.path.join(args.save, train_dataset, f'epoch_{args.epochs}')

    # Check if checkpoints already exist
    zs_path = os.path.join(args.save, train_dataset, 'checkpoint_0.pt')  
    ft_path = os.path.join(args.save, train_dataset, 'finetuned.pt')
    if os.path.exists(zs_path) and os.path.exists(ft_path):
        evaluate(torch.load(ft_path), args)
        print(f'Skipping fine-tuning because {ft_path} exists.')
        return zs_path, ft_path

    assert train_dataset is not None, "Please provide a training dataset."
    if args.load is not None and args.load.endswith('pt'):
        image_encoder = ImageEncoder.load(args.load)
    else:
        print('Building image encoder.')
        image_encoder = ImageEncoder(args, keep_lang=False)

    classification_head = get_classification_head(args, train_dataset)

    model = ImageClassifier(image_encoder, classification_head)

    model.freeze_head()
    torch.manual_seed(args.seed)

    preprocess_fn = model.train_preprocess
    print_every = 100
    save_every = 1000

    dataset = get_dataset(
        train_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size
    )
    num_batches = len(dataset.train_loader)

    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    model = torch.nn.DataParallel(model, device_ids=devices)

    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()


    params = [p for p in model.parameters() if p.requires_grad]
    lrs = []
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    # Saving zero-shot model
    if args.save is not None:
        os.makedirs(ckpdir, exist_ok=True)
        model_path = os.path.join(args.save, f'zeroshot.pt')
        model.module.image_encoder.save(model_path)

    validation_acc = []
    

    for epoch in range(args.epochs):
        model = model.cuda()
        model.train()
        data_loader = get_dataloader(
            dataset, is_train=True, args=args, image_encoder=None)

        for i, batch in enumerate(data_loader):
            start_time = time.time()
            
            step = i + epoch * num_batches
            scheduler(step)
            lrs.append(optimizer.param_groups[0]['lr'])
            optimizer.zero_grad()

            batch = maybe_dictionarize(batch)
            inputs = batch['images'].to('cuda:0')
            labels = batch['labels'].to('cuda:0')
            data_time = time.time() - start_time
    
            logits = model(inputs)

            loss = loss_fn(logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(params, 1.0)

            optimizer.step()
            batch_time = time.time() - start_time

            if step % print_every == 0:
                percent_complete = 100 * i / len(data_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )
        
        # Evaluate the validation acc
        curr_val_acc = eval_single_dataset(model.module.image_encoder, f'{args.train_dataset}', args)['top1']
        print(f"Validation Accuracy: {curr_val_acc}")
        validation_acc.append(curr_val_acc)


        if args.save_what == 'all':
            checkpoint_path = os.path.join(ckpdir, 'checkpoint_{}.pt'.format(epoch))
            model.module.image_encoder.save(checkpoint_path)
        elif args.save_what == 'finetuned':
            continue


          
    # Plot Validation Acc
    plt.plot([e for e in range(len(validation_acc))], validation_acc)
    plot_save_path = os.path.join(ckpdir, 'validation_acc_plot')
    plt.savefig(plot_save_path)
    plt.close()

    # Evaluate
    image_encoder = model.module.image_encoder
    evaluate(image_encoder, args)

    if args.save is not None:
        zs_path = os.path.join(ckpdir, 'zeroshot.pt')  
        ft_path = os.path.join(ckpdir, 'finetuned.pt')
        image_encoder.save(ft_path)
        with open(os.path.join(ckpdir, 'learning_rates'), 'wb') as f:
            pickle.dump(lrs, f)
        with open(os.path.join(ckpdir, 'validation_acc'), 'wb') as f:
            pickle.dump(validation_acc, f)
        return zs_path, ft_path


def finetune_feddyn(args):
    train_dataset = args.train_dataset
    ckpdir = os.path.join(args.save, f'{train_dataset}/epoch_{args.epochs}')


    assert train_dataset is not None, "Please provide a training dataset."
    if args.load is not None and args.load.endswith('pt'):
        image_encoder = ImageEncoder.load(args.load)
    else:
        print('Building image encoder.')
        image_encoder = ImageEncoder(args, keep_lang=False)

    classification_head = get_classification_head(args, train_dataset)

    model = ImageClassifier(image_encoder, classification_head)
    

    model.freeze_head()
    torch.manual_seed(args.seed)
    preprocess_fn = model.train_preprocess


    dataset = get_dataset(
        train_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size
    )
    num_batches = len(dataset.train_loader)

    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    model = torch.nn.DataParallel(model, device_ids=devices)
    zeroshot = copy.deepcopy(model)

    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()


    params = [p for p in model.parameters() if p.requires_grad]
    lrs = []
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    # Saving zero-shot model
    if args.save is not None:
        os.makedirs(args.save, exist_ok=True)
        model_path = os.path.join(args.save, 'zeroshot.pt')
        if not os.path.exists(model_path):
            model.module.image_encoder.save(model_path)
            print('zeroshot saved')


    # Define prev_grad
    prev_grads = None
    for param in model.parameters():
        if param.requires_grad:
            if not isinstance(prev_grads, torch.Tensor):
                prev_grads = torch.zeros_like(param.view(-1)).cuda()
            else:
                prev_grads = torch.cat((prev_grads, torch.zeros_like(param.view(-1), device = 'cuda:0')), dim=0).cuda()


    validation_acc = []
    print_every = 100
    for epoch in range(args.epochs):
        model = model.cuda()
        model.train()
        data_loader = get_dataloader(
            dataset, is_train=True, args=args, image_encoder=None)

        for i, batch in enumerate(data_loader):
            start_time = time.time()
            
            step = i + epoch * num_batches
            scheduler(step)
            lrs.append(optimizer.param_groups[0]['lr'])
            optimizer.zero_grad()

            batch = maybe_dictionarize(batch)
            inputs = batch['images'].to('cuda:0')
            labels = batch['labels'].to('cuda:0')
            data_time = time.time() - start_time
    
            logits = model(inputs)

            loss = loss_fn(logits, labels)


            # Linear term loss
            lin_penalty = 0.0
            curr_params = None
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if not isinstance(curr_params, torch.Tensor):
                        curr_params = param.view(-1)
                    else:
                        curr_params = torch.cat((curr_params, param.view(-1)), dim=0)

            lin_penalty = torch.sum(curr_params * prev_grads)
            loss -= lin_penalty

            # Quadratic penalty 

            quad_penalty = 0.0
            for name, param in model.named_parameters():
                if param.requires_grad:
                    quad_penalty += torch.nn.functional.mse_loss(param, zeroshot.state_dict()[name].cuda(), reduction='sum')

            loss += args.alpha_feddyn/2.0 * quad_penalty

            loss.backward()

            # Update the previous gradients
            prev_grads = None
            for param in model.parameters():
                if param.requires_grad:
                    if not isinstance(prev_grads, torch.Tensor):
                        prev_grads = param.grad.view(-1).clone().cuda()
                    else:
        
                        prev_grads = torch.cat((prev_grads, param.grad.view(-1).clone()), dim=0).cuda()

            torch.nn.utils.clip_grad_norm_(params, 1.0)

            optimizer.step()
            batch_time = time.time() - start_time

            if step % print_every == 0:
                percent_complete = 100 * i / len(data_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )
        
        # Evaluate the validation acc
        curr_val_acc = eval_single_dataset(model.module.image_encoder, f'{args.train_dataset}', args)['top1']
        print(f"Validation Accuracy: {curr_val_acc}")
        validation_acc.append(curr_val_acc)


        if args.save_what == 'all':
            checkpoint_path = os.path.join(ckpdir, 'checkpoint_{}.pt'.format(epoch))
            model.module.image_encoder.save(checkpoint_path)
        elif args.save_what == 'finetuned':
            continue


          
    # Plot Validation Acc
    plt.plot([e for e in range(len(validation_acc))], validation_acc)
    plot_save_path = os.path.join(ckpdir, 'validation_acc_plot')
    plt.savefig(plot_save_path)
    plt.close()

    # Evaluate
    image_encoder = model.module.image_encoder
    evaluate(image_encoder, args)

    if args.save is not None:
        zs_path = os.path.join(ckpdir, 'zeroshot.pt')  
        ft_path = os.path.join(ckpdir, 'finetuned.pt')
        image_encoder.save(ft_path)
        with open(os.path.join(ckpdir, 'learning_rates'), 'wb') as f:
            pickle.dump(lrs, f)
        with open(os.path.join(ckpdir, 'validation_acc'), 'wb') as f:
            pickle.dump(validation_acc, f)
        return zs_path, ft_path


if __name__ == '__main__':
    # data_location = '<your_data_location>'
    # models = ['ViT-B-32', 'ViT-B-16', 'ViT-L-14']
    # datasets = ['Cars', 'DTD', 'EuroSAT', 'GTSRB', 'MNIST', 'RESISC45', 'SUN397', 'SVHN']
    # epochs = {
    #     'Cars': 35,
    #     'DTD': 76,
    #     'EuroSAT': 12,
    #     'GTSRB': 11,
    #     'MNIST': 5,
    #     'RESISC45': 15,
    #     'SUN397': 14,
    #     'SVHN': 4,
    #     'ImageNet': 4
    # }


    data_location = os.path.join(WORK_DIR, 'datasets')
    model = 'ViT-B-32'
    save = os.path.join(WORK_DIR, f'ta_experiments_standard_1e-05/checkpoints/{model}')
    ds = 'EuroSATVal'
    args = parse_arguments()
    args.train_dataset = 'DTDVal'
    args.data_location = data_location
    args.model = model
    args.save = save
    args.batch_size = 128
    args.lr = 1e-5
    args.epochs = 76
    finetune(args)
    #ckp = os.path.join(args.save, f'{args.train_dataset}/epoch_{args.epochs}/finetuned.pt')
    #eval_single_dataset(torch.load(ckp), 'DTD', args)