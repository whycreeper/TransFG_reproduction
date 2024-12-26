# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import torch
import random
import numpy as np
import time

from datetime import timedelta

# import torch
import jittor

from tqdm import tqdm

from models.modeling import VisionTransformer, CONFIGS
from models.torch_modeling import t_VisionTransformer, t_CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.t_scheduler import t_WarmupCosineSchedule
from utils.data_utils import get_loader
# from utils.dist_util import get_world_size

logger = logging.getLogger(__name__)

import json
import datetime
import pytz
log_path = "/home/aiuser/TransFG/logs"
tz = pytz.timezone('Asia/Shanghai')
now_bj = datetime.datetime.now(tz)
# log_file = os.path.join(log_path, now_bj.strftime("%m-%d-%H-%M") + ".jsonl")
log_file = "/home/aiuser/TransFG/logs/torch.jsonl"


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

# def reduce_mean(tensor, nprocs):
#     rt = tensor.clone()
#     dist.all_reduce(rt, op=dist.ReduceOp.SUM)
#     rt /= nprocs
#     return rt

def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    if args.fp16:
        # checkpoint = {
        #     'model': model_to_save.state_dict(),
        #     'amp': amp.state_dict()
        # }
        None
    else:
        checkpoint = {
            'model': model_to_save.state_dict(),
        }
    jittor.save(checkpoint, model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    config.split = args.split
    config.slide_step = args.slide_step

    if args.dataset == "CUB_200_2011":
        num_classes = 200
    elif args.dataset == "car":
        num_classes = 196
    elif args.dataset == "nabirds":
        num_classes = 555
    elif args.dataset == "dog":
        num_classes = 120
    elif args.dataset == "INat2017":
        num_classes = 5089
        
    t_model = t_VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes,                                                   smoothing_value=args.smoothing_value)

    t_model.load_from(np.load(args.pretrained_dir))
    t_model.to(args.device)
    num_params = count_parameters(t_model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, None, t_model

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    jittor.misc.set_global_seed(args.seed, False)
    # import
    # torch.manual_seed(args.seed)
    # if args.n_gpu > 0:
    #     torch.cuda.manual_seed_all(args.seed)

def valid(args, model, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    # epoch_iterator = tqdm(test_loader,
    #                       desc="Validating... (loss=X.X)",
    #                       bar_format="{l_bar}{r_bar}",
    #                       dynamic_ncols=True,
    #                       disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    # for step, batch in enumerate(epoch_iterator):
    print("test_epoch", len(test_loader))
    cnt = 0
    for x, y in test_loader:
        cnt  = cnt + 1
        if(cnt % 100 == 0):
            print("cnt = ",cnt)
        # batch = tuple(t.to(args.device) for t in batch)
        # x, y = batch
        
        x = torch.from_numpy(x.numpy()).to("cuda")
        y = torch.from_numpy(y.numpy()).to("cuda").long()
        with jittor.no_grad():
            logits = model(x)
            
            y = y.squeeze(1)

            eval_loss = loss_fct(logits, y)
            eval_loss = eval_loss.mean()
            # eval_losses.update(eval_loss.numpy())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(torch.detach(preds).cpu())
            all_label.append(torch.detach(y).cpu())
        else:
            all_preds[0] = np.append(
                all_preds[0], torch.detach(preds).cpu(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], torch.detach(y).cpu(), axis=0
            )
        # epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)
    accuracy_jittor = torch.tensor(accuracy).to(args.device)
    # dist.barrier()
    # val_accuracy = reduce_mean(accuracy, args.nprocs)
    val_accuracy = torch.detach(accuracy_jittor).cpu().numpy()

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % val_accuracy)
    with open(log_file, 'a') as f:
        f.write(json.dumps({"val_accuracy": accuracy, "global_step": global_step}) + '\n')
        
    return val_accuracy

def train(args, _, t_model):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader = get_loader(args)

    # Prepare optimizer and scheduler
    # optimizer = jittor.optim.SGD(model.parameters(),
    #                             lr=args.learning_rate,
    #                             momentum=0.9,
    #                             # weight_decay=args.weight_decay)
    import torch
    t_optimizer = torch.optim.SGD(t_model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        # scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
        t_scheduler = t_WarmupCosineSchedule(t_optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
        None

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    # model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    
    global_step, best_acc = 0, 0
    start_time = time.time()
    while True:
        # model.train()
        t_model.train()
        all_preds, all_label = [], []
        print("train epoch", len(train_loader))
        for x, y in train_loader:
            if(global_step % 100 == 0):
                print("global step = ",global_step)
            # batch = tuple(t.to(args.device) for t in batch)
            # x, y = batch
            # loss, logits = model(x, y)
            
            tx = torch.from_numpy(x.numpy()).to("cuda")
            ty = torch.from_numpy(y.numpy()).to("cuda").long()
            t_loss, t_logits = t_model(tx, ty)
            # with open(log_file, 'a') as f:
            #     f.write(json.dumps({"y": y.detach().cpu().numpy().tolist()}) + '\n')
            # with open(log_file, 'a') as f:
            #     f.write(json.dumps({"loss": t_loss.detach().cpu().numpy().tolist()}) + '\n')
                
            # with open(log_file, 'a') as f:
            #     f.write(json.dumps({"logits": t_logits.detach().cpu().numpy().tolist()}) + '\n')
                
            # loss = loss.mean()
            t_loss = t_loss.mean()

            preds = torch.argmax(t_logits, dim=-1)
            print(ty, preds)

            # if len(all_preds) == 0:
            #     all_preds.append(jittor.detach(preds).cpu())
            #     all_label.append(jittor.detach(y).cpu())
            # else:
            #     all_preds[0] = np.append(
            #         all_preds[0], jittor.detach(preds).cpu(), axis=0
            #     )
            #     all_label[0] = np.append(
            #         all_label[0], jittor.detach(y).cpu(), axis=0
            #     )

            # if args.gradient_accumulation_steps > 1:
            #     loss = loss / args.gradient_accumulation_steps

            # optimizer.backward(loss)
            
            t_loss.backward()

            # if (step + 1) % args.gradient_accumulation_steps == 0:
            if True:
                

                # optimizer.clip_grad_norm(args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(t_model.parameters(), args.max_grad_norm)
                    
                # scheduler.step()
                t_scheduler.step()
                # optimizer.step()
                t_optimizer.step()
                t_optimizer.zero_grad()
                # optimizer.zero_grad()
                global_step += 1

                if global_step % args.eval_every == 0:
                    with jittor.no_grad():
                        accuracy = valid(args, t_model, test_loader, global_step)
                    if args.local_rank in [-1, 0]:
                        if best_acc < accuracy:
                            save_model(args, t_model)
                            best_acc = accuracy
                        logger.info("best accuracy so far: %f" % best_acc)
                    t_model.train()

                if global_step % t_total == 0:
                    break
        all_preds, all_label = all_preds[0], all_label[0]
        accuracy = simple_accuracy(all_preds, all_label)
        accuracy_jittor = jittor.array(accuracy).to(args.device)
        # dist.barrier()
        # train_accuracy = reduce_mean(accuracy, args.nprocs)
        train_accuracy = jittor.detach(accuracy_jittor).cpu()
        logger.info("train accuracy so far: %f" % train_accuracy)
        
        if global_step % t_total == 0:
            break

        with open(log_file, 'a') as f:
            f.write(json.dumps({"train_accuracy": accuracy, "global_step": global_step}) + '\n')

    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")
    end_time = time.time()
    logger.info("Total Training Time: \t%f" % ((end_time - start_time) / 3600))

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["CUB_200_2011", "car", "dog", "nabirds", "INat2017"], default="CUB_200_2011",
                        help="Which dataset.")
    parser.add_argument('--data_root', type=str, default='/home/aiuser/TransFG/minist')
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="/home/aiuser/TransFG/minist/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="load pretrained model")
    parser.add_argument("--output_dir", default="./output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--img_size", default=448, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=1000, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    parser.add_argument('--smoothing_value', type=float, default=0.0,
                        help="Label smoothing value\n")

    parser.add_argument('--split', type=str, default='non-overlap',
                        help="Split method")
    parser.add_argument('--slide_step', type=int, default=12,
                        help="Slide step for overlap split")

    args = parser.parse_args()

    # if args.fp16 and args.smoothing_value != 0:
    #     raise NotImplementedError("label smoothing not supported for fp16 training now")
    args.data_root = '{}/{}'.format(args.data_root, args.dataset)
    # Setup CUDA, GPU & distributed training
    print(args.local_rank, args.fp16)
    if args.local_rank == -1:
        device = "cuda"
        args.n_gpu = 1
    # else:
    #     # jittor.cuda.set_device(args.local_rank)
    #     device = "cuda"
    #     dist.init_process_group(backend='nccl',
    #                                          timeout=timedelta(minutes=60))
    #     args.n_gpu = 1
    torch.manual_seed(42)
    args.device=device
    args.nprocs = 1

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    with open(log_file, 'w') as f:
        f.write(json.dumps(vars(args)) + '\n')

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, _, t_model = setup(args)
    # Training
    train(args, _, t_model)

if __name__ == "__main__":
    main()
