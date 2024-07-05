import argparse
import time 
import os 
import datetime

import torch 
import torch.distributed as dist 

from torchvision.models import resnet18

from tqdm import tqdm 
import ddp
import utils 
from data import create_dataset, create_sampler, create_loader

logger = utils.Logger()

def train(epoch, model, data_loader, optimizer, loss_function, args):

    model.train() 
    
    total = 0
    correct = 0
    
    for batch_idx, (images, labels) in enumerate((data_loader)):

        images = images.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        trained_samples=batch_idx * args.batch_size + len(images)
        total_samples=len(data_loader.sampler)
        
        lr = optimizer.param_groups[0]['lr']
        # print(f'Train Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {loss.item():0.4f}\tLearning_rate: {lr:0.6f}')

    train_acc = correct / total * 100

    if args.rank == 0:
        logger.write(f'Train Epoch: {epoch}\tAccuracy: {train_acc:0.2f}%, rank : {args.rank}')

    return train_acc 


@torch.no_grad()
def test(epoch, model, data_loader, loss_function, args):

    model.eval()

    test_loss = 0.0 
    total = 0
    correct = 0

    for batch_idx, (images, labels) in enumerate((data_loader)):

        images = images.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)
        
        outputs = model(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        
        trained_samples=batch_idx * args.batch_size + len(images)
        total_samples=len(data_loader.sampler)
        
        # print(f'Test Epoch: {epoch} [{trained_samples}/{total_samples}], Loss: {loss.item():0.4f}')

    test_acc = correct / total * 100

    if args.rank == 0:
        logger.write(f'Test Epoch: {epoch}\tAccuracy: {test_acc:0.2f}%, rank : {args.rank}')
    
    return test_acc

def main(args):
    ddp.init_distributed_mode(args)

    args.seed = args.seed + ddp.get_rank() 
    device = torch.device(args.device)
    utils.seed_everything(args.seed)

    train_dataset, val_dataset, test_dataset = create_dataset(args)

    if args.distributed:
        num_tasks = ddp.get_world_size()
        global_rank = ddp.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[args.batch_size]*3,
                                                          num_workers=[4,4,4],
                                                          is_trains=[True, False, False], 
                                                          collate_fns=[None,None,None])


    model = resnet18(weights=None, num_classes=10).to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = torch.optim.SGD(params=model.parameters(), lr = 0.1, weight_decay=5e-4)

    best_acc = 0 
    best_epoch = 0 

    print("Start training")
    start_time = time.time()

    for epoch in range(0, args.max_epoch):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            loss_function = torch.nn.CrossEntropyLoss()
            train(epoch, model, train_loader, optimizer, loss_function, args)

        if dist.is_initialized():
            dist.barrier()
        
        torch.cuda.empty_cache()

        if ddp.is_main_process():
            test_acc = test(epoch, model, test_loader, loss_function, args)
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                print(f"Best accuracy: {best_acc:0.2f}% at epoch {best_epoch}")


    print(f"Best accuracy: {best_acc:0.2f}% at epoch {best_epoch}")
    end = time.time()
    print(f"Training time: {end-start_time:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)

    parser.add_argument('--log_dir', type=str, default='/media/NAS/USERS/sangjunchung/MY_SPACE/practice/log')

    parser.add_argument("--dataset", type=str, default="cifar10")

    parser.add_argument("--evaluate", type=bool, default=False)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epoch", type=int, default=5)

    args = parser.parse_args()

    logger.open(os.path.join(args.log_dir, f"log.txt_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"))

    main(args)
