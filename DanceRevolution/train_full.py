import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from dataset import DanceDataset, paired_collate_fn
from model import Encoder, Decoder, Model
from utils.log import Logger
from utils.functional import str2bool, load_data, printer
import warnings
#warnings.filterwarnings('ignore')


def valid(model, criterion, validation_data, device):
    """ Epoch operation in validation phase """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch_i, batch in enumerate(validation_data):
            src_seq, src_pos, tgt_seq = map(lambda x: x.to(device), batch)
            gold_seq = tgt_seq[:, 1:]
            src_seq = src_seq[:, :-1]
            src_pos = src_pos[:, :-1]
            tgt_seq = tgt_seq[:, :-1]

            hidden, out_frame = model.init_decoder_hidden(tgt_seq.size(0))

            output = model(src_seq, src_pos, tgt_seq, hidden, out_frame, 3000)
            loss = criterion(output, gold_seq)

            total_loss += loss.item()

        loss = total_loss / len(validation_data)
        stats = {
            'valid_loss': loss
        }

        return stats


def train(model, training_data, validation_data, optimizer, device, args, log):
    """ Start training """
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    valid_losses = []
    updates = 0  # global step

    for epoch_i in range(1, args.epochs + 1):
        log.set_progress(epoch_i, len(training_data))
        model.train()
        # scheduler.step()

        # epoch_loss = 0
        for batch_i, batch in enumerate(training_data):
            # prepare data
            # Only training for hiphop part
            if batch[0] == "h":
                src_seq, src_pos, tgt_seq = map(lambda x: x.to(device), batch)
                gold_seq = tgt_seq[:, 1:]
                src_seq = src_seq[:, :-1]
                src_pos = src_pos[:, :-1]
                tgt_seq = tgt_seq[:, :-1]

                hidden, out_frame = model.init_decoder_hidden(tgt_seq.size(0))

                # forward
                optimizer.zero_grad()

                output = model(src_seq, src_pos, tgt_seq, hidden, out_frame, epoch_i)

                # backward
                loss = criterion(output, gold_seq)
                loss.backward()

            '''
            for name,para in model.named_parameters():
                print(name)
                if para.grad is not None:
                    print(torch.mean(para.grad))
            '''

                # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                # print(grad_norm)

                # update parameters
                optimizer.step()

                stats = {
                'updates': updates,
                'loss': loss.item()
            }
                log.update(stats)
                updates += 1

            # scheduler.step(loss)
            valid_stats = valid(model, criterion, validation_data, device)
            log.log_eval(valid_stats)
            valid_losses.append(valid_stats['valid_loss'])

            checkpoint = {
            'model': model.state_dict(),
            'args': args,
            'epoch': epoch_i
        }

            if epoch_i % args.save_per_epochs == 0 or epoch_i == 1:
                filename = os.path.join(args.output_dir, f'epoch_{epoch_i}.pt')
                torch.save(checkpoint, filename)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir', type=str, default='data/train_1min_2')
    parser.add_argument('--valid_dir', type=str, default='data/valid_1min_2')
    parser.add_argument('--data_type', type=str, default='2D', help='the type of pose data')
    parser.add_argument('--output_dir', metavar='PATH', default='checkpoints/')

    parser.add_argument('--save_per_epochs', type=int, metavar='N', default=500)
    parser.add_argument('--log_per_updates', type=int, metavar='N', default=1,
                        help='log model loss per x updates (mini-batches).')
    parser.add_argument('--seed', type=int, default=2020,
                        help='random seed for data shuffling, dropout, etc.')
    parser.add_argument('--tensorboard', action='store_false')

    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--frame_dim', type=int, default=438)
    parser.add_argument('--encoder_hidden_size', type=int, default=512)
    parser.add_argument('--pose_dim', type=int, default=50)
    parser.add_argument('--decoder_hidden_size', type=int, default=256)

    parser.add_argument('--seq_len', type=int, default=900)
    parser.add_argument('--max_seq_len', type=int, default=4500)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--window_size', type=int, default=100)

    parser.add_argument('--fixed_steps', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.1)

    parser.add_argument('--cuda', type=str2bool, nargs='?', metavar='BOOL',
                        const=True, default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')

    ##Add Parser for pre-existing weights
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='path to pretrained model')
  	
    return parser.parse_args()


def prepare_dataloader(music_data, dance_data, args):
    data_loader = torch.utils.data.DataLoader(
        DanceDataset(music_data, dance_data),
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=True,
        collate_fn=paired_collate_fn,
        pin_memory=True
     )

    return data_loader


def main():
    # print("starting main")
    args = get_args()
    print(args)

    # Initialize logger
    #print("starting log")
    global log
    log = Logger(args)
    #print("Initialized logger")

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Check cuda device
    device = torch.device('cuda' if args.cuda else 'cpu')
    print(device)

    # Loading data
    print("test1")
    train_music_data, train_dance_data, _ = load_data(
        args.train_dir, interval=args.seq_len, data_type=args.data_type)
    print("test2")
    training_data = prepare_dataloader(train_music_data, train_dance_data, args)
    print("loaded training data")
    valid_music_data, valid_dance_data, _ = load_data(
        args.valid_dir, interval=args.seq_len, data_type=args.data_type)
    validation_data = prepare_dataloader(valid_music_data, valid_dance_data, args)

    print("proceed to encoder")
    encoder = Encoder(args)
    decoder = Decoder(args)
    print("proceed to model")
    model = Model(encoder, decoder, args, device=device)

    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())

    # Data Parallel to use multi-gpu
    # model = nn.DataParallel(model).to(device)
    model = model.to(device)

    # optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.module.parameters()), lr=args.lr)
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr)
    # scheduler = ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.8, patience=30, verbose=True, min_lr=1e-06, eps=1e-07)
    # scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
    #                                       step_size=args.lr_step_size,
    #                                       gamma=0.5)
    # for name,para in model.named_parameters():
    #        print(name,para.size())

    if args.pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path)['model'])

    train(model, training_data, validation_data, optimizer, device, args, log)


if __name__ == '__main__':
    main()
