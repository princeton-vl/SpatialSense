import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import progressbar
from options import parse_args
from models.recurrent_phrase_encoder import RecurrentPhraseEncoder
from models.simple_language_only_model import SimpleLanguageOnlyModel
from dataloader import create_dataloader
import pickle
import os.path
import sys
import shutil
from util import accuracies, num_true_positives


def train(model, criterion, optimizer, loader, epoch, args):
    'train for one epoch'
    model.train()
    loss = 0.
    acc = 0.
    num_samples = 0

    bar = progressbar.ProgressBar(max_value=len(loader))
    for idx, data_batch in enumerate(loader):
        subj_batch_var = data_batch['subject']['embedding']
        obj_batch_var = data_batch['object']['embedding']
        label_batch_var = torch.squeeze(data_batch['label'])
        predi_batch_var = data_batch['predicate']
        if torch.cuda.is_available():
            subj_batch_var = subj_batch_var.cuda()
            obj_batch_var = obj_batch_var.cuda()
            label_batch_var = label_batch_var.cuda()
            predi_batch_var = predi_batch_var.cuda()

        output_var = model(subj_batch_var, obj_batch_var, predi_batch_var)
        loss_batch_var = criterion(output_var, label_batch_var)
  
        loss_batch = loss_batch_var.item()
        loss += (len(data_batch['label']) * loss_batch)
        acc += num_true_positives(output_var, label_batch_var)
        num_samples += len(data_batch['label'])

        optimizer.zero_grad()
        loss_batch_var.backward()
        optimizer.step()

        bar.update(idx)

    loss /= num_samples
    acc /= (num_samples / 100.)
  
    return loss, acc


def test(split, model, criterion, loader, epoch, args):
    'validate/test the model'
    model.eval()
    loss = 0.
    _ids = [] 
    predictions = []
 
    with torch.no_grad():
        bar = progressbar.ProgressBar(max_value=len(loader))
        for idx, data_batch in enumerate(loader):
            _ids.extend(data_batch['_id'])
            subj_batch_var = data_batch['subject']['embedding']
            obj_batch_var = data_batch['object']['embedding']
            predi_batch_var = data_batch['predicate']
            if torch.cuda.is_available():
                subj_batch_var = subj_batch_var.cuda()
                obj_batch_var = obj_batch_var.cuda()
                predi_batch_var = predi_batch_var.cuda()

            output_var = model(subj_batch_var, obj_batch_var, predi_batch_var)
            predictions.append(output_var.detach())

            if 'label' in data_batch:
                label_batch_var = torch.squeeze(data_batch['label'])
                if torch.cuda.is_available():
                    label_batch_var = label_batch_var.cuda()
                loss_batch_var = criterion(output_var, label_batch_var)
                loss_batch = loss_batch_var.item()
                loss += (len(data_batch['label']) * loss_batch)
  
            bar.update(idx)
 
    if loss != 0:
        loss /= len(_ids)
    predictions = [v.item() for v in torch.cat(predictions)]
    if split == 'valid':
        pred_file = os.path.join(args.log_dir, 'predictions/pred_%02d.pickle' % epoch)
    else:
        pred_file = os.path.join(args.log_dir, 'predictions/pred_test.pickle')
    pickle.dump((_ids, predictions), open(pred_file, 'wb'))
    accs = accuracies(pred_file, args.datapath, split, False, args)
    return loss, accs


def main():
    args = parse_args()

    dataloader_train = create_dataloader(args.train_split, False, args)
    dataloader_valid = create_dataloader('valid', False, args)
    dataloader_test = create_dataloader('test', False, args)
    print('%d batches of training examples' % len(dataloader_train))
    print('%d batches of validation examples' % len(dataloader_valid))
    print('%d batches of testing examples' % len(dataloader_test))

    phrase_encoder = RecurrentPhraseEncoder(300, 300)
    model = SimpleLanguageOnlyModel(phrase_encoder, args.feature_dim, 9)
    criterion = nn.BCEWithLogitsLoss()
    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.l2)
    if args.train_split == 'train_valid':
      scheduler = StepLR(optimizer, step_size=8, gamma=0.1)
    else:
      scheduler = ReduceLROnPlateau(optimizer, patience=5, verbose=True)

    best_acc = -1.

    for epoch in range(args.n_epochs):
        print('epoch #%d' % epoch)
   
        print('training..')
        loss, acc = train(model, criterion, optimizer, dataloader_train, epoch, args)
        print('\n\ttraining loss = %.4f' % loss)
        print('\ttraining accuracy = %.3f' % acc)

        print('validating..')
        loss, accs = test('valid', model, criterion, dataloader_valid, epoch, args)
        print('\n\tvalidation loss = %.4f' % loss)
        print('\tvalidation accuracy = %.3f' % accs['overall'])
        for predi in accs:
            if predi != 'overall':
                print('\t\t%s: %.3f' % (predi, accs[predi]))

        checkpoint_filename = os.path.join(args.log_dir, 'checkpoints/model_%02d.pth' % epoch)
        model.cpu()
        torch.save({'epoch': epoch + 1,
                    'args': args,
                    'state_dict': model.state_dict(),
                    'accuracy': acc,
                    'optimizer' : optimizer.state_dict(),
                   }, checkpoint_filename)
        if torch.cuda.is_available():
            model.cuda()
        if best_acc < acc:
            best_acc = acc
            shutil.copyfile(checkpoint_filename, os.path.join(args.log_dir, 'checkpoints/model_best.pth'))
            shutil.copyfile(os.path.join(args.log_dir, 'predictions/pred_%02d.pickle' % epoch), 
                            os.path.join(args.log_dir, 'predictions/pred_best.pickle'))

        if args.train_split == 'train_valid':
            scheduler.step()
        else:
            scheduler.step(loss)

    print('testing..')
    loss, accs = test('test', model, criterion, dataloader_test, epoch, args)
    print('\n\ttesting loss = %.4f' % loss)
    print('\ttesting accuracy = %.3f' % accs['overall'])
    for predi in accs:
        if predi != 'overall':
            print('\t\t%s: %.3f' % (predi, accs[predi]))


if __name__ == '__main__':
    main()
