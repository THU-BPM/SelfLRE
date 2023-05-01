import sys
from networks import RelationClassification, LabelGeneration, MLP
from transformers import AdamW
import torch.optim as optim
from transformers import BertTokenizer
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import random
import numpy as np
import os
import copy
from tqdm import tqdm
import time, json
import datetime
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split, Subset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.autograd import Variable
from sklearn.metrics import f1_score
from collections import Counter
from mask_words_predict import get_enhance_result
import cbert_finetune
import argparse
import logging
# ------------------------init parameters----------------------------
parser = argparse.ArgumentParser(description='Pytorch For SelfLRE')
parser.add_argument('--cuda', type=str, default="0,1",help='appoint GPU devices')
parser.add_argument('--dataset', type=str, default="SemEval",help='dataset name')
parser.add_argument('--num_labels', type=int, default=19, help='num labels of the dataset')
parser.add_argument('--max_length', type=int, default=128, help='max token length of the sentence for bert tokenizer')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--initial_lr', type=float, default=7e-5, help='initial learning rate')
parser.add_argument('--initial_eps', type=float, default=1e-8, help='initial adam_epsilon')
parser.add_argument('--epochs', type=int, default=10, help='training epochs for labeled data')
parser.add_argument('--seed_val', type=int, default=42, help='initial random seed value')
parser.add_argument('--unlabel_of_train', type=float, default=0.5, help='unlabeled data percent of the dataset')
parser.add_argument('--label_of_train', type=float, default=0.10, help='labeled data percent of the dataset')
parser.add_argument('--use_aug', type=bool, default=True, help='whether to use data aug')
parser.add_argument('--lambda_ctr',type=float, default = 0.1, help = 'scalar hyperparameter to control ctr loss')
parser.add_argument('--lambda_sim',type=float, default = 0.1, help = 'scalar hyperparameter to control sim loss')
parser.add_argument('--T',type=float, default =0.2, help = 'threshold T in similar Matrix')
parser.add_argument('--tau',type=float, default =0.95, help = 'temperature coefficient')


args = parser.parse_args()

LOG_DIR = args.dataset + '_' + str(int(args.label_of_train * 100)) + '_exploss'
os.system('mkdir ' + LOG_DIR)

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
device = torch.device("cuda")

# ------------------------functions----------------------------

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    non_zero_idx = (labels_flat != 0)
    # if len(labels_flat[non_zero_idx])==0:
    #     print("error occur: ", labels_flat)
    #     return 0
    return np.sum(pred_flat[non_zero_idx] == labels_flat[non_zero_idx]) / len(labels_flat[non_zero_idx])


# Takes a time in seconds and returns a string hh:mm:ss
def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# cited: https://github.com/INK-USC/DualRE/blob/master/utils/scorer.py#L26
def score(key, prediction, verbose=True, NO_RELATION=0):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]

        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print the aggregate score
    if verbose:
        print("Final Score:")
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(
            sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(
            sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    print("SET NO_RELATION ID: ", NO_RELATION)
    print("Precision (micro): {:.3%}".format(prec_micro))
    print("   Recall (micro): {:.3%}".format(recall_micro))
    print("       F1 (micro): {:.3%}".format(f1_micro))
    return prec_micro, recall_micro, f1_micro

# ------------------------prepare sentences----------------------------

# Tokenize all of the sentences and map the tokens to thier word IDs.
def pre_processing(sentence_train, sentence_train_label,mode='labeled',sentence_org=None,sentence_org_label=None):
    input_ids = []
    attention_masks = []
    labels = []
    e1_pos = []
    e2_pos = []
    # index_list = []

    # Load tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    # tokenizer.add_special_tokens({'additional_special_tokens':["<e1>","</e1>","<e2>","</e2>"]})

    # pre-processing sentenses to BERT pattern
    for i in range(len(sentence_train)):
        encoded_dict = tokenizer.encode_plus(
            sentence_train[i],  # Sentence to encode.
            add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
            max_length=args.max_length,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
            truncation=True
        )
        try:
            # Find e1(id:2487) and e2(id:2475) position
            pos1 = (encoded_dict['input_ids'] == 2487).nonzero()[0][1].item()
            pos2 = (encoded_dict['input_ids'] == 2475).nonzero()[0][1].item()
            e1_pos.append(pos1)
            e2_pos.append(pos2)
            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])
            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])
            labels.append(sentence_train_label[i])
            # index_list.append(i)
        except:
            pass
            #print(sent)

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0).to(device)
    attention_masks = torch.cat(attention_masks, dim=0).to(device)
    labels = torch.tensor(labels, device='cuda')
    e1_pos = torch.tensor(e1_pos, device='cuda')
    e2_pos = torch.tensor(e2_pos, device='cuda')
    # index_list = torch.tensor(index_list, device='cuda')
    # w = torch.ones(len(e1_pos), device='cuda')
    if mode == 'unlabeled' :
        input_ids_org = []
        attention_masks_org = []
        labels_org = []
        e1_pos_org = []
        e2_pos_org = []
        # index_list = []

        # Load tokenizer.
        print('Loading BERT tokenizer...')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        # tokenizer.add_special_tokens({'additional_special_tokens':["<e1>","</e1>","<e2>","</e2>"]})

        # pre-processing sentenses to BERT pattern
        for i in range(len(sentence_train)):
            encoded_dict = tokenizer.encode_plus(
                sentence_org[i],  # Sentence to encode.
                add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
                max_length=args.max_length,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
                truncation=True
            )
            try:
                # Find e1(id:2487) and e2(id:2475) position
                pos1 = (encoded_dict['input_ids'] == 2487).nonzero()[0][1].item()
                pos2 = (encoded_dict['input_ids'] == 2475).nonzero()[0][1].item()
                e1_pos_org.append(pos1)
                e2_pos_org.append(pos2)
                # Add the encoded sentence to the list.
                input_ids_org.append(encoded_dict['input_ids'])
                # And its attention mask (simply differentiates padding from non-padding).
                attention_masks_org.append(encoded_dict['attention_mask'])
                labels_org.append(sentence_org_label[i])
                # index_list.append(i)
            except:
                pass
                #print(sent)

        # Convert the lists into tensors.
        input_ids_org = torch.cat(input_ids_org, dim=0).to(device)
        attention_masks_org = torch.cat(attention_masks_org, dim=0).to(device)
        labels_org = torch.tensor(labels_org, device='cuda')
        e1_pos_org = torch.tensor(e1_pos_org, device='cuda')
        e2_pos_org = torch.tensor(e2_pos_org, device='cuda')
        train_dataset = TensorDataset(input_ids, attention_masks, labels, e1_pos, e2_pos,input_ids_org, attention_masks_org, labels_org, e1_pos_org, e2_pos_org)
        return train_dataset
    # Combine the training inputs into a TensorDataset.
    # train_dataset = TensorDataset(input_ids, attention_masks, labels, e1_pos, e2_pos, w)
    train_dataset = TensorDataset(input_ids, attention_masks, labels, e1_pos, e2_pos)

    return train_dataset


def stratified_sample(dataset, ratio):
    data_dict = {}
    for i in range(len(dataset)):
        if not data_dict.get(dataset[i][2].item()):
            data_dict[dataset[i][2].item()] = []
        data_dict[dataset[i][2].item()].append(i)
    sampled_indices = []
    rest_indices = []
    for indices in data_dict.values():
        random.shuffle(indices)
        sampled_indices += indices[0:int(len(indices) * ratio)]
        rest_indices += indices[int(len(indices) * ratio):int(len(indices) * (ratio + args.unlabel_of_train))]
    # print("****************************************************")
    # print(sampled_indices)
    # print("****************************************************")
    return [Subset(dataset, sampled_indices), Subset(dataset, rest_indices), sampled_indices, rest_indices]


def add_space(sentence_train, dataset_name):
    if dataset_name == "tacred":
        return sentence_train
    else:
        sentence_train_new = []
        for sentence in sentence_train:
            sentence_train_new.append(sentence.replace("<e1>","<e1> ").replace("</e1>"," </e1>").replace("<e2>","<e2> ").replace("</e2>"," </e2>"))
        return sentence_train_new


def aug_data(sentence_train, sentence_train_label, b_index_list, masked_model):
    batch_aug_texts = []
    batch_aug_texts_ids = []
    batch_org_texts = []
    batch_org_texts_ids = []
    for i in range(len(b_index_list)):
        batch_org_texts.append(sentence_train[b_index_list[i]])
        batch_org_texts_ids.append(sentence_train_label[b_index_list[i]])
    
    return batch_org_texts,batch_org_texts_ids,batch_org_texts,batch_org_texts_ids
    relation2id = json.load(open('../data/' + args.dataset + '/relation2id.json', 'r'))
    for i in tqdm(range(len(b_index_list))):
        enhance_result = []
        cls_dist = 0.0
        enhance_result, cls_dist = get_enhance_result(sentence_train[b_index_list[i]], masked_model)
        aug_text = " ".join(enhance_result).replace(" ##", "")
        batch_aug_texts.append(aug_text)
        batch_aug_texts_ids.append(sentence_train_label[b_index_list[i]])
    print(batch_aug_texts[:10],batch_org_texts[:10])
    return batch_aug_texts, batch_aug_texts_ids ,batch_org_texts,batch_org_texts_ids


def compute_loss_ctr(un_e, un_e_org, tau):

    
    # Compute exponent matrix
    #exp_matrix = construct_W_e(un_e,un_e_org,tau)
    exp_matrix = torch.exp(torch.matmul(un_e_org, un_e_org.t()) / tau)
    
    # Compute exponent sum for each row
    exp_sum = torch.sum(exp_matrix, dim=1)
    
    
    # Compute loss for each sample
    loss_samples = -torch.log(torch.exp(torch.diag(torch.matmul(un_e, un_e_org.t())) / tau) / exp_sum)
    
    # Compute overall loss by taking the mean
    loss_ctr = torch.mean(loss_samples)
    
    return loss_ctr

def construct_W_p(un_logits,T=0.20):
    # (B,labels)
    W_p = torch.matmul(un_logits, un_logits.t())
    W_p.fill_diagonal_(1)
    W_p[W_p < T] = 0 
    return W_p

def construct_W_e(un_e, un_e_org,tau=1):
    bsz = un_e.shape[0]
    diag_indices = torch.arange(bsz)

    diagonal_elements = torch.exp(torch.sum(un_e * un_e_org, dim=1) / tau)
    dot_product = torch.matmul(un_e_org, un_e_org.t())

    off_diagonal_elements = torch.exp(dot_product.clone() / tau).clone()


    W_e_temp = torch.zeros((bsz, bsz)).to(device)
    W_e_temp[diag_indices, diag_indices] = diagonal_elements[diag_indices]


    mask = torch.ones_like(W_e_temp).to(device) - torch.eye(W_e_temp.size(0)).to(device)

    W_e = off_diagonal_elements * mask + W_e_temp * (1 - mask)


    return W_e.to(device)

# ------------------------training----------------------------

def main(argv=None):
    # Load the dataset.
    sentence_train = json.load(open('../data/' + args.dataset + '/train_sentence.json', 'r'))
    sentence_train_label = json.load(open('../data/' + args.dataset + '/train_label_id.json', 'r'))
    sentence_train = add_space(sentence_train, args.dataset)

    train_dataset = pre_processing(sentence_train, sentence_train_label)

    # define the loss function 
    criterion = nn.CrossEntropyLoss()

    # split training data to labeled set and unlabeled set
    # labeled_dataset, unlabeled_dataset_total = random_split(train_dataset, [int(LABEL_OF_TRAIN * len(train_dataset)),
    #                                                                         len(train_dataset) -
    #                                                                         int(LABEL_OF_TRAIN * len(train_dataset))])
    labeled_dataset, unlabeled_dataset_total, labeled_indices, unlabeled_indices = stratified_sample(train_dataset, args.label_of_train)
    print(len(train_dataset),len(labeled_dataset),len(unlabeled_dataset_total))

    if args.use_aug:
        print("##################################")
        print("fine tune MLM model with labeled data")
        print("##################################")
        relation2id = json.load(open('../data/' + args.dataset + '/relation2id.json', 'r'))
        train_examples = []
        for index in labeled_indices:
            relation = list(relation2id.keys())[list(relation2id.values()).index(sentence_train_label[index])]
            sentence_relation = sentence_train[index].replace("[CLS]", "[CLS] " + relation + " [SEP]")
            train_examples.append(sentence_relation)
        #masked_model = cbert_finetune.train_mlm(train_examples)

        print("##################################")
        print("perform augmentation for unlabeled data...")
        print("##################################")
        unlabeled_aug, unlabeled_aug_labels,unlabeled_org,unlabeled_org_labels = aug_data(sentence_train, sentence_train_label, unlabeled_indices, None)
        unlabeled_dataset = pre_processing(unlabeled_aug,unlabeled_aug_labels,mode="unlabeled",sentence_org=unlabeled_org,sentence_org_label=unlabeled_org_labels)



    # Create the DataLoaders for our label and unlabel sets.
    labeled_dataloader = DataLoader(
        labeled_dataset,  # The training samples.
        sampler=RandomSampler(labeled_dataset),  # Select batches randomly
        batch_size=args.batch_size  # Trains with this batch size.
    )
    mu = int((1-args.label_of_train)/args.label_of_train*args.unlabel_of_train) 
    unlabeled_dataloader = DataLoader(
        unlabeled_dataset,  # The training samples.
        sampler=RandomSampler(unlabeled_dataset),  # Select batches randomly
        batch_size=args.batch_size*mu  # Trains with this batch size.
    )



    sentence_val = json.load(open('../data/' + args.dataset + '/test_sentence.json', 'r'))
    sentence_val_label = json.load(open('../data/' + args.dataset + '/test_label_id.json', 'r'))
    val_dataset = pre_processing(sentence_val, sentence_val_label)

    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=args.batch_size  # Evaluate with this batch size.
    )
    # Load models
    model_teacher = LabelGeneration.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=args.num_labels,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    model_teacher = nn.DataParallel(model_teacher)

    model_teacher = model_teacher.to(device)
    
    optimizer = AdamW(model_teacher.parameters(),
                       lr=args.initial_lr,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                       eps=args.initial_eps,  # args.adam_epsilon  - default is 1e-8.,
                       weight_decay = 0.0005
                       )
    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    total_steps1 = len(labeled_dataloader) * (args.epochs)
    

    # total_steps = len(labeled_dataloader) * EPOCHS + (1 * len(labeled_dataloader) + 1 * len(unlabeled_dataloader)) * TOTAL_EPOCHS
    total_steps = total_steps1 
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, 
                                                num_training_steps = total_steps)
    # Set the seed value all over the place to make this reproducible.
    random.seed(args.seed_val)
    np.random.seed(args.seed_val)
    torch.manual_seed(args.seed_val)
    torch.cuda.manual_seed_all(args.seed_val)


    # Measure the total training time for the whole run.
    total_t0 = time.time()

#
    for epoch_i in range(0, args.epochs):
        
        # ========================================
        #               Training
        # ========================================
        
        # Perform one full pass over the training set.
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.epochs))
        print('Training...')
        # Measure how long the training epoch takes.
        t0 = time.time()
        # Reset the total loss for this epoch.
        total_train_loss = 0
        # Put the model into training mode.
        model_teacher.train()

        # save mixup features
        all_ground_truth = np.array([])
        bar = tqdm(labeled_dataloader)
        # For each batch of training data...
        dl_u = iter(unlabeled_dataloader)
        for step, batch in enumerate(bar):
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(labeled_dataloader), elapsed))
            # Unpack this training batch from our dataloader.
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_e1_pos = batch[3].to(device)
            b_e2_pos = batch[4].to(device)
            # b_index_list = batch[5].to(device)

            model_teacher.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch)
            e,logits, _ = model_teacher(b_input_ids, 
                                    token_type_ids=None,
                                    attention_mask=b_input_mask,
                                    labels=b_labels,
                                    e1_pos=b_e1_pos,
                                    e2_pos=b_e2_pos)
            logits = logits[0]
            loss = criterion(logits.view(-1, args.num_labels), batch[2].view(-1))
            batch_unlabeled = next(dl_u)
            un_input_ids = batch_unlabeled[0].to(device)
            un_input_mask = batch_unlabeled[1].to(device)
            un_labels = batch_unlabeled[2].to(device)
            un_e1_pos = batch_unlabeled[3].to(device)
            un_e2_pos = batch_unlabeled[4].to(device)

            un_input_ids_org = batch_unlabeled[5].to(device)
            un_input_mask_org = batch_unlabeled[6].to(device)
            un_labels_org = batch_unlabeled[7].to(device)
            un_e1_pos_org = batch_unlabeled[8].to(device)
            un_e2_pos_org = batch_unlabeled[9].to(device)



            #model_teacher.zero_grad()

            un_e,un_logits, _ = model_teacher(un_input_ids, 
                        token_type_ids=None,
                        attention_mask=un_input_mask,
                        labels=un_labels,
                        e1_pos=un_e1_pos,
                        e2_pos=un_e2_pos)
            un_logits = un_logits[0]
            un_e_org,un_logits_org, _ = model_teacher(un_input_ids_org, 
                        token_type_ids=None,
                        attention_mask=un_input_mask_org,
                        labels=un_labels_org,
                        e1_pos=un_e1_pos_org,
                        e2_pos=un_e2_pos_org)
            un_logits_org = un_logits_org[0]
            with torch.no_grad():
                un_logits_org = torch.softmax(un_logits_org,dim=1)
                un_logits = torch.softmax(un_logits,dim=1)

            W_p = torch.mm(un_logits_org, un_logits_org.t())       
            W_p.fill_diagonal_(1)    
            pos_mask = (W_p>=args.T).float()
            W_p = W_p * pos_mask

            W_p = W_p / W_p.sum(1, keepdim=True)

            W_e = construct_W_e(un_e,un_e_org)
            W_e = W_e / W_e.sum(1, keepdim=True)

            loss_ctr = compute_loss_ctr(un_logits, un_logits_org, args.tau)


            loss_sim = - (torch.log(W_p + 1e-7) * W_e).sum(1)
            loss_sim = loss_sim.mean() 
            # loss_u = - torch.sum((F.log_softmax(logits_u_s0,dim=1) * probs),dim=1) * mask                
            # loss_u = loss_u.mean()
            train_loss = loss.mean() +args.lambda_ctr *loss_ctr + + args.lambda_sim * loss_sim
            bar.set_description(f"loss: {train_loss.item()} loss_ctr:{ loss_ctr.item() } loss_sim:{loss_sim.item()} loss_x:{loss.mean().item()}")

            train_loss.backward()
            optimizer.step()
            scheduler.step()
            # Perform a backward pass to calculate the gradients.
            

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            labels_flat = label_ids.flatten()
            all_ground_truth = np.concatenate((all_ground_truth, labels_flat), axis=None)




        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.
        print("")
        print("Running Validation...")
        t0 = time.time()
        # Put the model in evaluation mode
        model_teacher.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        all_prediction = np.array([])
        all_ground_truth = np.array([])
        
        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Unpack this training batch from our dataloader.
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_e1_pos = batch[3].to(device)
            b_e2_pos = batch[4].to(device)
            with torch.no_grad():
                (e,logits, _) = model_teacher(b_input_ids,
                                            token_type_ids=None,
                                            attention_mask=b_input_mask,
                                            labels=b_labels,
                                            e1_pos=b_e1_pos,
                                            e2_pos=b_e2_pos)
            logits = logits[0]
            loss = criterion(logits.view(-1, args.num_labels), batch[2].view(-1))
            # Accumulate the validation loss.
            total_eval_loss += loss.sum().item()
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            pred_flat = np.argmax(logits, axis=1).flatten()
            labels_flat = label_ids.flatten()
            all_prediction = np.concatenate((all_prediction, pred_flat), axis=None)
            all_ground_truth = np.concatenate((all_ground_truth, labels_flat), axis=None)

        
        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        print(f"avg_val_loss: {avg_val_loss} validation_time: {validation_time}")
        score(all_ground_truth, all_prediction)



if __name__ == "__main__":
    sys.exit(main())