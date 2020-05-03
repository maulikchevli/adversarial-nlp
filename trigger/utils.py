'''
utils.py

author: @Eric-Wallace
'''
import heapq
from copy import deepcopy
import numpy
from operator import itemgetter

def hotflip_attack(averaged_grad, embedding_matrix, trigger_token_ids,
                   increase_loss=False, num_candidates=1):
    """
    This function takes in the model's average_grad over a batch of examples, the model's
    token embedding matrix, and the current trigger token IDs. It returns the top token
    candidates for each position.

    If increase_loss=True, then the attack reverses the sign of the gradient and tries to increase
    the loss (decrease the model's probability of the true class). For targeted attacks, you want
    to decrease the loss of the target class (increase_loss=False).
    """
    averaged_grad = averaged_grad.cpu()
    embedding_matrix = embedding_matrix.cpu()
    trigger_token_embeds = torch.nn.functional.embedding(torch.LongTensor(trigger_token_ids),
                                                         embedding_matrix).detach().unsqueeze(0)
    averaged_grad = averaged_grad.unsqueeze(0)

    gradient_dot_embedding_matrix = torch.einsum("bij,kj->bik",
                                                 (averaged_grad, embedding_matrix))        
    if not increase_loss:
        gradient_dot_embedding_matrix *= -1    # lower versus increase the class probability.
    if num_candidates > 1: # get top k options
        _, best_k_ids = torch.topk(gradient_dot_embedding_matrix, num_candidates, dim=2)
        return best_k_ids.detach().cpu().numpy()[0]
    _, best_at_each_step = gradient_dot_embedding_matrix.max(2)
    return best_at_each_step[0].detach().cpu().numpy()

def get_best_candidates(model, batch, trigger_token_ids, cand_trigger_token_ids, beam_size=1):
    """"
    Given the list of candidate trigger token ids (of number of trigger words by number of candidates
    per word), it finds the best new candidate trigger.
    This performs beam search in a left to right fashion.
    """
    # first round, no beams, just get the loss for each of the candidates in index 0.
    # (indices 1-end are just the old trigger)
    loss_per_candidate = get_loss_per_candidate(0, model, batch, trigger_token_ids,
                                                cand_trigger_token_ids)
    # maximize the loss
    top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))

    # top_candidates now contains beam_size trigger sequences, each with a different 0th token
    for idx in range(1, len(trigger_token_ids)): # for all trigger tokens, skipping the 0th (we did it above)
        loss_per_candidate = []
        for cand, _ in top_candidates: # for all the beams, try all the candidates at idx
            loss_per_candidate.extend(get_loss_per_candidate(idx, model, batch, cand,
                                                             cand_trigger_token_ids))
        top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))
    return max(top_candidates, key=itemgetter(1))[0]

def get_loss_per_candidate(index, model, batch, trigger_token_ids, cand_trigger_token_ids):
    """
    For a particular index, the function tries all of the candidate tokens for that index.
    The function returns a list containing the candidate triggers it tried, along with their loss.
    """
    if isinstance(cand_trigger_token_ids[0], (numpy.int64, int)):
        print("Only 1 candidate for index detected, not searching")
        return trigger_token_ids

    loss_per_candidate = []
    # loss for the trigger without trying the candidates
    curr_loss = evaluate_batch_trigger(model, batch, criterion, trigger_token_ids)['loss'] \
                    .cpu().detach().numpy()

    loss_per_candidate.append((deepcopy(trigger_token_ids), curr_loss))
    for cand_id in range(len(cand_trigger_token_ids[0])):
        trigger_token_ids_one_replaced = deepcopy(trigger_token_ids) # copy trigger
        trigger_token_ids_one_replaced[index] = cand_trigger_token_ids[index][cand_id] # replace one token

        loss = evaluate_batch_trigger(model, batch, criterion, trigger_token_ids_one_replaced)['loss'].cpu().detach().numpy()
        loss_per_candidate.append((deepcopy(trigger_token_ids_one_replaced), loss))

    return loss_per_candidate
