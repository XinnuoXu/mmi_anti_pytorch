from __future__ import division
import torch
import numpy as np


class Beam(object):
    """
    Class for managing the internals of the beam search process.

    Takes care of beams, back pointers, and scores.

    Args:
       size (int): beam size
       pad, bos, eos (int): indices of padding, beginning, and ending.
       n_best (int): nbest size to use
       cuda (bool): use gpu
       global_scorer (:obj:`GlobalScorer`)
    """
    def __init__(self, size, pad, bos, eos,
                 n_best=1, cuda=False,
                 global_scorer=None,
                 min_length=0):

        self.size = size
        self.cuda = cuda
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []

        # MMI: The score for each translation on the beam
        self.seq_scores = self.tt.FloatTensor(size).zero_()
        self.lm_scores = self.tt.FloatTensor(size).zero_()
        self.all_seq_scores = []
        self.all_lm_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [self.tt.LongTensor(size)
                        .fill_(pad)]
        self.next_ys[0][0] = bos

        # Has EOS topped the beam yet.
        self._eos = eos
        self.eos_top = False

        # The attentions (matrix) for each time.
        self.attn = []

        # Time and k pair for finished.
        self.finished = []
        self.n_best = n_best

        # Information for global scoring.
        self.global_scorer = global_scorer
        self.global_state = {}

        # Minimum prediction length
        self.min_length = min_length

        # Ended sentences
        self.ended_sentences = []
        self.ended_sentences_attn = []
        self.ended_sentences_scores = []
        self.ended_sentences_seq_scores = []
        self.ended_sentences_lm_scores = []

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.next_ys[-1]

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    def get_current_seq(self):
        "Get the predicted string for the current timestep."
        res = []; res_torch = []

        if len(self.next_ys) == 1:
            return []

        for k in range(0, self.next_ys[-1].size(0)):
            hyp, attn = self.get_hyp(len(self.next_ys), k)
            res.append(hyp)
        mtx = np.array(res).transpose()

        if self.cuda:
            for i in range(0, mtx.shape[0]):
                res_torch.append(torch.from_numpy(mtx[i]).cuda())
        else:
            for i in range(0, mtx.shape[0]):
                res_torch.append(torch.from_numpy(mtx[i]))
        return res_torch

    def advance(self, word_probs, attn_out, mmi=False, seq_score = None, lm_score = None):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attn_out`: Compute and update the beam search.

        Parameters:

        * `word_probs`- probs of advancing from the last step (K x words)
        * `attn_out`- attention at the last step

        Returns: True if beam search is complete.
        """
        num_words = word_probs.size(1)

        # force the output to be longer than self.min_length
        cur_len = len(self.next_ys)
        if cur_len < self.min_length:
            for k in range(len(word_probs)):
                word_probs[k][self._eos] = -1e20

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_scores = word_probs + self.scores.unsqueeze(1).expand_as(word_probs)
            # Don't let EOS have children.
            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == self._eos:
                    beam_scores[i] = -1e20
        else:
            beam_scores = word_probs[0]
        # beam_score = beam_size * vocab_size
        flat_beam_scores = beam_scores.view(-1)

        if not mmi:
            best_scores, best_scores_id = flat_beam_scores.topk(self.size, 0, True, True)
        else:
            # MMI

            if len(self.prev_ks) > 0:
                beam_seq_score = seq_score + self.seq_scores.unsqueeze(1).expand_as(seq_score)
                beam_lm_score = lm_score + self.lm_scores.unsqueeze(1).expand_as(lm_score)
            else:
                beam_seq_score = seq_score[0]
                beam_lm_score = lm_score[0]
            flat_beam_seq_score = beam_seq_score.view(-1)
            flat_beam_lm_score = beam_lm_score.view(-1)
        
            best_scores_beam = []; best_scores_id_beam = []
            best_seq_scores_beam = []; best_lm_scores_beam = []
            best_scores, best_scores_id = flat_beam_scores.topk(self.size * self.size, 0, True, True)

            # Save all ended sentences
            # Get top_beam_size not ended sentences
            prev_k = best_scores_id / num_words
            next_ys = best_scores_id - prev_k * num_words
            for i in range(next_ys.size(0)):
                if next_ys[i] == self._eos:
                    # Save all ended sentences
                    hyp, attn = self.get_hyp(len(self.next_ys), prev_k[i])
                    if len(hyp) == 0: continue
                    score = self.global_scorer.score_iter(self, best_scores[i], prev_k[i])

                    score_seq = self.global_scorer.score_iter(self, flat_beam_seq_score[best_scores_id[i]], prev_k[i])
                    score_lm = self.global_scorer.score_iter(self, flat_beam_lm_score[best_scores_id[i]], prev_k[i])
        
                    self.ended_sentences.append(hyp)
                    self.ended_sentences_scores.append(score)
                    self.ended_sentences_seq_scores.append(score_seq)
                    self.ended_sentences_lm_scores.append(score_lm)
                    self.ended_sentences_attn.append(attn)
                elif len(best_scores_beam) < self.size:
                    # Get top_beam_size not ended sentences
                    best_scores_beam.append(best_scores[i])
                    best_scores_id_beam.append(best_scores_id[i])
                    best_seq_scores_beam.append(flat_beam_seq_score[best_scores_id[i]])
                    best_lm_scores_beam.append(flat_beam_lm_score[best_scores_id[i]])
                
                    # Debug
                    hyp, attn = self.get_hyp(len(self.next_ys), prev_k[i])
                    hyp.append(next_ys[i])
                    #print hyp, best_scores[i], flat_beam_seq_score[best_scores_id[i]], flat_beam_lm_score[best_scores_id[i]]

            best_scores = self.tt.FloatTensor(best_scores_beam)
            best_scores_id = self.tt.LongTensor(best_scores_id_beam)
            best_seq_scores = self.tt.FloatTensor(best_seq_scores_beam)
            best_lm_scores = self.tt.FloatTensor(best_lm_scores_beam)

            self.all_seq_scores.append(self.seq_scores)
            self.all_lm_scores.append(self.lm_scores)
            self.seq_scores = best_seq_scores
            self.lm_scores = best_lm_scores

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # best_scores_id is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = best_scores_id / num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append((best_scores_id - prev_k * num_words))
        self.attn.append(attn_out.index_select(0, prev_k))

        if self.global_scorer is not None:
            self.global_scorer.update_global_state(self)

        for i in range(self.next_ys[-1].size(0)):
            if self.next_ys[-1][i] == self._eos:
                s = self.scores[i]
                if self.global_scorer is not None:
                    global_scores = self.global_scorer.score(self, self.scores)
                    s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.next_ys[-1][0] == self._eos:
            # self.all_scores.append(self.scores)
            self.eos_top = True

    def done(self):
        return self.eos_top and len(self.finished) >= self.n_best

    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                s = self.scores[i]
                if self.global_scorer is not None:
                    global_scores = self.global_scorer.score(self, self.scores)
                    s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))
        # should be i += 1???

        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def get_hyp(self, timestep, k):
        """
        Walk back to construct the full hypothesis.
        """
        hyp, attn = [], []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hyp.append(self.next_ys[j+1][k])
            attn.append(self.attn[j][k])
            k = self.prev_ks[j][k]
        if len(hyp) > 0:
            return hyp[::-1], torch.stack(attn[::-1])
        else:
            return hyp, attn


class GNMTGlobalScorer(object):
    """
    NMT re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`

    Args:
       alpha (float): length parameter
       beta (float):  coverage parameter
    """
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def score(self, beam, logprobs):
        "Additional term add to log probability"
        cov = beam.global_state["coverage"]
        pen = self.beta * torch.min(cov, cov.clone().fill_(1.0)).log().sum(1)
        l_term = (((5 + len(beam.next_ys)) ** self.alpha) /
                  ((5 + 1) ** self.alpha))
        return (logprobs / l_term) + pen

    def score_iter(self, beam, logprobs, i):
        "Additional term add to log probability"
        cov = beam.global_state["coverage"]
        pen = self.beta * torch.min(cov, cov.clone().fill_(1.0)).log().sum(1)[i]
        l_term = (((5 + len(beam.next_ys)) ** self.alpha) /
                  ((5 + 1) ** self.alpha))
        return (logprobs / l_term) + pen

    def update_global_state(self, beam):
        "Keeps the coverage vector as sum of attens"
        if len(beam.prev_ks) == 1:
            beam.global_state["coverage"] = beam.attn[-1]
        else:
            beam.global_state["coverage"] = beam.global_state["coverage"] \
                .index_select(0, beam.prev_ks[-1]).add(beam.attn[-1])
