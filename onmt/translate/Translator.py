import torch
from torch.autograd import Variable

import onmt.translate.Beam
import onmt.io

import sys
sys.path.append("./lm/")
import model

class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
    """
    def __init__(self, model, lm_model, fields,
                 beam_size, n_best=1,
                 max_length=100,
                 global_scorer=None, copy_attn=False, cuda=False,
                 beam_trace=False, mmi=False, mmi_g = 0, mmi_lambda=0, mmi_gamma=0, min_length=0):
        self.model = model
        self.fields = fields
        self.n_best = n_best
        self.max_length = max_length
        self.global_scorer = global_scorer
        self.copy_attn = copy_attn
        self.beam_size = beam_size
        self.cuda = cuda
        self.min_length = min_length
	self.mmi = mmi
	self.mmi_g = mmi_g
	self.mmi_lambda = mmi_lambda
	self.mmi_gamma = mmi_gamma
	self.lm_model = lm_model
	self.tt = torch.cuda if self.cuda else torch

        # for debugging
        self.beam_accum = None
        if beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def id2w(self, pred):
	vocab = self.fields["tgt"].vocab
	tokens = []
	for tok in pred:
	    if tok < len(vocab):
		tokens.append(vocab.itos[tok])
	    else:
		tokens.append(src_vocab.itos[tok - len(vocab)])
	    if tokens[-1] == onmt.io.EOS_WORD:
		tokens = tokens[:-1]
		break
	return tokens

    def translate_batch(self, batch, data):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object


        Todo:
           Shouldn't need the original dataset.
        """

        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        beam_size = self.beam_size
        batch_size = batch.batch_size
        data_type = data.data_type
        vocab = self.fields["tgt"].vocab
        beam = [onmt.translate.Beam(beam_size, n_best=self.n_best,
                                    cuda=self.cuda,
                                    global_scorer=self.global_scorer,
                                    pad=vocab.stoi[onmt.io.PAD_WORD],
                                    eos=vocab.stoi[onmt.io.EOS_WORD],
                                    bos=vocab.stoi[onmt.io.BOS_WORD],
                                    min_length=self.min_length)
                for __ in range(batch_size)]

        # Help functions for working with beams and batches
        def var(a): return Variable(a, volatile=True)

        def rvar(a): return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # (1) Run the encoder on the src.
        src = onmt.io.make_features(batch, 'src', data_type)
        src_lengths = None
        if data_type == 'text':
            _, src_lengths = batch.src

        enc_states, context = self.model.encoder(src, src_lengths)
        dec_states = self.model.decoder.init_decoder_state(
                                        src, context, enc_states)

        if src_lengths is None:
            src_lengths = torch.Tensor(batch_size).type_as(context.data)\
                                                  .long()\
                                                  .fill_(context.size(0))

        # (2) Repeat src objects `beam_size` times.
        src_map = rvar(batch.src_map.data) \
            if data_type == 'text' and self.copy_attn else None
        context = rvar(context.data)
        context_lengths = src_lengths.repeat(beam_size)
        dec_states.repeat_beam_size_times(beam_size)

        # (3) run the decoder to generate sentences, using beam search.
        for i in range(self.max_length):
            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.get_current_state() for b in beam])
                      .t().contiguous().view(1, -1))

            # Turn any copied words to UNKs
            # 0 is unk
            if self.copy_attn:
                inp = inp.masked_fill(
                    inp.gt(len(self.fields["tgt"].vocab) - 1), 0)

            # Temporary kludge solution to handle changed dim expectation
            # in the decoder
            inp = inp.unsqueeze(2)

            # Run one step.
            dec_out, dec_states, attn = self.model.decoder(
                inp, context, dec_states, context_lengths=context_lengths)

            dec_out = dec_out.squeeze(0)
            # dec_out: beam x rnn_size

            # (b) Compute a vector of batch*beam word scores.
            if not self.copy_attn:
                out = self.model.generator.forward(dec_out).data
                out = unbottle(out)
                # beam x tgt_vocab
            else:
                out = self.model.generator.forward(dec_out,
                                                   attn["copy"].squeeze(0),
                                                   src_map)
                # beam x (tgt_vocab + extra_vocab)
                out = data.collapse_copy_scores(
                    unbottle(out.data),
                    batch, self.fields["tgt"].vocab, data.src_vocabs)
                # beam x tgt_vocab
                out = out.log()
	
	    if self.mmi and self.lm_model != None:
	    	# MMI
	        ntokens = len(self.fields["tgt"].vocab.itos)
		for j, b in enumerate(beam):
		    cur_seq_candidates = b.get_current_seq()
		    # cur_seq_candidates: time_step * beam_size
		    if len(cur_seq_candidates) > 0 and len(cur_seq_candidates) < self.mmi_g:
		    	tmp_data = Variable(torch.stack(cur_seq_candidates, 0))
			hidden = self.lm_model.init_hidden(tmp_data.size(1))
			output, hidden = self.lm_model(tmp_data, hidden)
			output_flat = torch.nn.functional.log_softmax(output.view(-1, ntokens), 0)
			mmi_lm_score = output_flat[-beam_size:]
		    else:
		        mmi_lm_score = Variable(self.tt.FloatTensor(beam_size, ntokens).zero_())
		    mmi_seq_score = Variable(out[:, j])
		    mmi_score = mmi_seq_score - self.mmi_lambda * mmi_lm_score
                    b.advance(mmi_score.data, unbottle(attn["std"]).data[:, j, :context_lengths[j]], 
			self.mmi, mmi_seq_score.data, mmi_lm_score.data)
                    dec_states.beam_update(j, b.get_current_origin(), beam_size)
	
	    else:
            	# (c) Advance each beam.
            	for j, b in enumerate(beam):
                    b.advance(out[:, j], unbottle(attn["std"]).data[:, j, :context_lengths[j]])
                    dec_states.beam_update(j, b.get_current_origin(), beam_size)

        # (4) Extract sentences from beam.
        ret = self._from_beam(beam)
        ret["gold_score"] = [0] * batch_size
        if "tgt" in batch.__dict__:
            ret["gold_score"] = self._run_target(batch, data)
        ret["batch"] = batch
        return ret

    def _from_beam(self, beam):
        ret = {"predictions": [],
               "scores": [],
               "attention": []}
        for b in beam:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn, score = [], [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
		score.append(scores[i])
	    if self.mmi:
		# MMI
		if len(b.ended_sentences) == 0:
		    for i, (times, k) in enumerate(ks[:n_best]):
			hyp, att = b.get_hyp(times, k)
			b.ended_sentences.append(hyp)
			b.ended_sentences_attn.append(att)
			b.ended_sentences_scores.append(scores[i])
		for i in range(0, len(b.ended_sentences)):
		    b.ended_sentences_scores[i] += self.mmi_gamma * len(b.ended_sentences[i])
		    #print "QQQQ\t" + " ".join(self.id2w(b.ended_sentences[i])), b.ended_sentences_scores[i], b.ended_sentences_seq_scores[i], b.ended_sentences_lm_scores[i]
		# Sort score
		best_scores, best_scores_id = self.tt.FloatTensor(b.ended_sentences_scores).topk(n_best, 0, True, True)
		# return n_best 
	        del hyps[:]; del attn[:]; del scores[:]
		for i in range(0, min(best_scores.size(0), n_best)):
		    id = best_scores_id[i]
		    hyps.append(b.ended_sentences[id])
		    attn.append(b.ended_sentences_attn[id])
		    scores.append(b.ended_sentences_scores[id])
		    
            ret["predictions"].append(hyps)
            ret["scores"].append(scores)
            ret["attention"].append(attn)
        return ret

    def _run_target(self, batch, data):
        data_type = data.data_type
        if data_type == 'text':
            _, src_lengths = batch.src
        else:
            src_lengths = None
        src = onmt.io.make_features(batch, 'src', data_type)
        tgt_in = onmt.io.make_features(batch, 'tgt')[:-1]

        #  (1) run the encoder on the src
        enc_states, context = self.model.encoder(src, src_lengths)
        dec_states = self.model.decoder.init_decoder_state(src,
                                                           context, enc_states)

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        tt = torch.cuda if self.cuda else torch
        gold_scores = tt.FloatTensor(batch.batch_size).fill_(0)
        dec_out, dec_states, attn = self.model.decoder(
            tgt_in, context, dec_states, context_lengths=src_lengths)

        tgt_pad = self.fields["tgt"].vocab.stoi[onmt.io.PAD_WORD]
        for dec, tgt in zip(dec_out, batch.tgt[1:].data):
            # Log prob of each word.
            out = self.model.generator.forward(dec)
            tgt = tgt.unsqueeze(1)
            scores = out.data.gather(1, tgt)
            scores.masked_fill_(tgt.eq(tgt_pad), 0)
            gold_scores += scores
        return gold_scores
