import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from torchtext import data
from torchtext import datasets

from code import interact

import random
import string

MT = False
PRINT = False
USE_RL_ENC = False
USE_RL_DEC = False

class KimPNDataset(datasets.TranslationDataset):

    def __init__(self, fields, num_tokens=20, max_depth=4, max_arity=4):

        def generate():
            def build_subtrees(depth=0):
                if depth == max_depth or depth > 1 and random.random() > 1 / (
                        max_depth + 1):
                    return ((string.ascii_lowercase[random.randint(
                        0, num_tokens - 1)],) * 2, 0)
                else:
                    subs, depths = zip(*[build_subtrees(depth + 1) for i in
                                         range(random.randint(2, max_arity))])
                    psubs, isubs = zip(*subs)
                    op = random.choice(('+', '*'))
                    return (('(' + op + ''.join(psubs) + ')',
                             '(' + op.join(isubs) + ')'), max(depths) + 1)
            while True:
                depth = random.randint(2, max_depth)
                this_depth = 0
                while this_depth != depth:
                    (prefix, infix), this_depth = build_subtrees()
                yield data.Example.fromlist([
                    list(reversed(prefix)),
                    list(reversed(infix))], fields)
                # print(''.join(self.examples[-1].src), '=>',
                #       ''.join(self.examples[-1].trg))

        super(datasets.TranslationDataset, self).__init__(generate(), fields)


class OneHot(nn.Module):

    def __init__(self, size):
        super().__init__()
        self.register_buffer('eye',
                             Variable(torch.eye(size), requires_grad=False))

    def forward(self, x):
        return self.eye.index_select(0, x.view(-1)).view(*x.size(), -1)


class S2S(nn.Module):

    def __init__(self, wv_size, hidden_size, num_layers, vocab):
        super().__init__()
        # wv_size = len(vocab)
        # self.embed = OneHot(len(vocab))
        self.embed = nn.Embedding(len(vocab), wv_size)
        self.encoder = nn.LSTM(wv_size, hidden_size, num_layers, persistent=True)
        self.decoder = nn.LSTM(wv_size, hidden_size, num_layers, persistent=True)
        self.out = nn.Linear(hidden_size, len(vocab))
        self.criterion = nn.CrossEntropyLoss()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, batch):
        src_shape, trg_shape = batch.src[1:].size(), batch.trg[1:].size()
        src = self.embed(batch.src[1:].view(-1)).view(*src_shape, -1)
        trg = self.embed(batch.trg[:-1].view(-1)).view(*trg_shape, -1)
        _, encoding = self.encoder(src)
        o, _ = self.decoder(trg, encoding)
        y = self.out(o.view(-1, self.hidden_size))
        t = batch.trg[1:].view(-1)
        print(y.max(1)[1].data[:, 0].tolist(), t.data.tolist())
        return self.criterion(y, t) * src.size(0), src.size(0)


class T2T(nn.Module):

    def __init__(self, size, vocab):
        super().__init__()
        embed = nn.Embedding(len(vocab), size)
        self.encoder = RNNG(size, embed, vocab)
        self.decoder = RNNG(size, embed, vocab, decoder=True,
                            attention=True, use_buffer=False)

        self.itos = vocab.itos
        self.enc_baseline_losses = dict()
        self.dec_baseline_losses = dict()
        self.enc_treewards_baseline = None
        self.dec_treewards_baseline = None
        self.coverage_baseline = None
        self.decay = 0.95
        self.register_buffer('gamma', torch.Tensor([0.95]))

    def forward(self, batch):
        # print(
        #     ''.join(reversed([
        #         self.itos[tok] for tok in batch.src.data[:, 0].tolist()])) +
        #     '=>' +
        #     ''.join(reversed([
        #         self.itos[tok] for tok in batch.trg.data[:, 0].tolist()])))
        encoding, enc_actions, enc_treewards, enc_losses = self.encoder(
            batch.src)
        best_loss = 100000
        for i in range(1):
            _, this_dec_actions, this_dec_treewards, this_losses = self.decoder(
                batch.trg, encoding)
            this_loss = sum(this_losses).data.mean() / sum(
                1 if l is not 0 else 0 for l in this_losses)
            if this_loss < best_loss:
                dec_actions, dec_treewards = this_dec_actions, this_dec_treewards
                dec_losses = this_losses

        enc_treewards, _ = enc_treewards
        dec_treewards, alphas = dec_treewards
        if len(alphas) == 0 or alphas[0].size(0) < 2:  # < 3:
            coverage = -100
        else:
            alphas = torch.stack(alphas, 0)
            max_0 = ((alphas[:, :-1].max(0)[0] - 1) ** 2).squeeze(0).mean(0)
            max_1 = ((alphas[:, :-1].max(1)[0] - 1) ** 2).squeeze(0).mean(0)
            sum_alphas = alphas.sum(0).squeeze(0)
            coverage = -torch.cat(((1 - sum_alphas[:-1])**2, sum_alphas[-1:]**2), 0).mean(0) - max_0 - max_1
            # coverage = -(1 - sum(alphas)[:-2]).abs().clamp(max=1).mean(0)
        if PRINT:
            print(*('{:0.3f}'.format(loss.data[0])
                    for loss in enc_losses + dec_losses if loss is not 0))
            print('coverage', coverage.data.tolist() if type(coverage) not in (
                float, int) else coverage)
            # print(enc_treewards.tolist(), dec_treewards.tolist())

        #if USE_RL_ENC or USE_RL_DEC:
        #    coverage_rebased = coverage if type(coverage) in (float, int) else coverage.data
        #    if self.coverage_baseline is None:
        #        self.coverage_baseline = coverage_rebased
        #    else:
        #        self.coverage_baseline = (coverage_rebased * (1 - self.decay) +
        #                                  self.coverage_baseline * self.decay)
        #    coverage_rebased = coverage_rebased - self.coverage_baseline
        if USE_RL_ENC:
            enc_num_shifts = sum((a.data[0] == 1).float() for a in enc_actions)
            enc_treewards_mean = sum(enc_treewards).mean() / enc_num_shifts
            if self.enc_treewards_baseline is None:
                self.enc_treewards_baseline = enc_treewards_mean
            else:
                self.enc_treewards_baseline = (
                    enc_treewards_mean * (1 - self.decay) +
                    self.enc_treewards_baseline * self.decay)
            enc_treewards_baseline = (
                self.enc_treewards_baseline * enc_num_shifts / len(enc_treewards))
            # enc_treewards = [(eu - enc_treewards_baseline) / 10 if sum(enc_treewards).mean() > 10 else eu * 0 for eu in enc_treewards]
            enc_treewards = [
                eu - enc_treewards_baseline for eu in enc_treewards]
            enc_losses_rebased = []
            for loss in enc_losses:
                if loss is 0:
                    loss_rebased = 0
                else:
                    loss_rebased = loss.data.clone()
                    for i, (this_loss, token) in enumerate(
                            zip(loss.data.tolist(), loss.tokens.tolist())):
                        if token not in self.enc_baseline_losses:
                            self.enc_baseline_losses[token] = this_loss
                        else:
                            self.enc_baseline_losses[token] = (
                                this_loss * (1 - self.decay) +
                                self.enc_baseline_losses[token] * self.decay)
                        loss_rebased[i] -= self.enc_baseline_losses[token]
                enc_losses_rebased.append(loss_rebased)
        if USE_RL_DEC:
            dec_num_shifts = sum((a.data[0] == 1).float() for a in dec_actions)
            dec_treewards_mean = sum(dec_treewards).mean() / dec_num_shifts
            if self.dec_treewards_baseline is None:
                self.dec_treewards_baseline = dec_treewards_mean
            else:
                self.dec_treewards_baseline = (
                    dec_treewards_mean * (1 - self.decay) +
                    self.dec_treewards_baseline * self.decay)
            dec_treewards_baseline = (
                self.dec_treewards_baseline * dec_num_shifts / len(dec_treewards))
            # dec_treewards = [(du - dec_treewards_baseline) / 10 if sum(enc_treewards).mean() > 10 else du * 0 for du in dec_treewards]
            dec_treewards = [
                du - dec_treewards_baseline for du in dec_treewards]
            dec_losses_rebased = []
            for loss in dec_losses:
                if loss is 0:
                    loss_rebased = 0
                else:
                    loss_rebased = loss.data.clone()
                    for i, (this_loss, token) in enumerate(
                            zip(loss.data.tolist(), loss.tokens.tolist())):
                        if token not in self.dec_baseline_losses:
                            self.dec_baseline_losses[token] = this_loss
                        else:
                            self.dec_baseline_losses[token] = (
                                this_loss * (1 - self.decay) +
                                self.dec_baseline_losses[token] * self.decay)
                        loss_rebased[i] -= self.dec_baseline_losses[token]
                dec_losses_rebased.append(loss_rebased)
            # print(sum(losses_rebased).tolist())
            # print(sum(dec_treewards).tolist())
            # reward = -(sum(losses_rebased) + sum(dec_treewards)) / num_shifts
        if USE_RL_ENC and USE_RL_DEC:
            loss_reward = -sum(dec_losses_rebased) / dec_num_shifts
            # self.baseline = reward * (1 - self.decay) + self.baseline * self.decay
            # reward = (reward - self.baseline) / (self.baseline * 100).abs()
            if PRINT:
                print('loss reward', loss_reward.tolist())
        if USE_RL_ENC or USE_RL_DEC:
            coverage_reward = coverage / 10 if type(coverage) in (float, int) else coverage.data / 10#coverage_rebased
            #if PRINT:
            #    print('coverage reward', coverage_reward.tolist())
            # reward = 0
        if USE_RL_ENC:
            for i, action in enumerate(enc_actions):
                enc_reward, num_losses = 0, 0
                for j in range(i, len(enc_actions)):
                    enc_reward += (
                        (enc_treewards[j] - enc_losses_rebased[j]) *
                        self.gamma ** (num_losses))
                    num_losses = num_losses + \
                        (enc_actions[j].data == 1).float()
                enc_reward += loss_reward + coverage_reward
                if PRINT:
                    print('{:0.3f}'.format(enc_reward[0] / 50), end=' ')
                action.reinforce(enc_reward.unsqueeze(1) / 50)
            if PRINT:
                print()
        if USE_RL_DEC:
            for i, action in enumerate(dec_actions):
                dec_reward, num_losses = 0, 0
                for j in range(i, len(dec_actions)):
                    dec_reward += (
                        (dec_treewards[j] - dec_losses_rebased[j]) *
                        self.gamma ** (num_losses))
                    num_losses = num_losses + \
                        (dec_actions[j].data == 1).float()
                dec_reward += coverage_reward
                if PRINT:
                    print('{:0.3f}'.format(dec_reward[0] / 50), end=' ')
                action.reinforce(dec_reward.unsqueeze(1) / 50)
        loss = sum(enc_losses)
        if loss is not 0:
            loss = loss.sum()
        loss += sum(dec_losses).sum()
        loss *= 10
        if PRINT:
            print()
        if USE_RL_ENC:
            loss += 0 * sum(enc_actions).sum().float()
        if USE_RL_DEC:
            loss += 0 * sum(dec_actions).sum().float()
        return loss - coverage * 10, sum(dec_losses).sum(), (batch.trg.size(0) - 2)
        # for action in enc_actions:
        #     action.reinforce(len(actions))
        # print(len(actions))
        # return 0 * sum(actions).sum(), batch.src.size(0)
        #loss = self.decoder(encoding, batch.trg)
        # return loss
        # return encoding.norm(), batch.src.size(0)


def get(x, ixs):
    # print('x', x)
    if isinstance(ixs, list):
        ixs = torch.LongTensor(ixs).cuda()
    # print('ixs', ixs)
    ixflat = torch.LongTensor(
        [i for i in range(x.size(1))]).cuda() + ixs * x.size(1)
    # print('ixflat', ixflat)
    ret = x.view(-1, *x.size()[2:]).index_select(0, Variable(ixflat))
    # print('ret', ret)
    return ret


class Node:

    def __init__(self, x, hc, first=None, orig_open=None, char=False):
        self.x = x
        self.hc = hc
        self.first = first
        self.orig_open = orig_open
        self.char = char
        if first is not None:
            first.nodes.append(self)


class First:

    def __init__(self, x, hc, parent=None):
        self.x = x
        self.hc = hc
        self.parent = parent
        self.first = self
        self.nodes = []


class StackLSTM(nn.Module):

    def __init__(self, size, vocab, attention=False):
        super().__init__()
        self.stack_rnn = nn.LSTMCell(size, size)
        self.compose_rnn = nn.LSTM(size, size, bidirectional=True, persistent=True)
        self.compose = nn.Sequential(nn.Linear(2 * size, size), nn.Tanh())

        self.size = size
        self.itos = vocab.itos
        self.attention = attention

    def init(self, batch_size, open_wv, encoding=None):
        if PRINT:
            print('[', end='')
        zero = Variable(torch.zeros(1, self.size).cuda())
        self.open_wv = open_wv.unsqueeze(0)

        self.tok = None
        self.batch_size = batch_size
        self.stack = []
        self.history = [[] for i in range(batch_size)]

        for i in range(batch_size):
            x = self.open_wv
            s0 = (zero, zero)
            if encoding is not None:
                if self.attention:
                    x = encoding[i]
                else:
                    s0 = (encoding[i], encoding[i])
            self.stack.append(First(x, self.stack_rnn(x, s0)))

        return torch.cat([node.hc[0] for node in self.stack], 0)

    def step(self, xs, actions, tokens, treewards, attn_states=None, alphas=[], buffer_ixs=None):
        treewards.append(torch.zeros(self.batch_size).cuda())
        for ix in range(self.batch_size):
            top = self.stack[ix]
            kill_alpha = alphas != []
            if actions[-1][ix] == 2 and top.first is not None and len(
                    top.first.nodes) > 0:  # reduce
                self.tok = None
                if len(actions) > 1 and actions[-2][ix] == 2 or len(top.first.nodes) < 2:
                    treewards[-1][ix] = -1
                else:
                    child_types = [n.char for n in top.first.nodes]
                    if buffer_ixs[ix] != 0:
                        #if all(child_types):
                        #    treewards[-1][ix] = len(child_types)
                        #else:
                        #    treewards[-1][ix] = 8 * (len(child_types) - sum(child_types))
                        if all(child_types):
                            treewards[-1][ix] = 4 * len(top.first.nodes)
                        #elif not any(child_types):
                        #    treewards[-1][ix] = 8 * len(top.first.nodes)**0.5
                        else:
                            treewards[-1][ix] = 10 * (len(child_types) - sum(child_types)) ** 0.5 #2 * max(sum(child_types), (len(child_types) - sum(child_types))**0.5)  #2 * max(len(top.first.nodes), 5) - 1  # 2 if self.attention else 2
                if ix == 0 and PRINT:
                    print(']', end='', flush=True)
                bidi_input = torch.cat(
                    [node.x.unsqueeze(0) for node in top.first.nodes], 0)
                _, (h, c) = self.compose_rnn(bidi_input)
                x = self.compose(h.view(1, -1))
                # if self.attention:
                #     attn = torch.cat((x, attn_states[ix].unsqueeze(0)), 1)
                # else:
                #     attn = x
                self.stack[ix] = Node(
                    x, self.stack_rnn(x, top.hc), first=top.first.parent,
                    orig_open=top.first)
                self.history[ix].append(self.stack[ix])
            elif actions[-1][ix] == 1:  # shift
                if ix == 0:
                    self.tok = self.itos[tokens.data[ix]].replace('<init>', '#')
                    if PRINT:
                        print(self.tok, end='', flush=True)
                x = xs[ix].unsqueeze(0)
                # if self.attention:
                #     attn = torch.cat((x, attn_states[ix].unsqueeze(0)), 1)
                # else:
                #     attn = x
                self.stack[ix] = Node(
                    x, self.stack_rnn(x, top.hc), first=top.first, char=True)
                # self.history[ix].append(self.stack[ix])
            elif actions[-1][ix] == 0:  # open
                self.tok = None
                if len(actions) > 1 and actions[-2][ix] == 0:
                    treewards[-1][ix] = -1
                if ix == 0 and PRINT:
                    if alphas != []:
                        print(' ', *('{:0.3f}'.format(a) for a in alphas[-1].data))
                    print('[', end='', flush=True)
                x = self.open_wv
                if self.attention:
                    # treewards[-1][ix] -= alphas[-1][-1]  # + alphas[-1][-2]  # TODO batch
                    x = attn_states[ix].unsqueeze(0)
                    kill_alpha = False
                self.stack[ix] = First(
                    x, self.stack_rnn(x, top.hc), parent=top.first)
            elif top.first is not None:
                treewards[-1][ix] = -1
            if kill_alpha:
                alphas.pop()
        return torch.cat([node.hc[0] for node in self.stack], 0)


class RNNG(nn.Module):

    def __init__(self, size, embed, vocab, decoder=False, attention=False,
                 use_buffer=False):
        super().__init__()
        self.embed = embed
        self.stack = StackLSTM(size, vocab, attention)
        self.buffer_rnn = nn.LSTM(size, size, persistent=True)
        self.action = nn.Linear((2 if use_buffer else 1) * size, 3)
        self.out = nn.Linear(size, embed.num_embeddings)
        self.attn_linear = nn.Linear(size, size)
        self.criterion = nn.CrossEntropyLoss()

        self.size = size
        self.itos = vocab.itos
        self.decoder = decoder
        self.use_buffer = use_buffer
        self.sentinel_x = nn.Parameter(torch.randn(1, size))
        self.sentinel_h = nn.Parameter(torch.randn(1, size))

    def attend(self):
        attn_sums = []
        for stack_state, hist_x, hist_h in zip(
                self.stack_states, self.history_x, self.history_h):
            # torch.cat([hist_x, self.sentinel_x], 0)
            # torch.cat([hist_h, self.sentinel_h], 0)
            # query = F.tanh(self.attn_linear.weight @ stack_state + self.attn_linear.bias)
            query = stack_state
            alpha = F.softmax(hist_h @ query)
            self.alphas.append(alpha)  # .tolist()  # TODO batch
            attn_sums.append(
                (alpha.unsqueeze(1).expand_as(hist_x) * hist_x).sum(0))
        self.attention_states = torch.cat(attn_sums, 0)

    def __call__(self, data, history=None):
        T, B = data.size()
        self.targets = data
        self.buffers = self.embed(data.view(-1)).view(T, B, -1)
        self.buffer_states, _ = self.buffer_rnn(self.buffers)
        self.ixs = T - data.data.eq(1).long().sum(0).squeeze() - 1
        self.actions, self.tensor_actions, self.losses = [], [], []
        self.treewards, self.alphas = [], []
        # print(data)
        # print(ixs)
        if history is None:
            encoding = None
        else:
            encoding = [hist[-1].x for hist in history]
            self.history_x = [torch.cat([node.x for node in hist], 0)
                              for hist in history]
            self.history_h = [torch.cat([node.hc[0] for node in hist], 0)
                              for hist in history]

        self.stack_states = self.stack.init(B, self.buffers[0, 0], encoding)
        self.attention_states = None
        done = [False for i in range(B)]

        while not all(done):

            if self.decoder and self.stack.attention:
                self.attend()
            if self.use_buffer:
                self.condition = torch.cat(
                    [self.stack_states, get(self.buffer_states, self.ixs)], 1)
            else:
                self.condition = self.stack_states

            action_logits = self.action(self.condition)

            targets = get(self.targets, self.ixs)

            for ix in range(B):
                top = self.stack.stack[ix]
                done[ix] = top.first is None
                if not done[ix]:
                    if top.first.parent is None and self.ixs[ix] != 0:
                        # cannot close initial open unless buffer is empty
                        # seems not to trigger towards end? verify
                        #print(ix, action_logits.data[ix, 2])
                        action_logits.data[ix, 2] = -1e10
                        #print(ix, action_logits.data[ix, 2])

                if (not USE_RL_ENC and not self.decoder) or (
                        not USE_RL_DEC and self.decoder):
                    if self.stack.tok == '(':
                        action_logits.data[ix, 0] = 1e10
                    elif self.stack.tok == ')':
                        action_logits.data[ix, 2] = 1e10
                    else:
                        action_logits.data[ix, 1] = 1e10
                    # action_logits.data[ix, 0] = (
                    #     1e10 if len(self.actions) == 0 and (  # self.decoder or
                    #         random.random() > 0.5)
                    #     else -1e10)
                    # if not done[ix] and self.ixs[ix] != 0:
                    #     action_logits.data[ix, 2] = -1e10

            # print(action_logits)
            action_probs = F.softmax(action_logits)
            # print(action_probs)
            # 1. how to keep building graph [solved]
            # 2. how to let model know that it should only produce a legal
            # parse
            actions = action_probs.multinomial()
            self.actions.append(actions)
            self.tensor_actions.append(actions.squeeze().data.byte())
            # print('done   :', done)
            # print('indices:', self.ixs.tolist())
            # print('actions:', actions.tolist())
            # print(actions.tolist()[0], end='')

            self.step(self.tensor_actions[-1], targets)

        if PRINT:
            if self.decoder:
                print()
            else:
                print('=>', end='', flush=True)

        # 1/0

        # all_reduce = torch.LongTensor(B).fill_(2).cuda()
        # all_shift = torch.LongTensor(B).fill_(1).cuda()
        # all_open = torch.LongTensor(B).fill_(0).cuda()
        # print(self.ixs.tolist())
        # self.step(all_open)
        # print(self.ixs.tolist())
        # self.step(all_shift)
        # print(self.ixs.tolist())
        # self.step(all_shift)
        # print(self.ixs.tolist())
        # self.step(all_reduce)
        # print(self.ixs.tolist())
        # 1/0
        return self.stack.history, self.actions, (self.treewards, self.alphas), self.losses

    def step(self, actions, targets):
        # must reduce if buffer is empty
        actions.masked_fill_(self.ixs.eq(0), 2)
        # print(actions.tolist()[0], end='')
        # targets = get(self.targets, self.ixs)
        if True:  # self.decoder:
            B, C = self.stack_states.size()
            mask = Variable(
                actions.eq(1).unsqueeze(1).expand_as(self.stack_states))
            x = self.stack_states.masked_select(mask).view(-1, C)
            if len(x.size()) > 0:  # TODO re-scatter (irrelevant w/b=1)
                y = self.out(x)
                t = targets.masked_select(Variable(actions.eq(1))).unsqueeze(1)
                #print(y, t)
                loss = -F.log_softmax(y).gather(1, t).squeeze(1)
                loss.tokens = t.squeeze(1).data
                self.losses.append(loss)
            else:
                self.losses.append(0)

        buffer_tops = get(self.buffers, self.ixs)
        self.stack_states = self.stack.step(
            buffer_tops, self.tensor_actions, targets, self.treewards,
            self.attention_states, self.alphas, self.ixs)
        self.ixs -= actions.eq(1).long()


def repeat(gen, n):
    for x in gen:
        for i in range(n):
            yield x


def eval_epoch(model, data_iter):
    loss = 0
    for batch in data_iter:
        _, this_loss, seq_len = model(batch)
        loss += this_loss.data[0] / seq_len
    return loss / len(data_iter)

def train_until_fit(model, optimizer, data_iter, n_repeat, dev_iter=None, n_smooth=100):
    batch_size = 10
    for i, batch in enumerate(repeat(data_iter, n_repeat)):
        loss, dec_loss, seq_len = model(batch)
        if i % batch_size == 0:
            optimizer.zero_grad()
        last_loss = dec_loss.data[0] / seq_len
        loss.backward()
        if i % batch_size == batch_size - 1:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.div_(batch_size)
            optimizer.step()
        if i == 0:
            avg_loss = last_loss
        else:
            avg_loss = avg_loss + (last_loss - avg_loss) / n_smooth
        if dev_iter is not None and i % 10000 == -1 % 10000:
            dev_loss = eval_epoch(model, dev_iter)
            print('$$$$$', i + 1, dev_loss)
            torch.save(model.state_dict(), 't2t-{i}-{loss}.pt'.format(i=i, loss=dev_loss))
        if i % n_smooth == n_smooth - 1:
            print('#####', i + 1, avg_loss)
            if avg_loss < 0.01:
                break
        # if i == 10: exit()

if MT:
    tokenize = lambda x: list(reversed(x))
else:
    tokenize = list

CHARS = data.Field(tokenize=tokenize, init_token='<init>', eos_token='<pad>')

if MT:
    mt_train = datasets.TranslationDataset(
        path='/media/data/flickr30k/train', exts=('.de', '.en'), fields=(CHARS, CHARS))
    # mt_train.sort_key = lambda ex: len(ex.src)
    mt_dev = datasets.TranslationDataset(
        path='/media/data/flickr30k/val', exts=('.de', '.en'), fields=(CHARS, CHARS))
    CHARS.build_vocab(mt_train.src, mt_train.trg)
# CHARS.build_vocab('0123456789 +-()')  # RPNDataset
else:
    CHARS.build_vocab('abcdefghijklmnopqrst+*()')

import sys
gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 0
# model = S2S(128, 128, 1, CHARS.vocab)
model = T2T(384, CHARS.vocab)
model.cuda(gpu)

# optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
optimizer = torch.optim.Adam(model.parameters())

with torch.cuda.device(gpu):

    if MT:
        train_iter = data.BucketIterator(mt_train, batch_size=1, device=gpu)
        dev_iter = data.BucketIterator(mt_dev, batch_size=1, device=gpu)
        train_until_fit(model, optimizer, train_iter, 1, dev_iter)
    else:
        dataset = KimPNDataset(fields=[('src', CHARS), ('trg', CHARS)])#, max_depth=2, max_arity=2)
        data_iter = data.BucketIterator(dataset, batch_size=1, device=gpu,
                                        shuffle=False)
        train_until_fit(model, optimizer, data_iter, 1)