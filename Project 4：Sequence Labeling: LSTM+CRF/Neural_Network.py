import torch.nn as nn
import torch


class Named_Entity_Recognition(nn.Module):

    def __init__(self, len_feature, len_words, len_hidden, type_num, pad_id, start_id, end_id, weight=None,
                 drop_out=0.5):
        super(Named_Entity_Recognition, self).__init__()
        self.len_feature = len_feature
        self.len_words = len_words
        self.len_hidden = len_hidden
        self.dropout = nn.Dropout(drop_out)
        if weight is None:
            x = nn.init.xavier_normal_(torch.Tensor(len_words, len_feature))
            self.embedding = nn.Embedding(num_embeddings=len_words, embedding_dim=len_feature, _weight=x).cuda()
        else:
            self.embedding = nn.Embedding(num_embeddings=len_words, embedding_dim=len_feature, _weight=weight).cuda()
        self.lstm = nn.LSTM(input_size=len_feature, hidden_size=len_hidden, batch_first=True, bidirectional=True).cuda()
        self.fc = nn.Linear(2 * len_hidden, type_num).cuda()
        self.crf = CRF(type_num, pad_id, start_id, end_id).cuda()

    def forward(self, x, tags, mask):
        mask = mask.int()
        x = self.embedding(x)
        x = self.dropout(x)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        scores = self.fc(x)
        loss = self.crf(scores, tags, mask)
        return loss

    def predict(self, x, mask):
        mask = mask.int()
        x = self.embedding(x)
        x = self.dropout(x)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        scores = self.fc(x)

        return self.crf.predict(scores, mask)


class CRF(nn.Module):

    def __init__(self, type_num, pad_id, start_id, end_id):
        super(CRF, self).__init__()
        self.type_num = type_num
        self.pad_id = pad_id
        self.start_id = start_id
        self.end_id = end_id

        transition = torch.zeros(type_num, type_num)
        transition[:, start_id] = -10000.0
        transition[end_id, :] = -10000.0
        transition[:, pad_id] = -10000.0
        transition[pad_id, :] = -10000.0
        transition[pad_id, pad_id] = 0.0
        transition[pad_id, :end_id] = 0.0

        self.transition = nn.Parameter(transition).cuda()

    def forward(self, scores, tags, mask):
        true_prob = self.true_prob(scores, tags, mask)
        total_prob = self.total_prob(scores, mask)
        return -torch.sum(true_prob - total_prob)

    def true_prob(self, scores, tags, mask):
        batch_size, sequence_len = tags.shape
        true_prob = torch.zeros(batch_size).cuda()

        first_tag = tags[:, 0]
        last_tag_index = mask.sum(1) - 1
        last_tag = torch.gather(tags, 1, last_tag_index.unsqueeze(1)).squeeze(1)

        tran_score = self.transition[self.start_id, first_tag]
        tag_score = torch.gather(scores[:, 0], 1, first_tag.unsqueeze(1)).squeeze(1)

        true_prob += tran_score + tag_score

        for i in range(1, sequence_len):
            non_pad = mask[:, i]
            pre_tag = tags[:, i - 1]
            curr_tag = tags[:, i]

            tran_score = self.transition[pre_tag, curr_tag]
            tag_score = torch.gather(scores[:, i], 1, curr_tag.unsqueeze(1)).squeeze(1)

            true_prob += tran_score * non_pad + tag_score * non_pad

        true_prob += self.transition[last_tag, self.end_id]

        return true_prob

    def total_prob(self, scores, mask):
        batch_size, sequence_len, num_tags = scores.shape
        log_sum_exp_prob = self.transition[self.start_id, :].unsqueeze(0) + scores[:, 0]
        for i in range(1, sequence_len):
            every_log_sum_exp_prob = list()
            for j in range(num_tags):
                tran_score = self.transition[:, j].unsqueeze(0)
                tag_score = scores[:, i, j].unsqueeze(1)

                prob = tran_score + tag_score + log_sum_exp_prob

                every_log_sum_exp_prob.append(torch.logsumexp(prob, dim=1))

            new_prob = torch.stack(every_log_sum_exp_prob).t()

            non_pad = mask[:, i].unsqueeze(-1)
            log_sum_exp_prob = non_pad * new_prob + (1 - non_pad) * log_sum_exp_prob

        tran_score = self.transition[:, self.end_id].unsqueeze(0)
        return torch.logsumexp(log_sum_exp_prob + tran_score, dim=1)

    def predict(self, scores, mask):
        batch_size, sequence_len, num_tags = scores.shape
        total_prob = self.transition[self.start_id, :].unsqueeze(0) + scores[:, 0]
        tags = torch.cat([torch.tensor(range(num_tags)).view(1, -1, 1) for _ in range(batch_size)], dim=0).cuda()
        for i in range(1, sequence_len):
            new_prob = torch.zeros(batch_size, num_tags).cuda()
            new_tag = torch.zeros(batch_size, num_tags, 1).cuda()
            for j in range(num_tags):
                prob = total_prob + self.transition[:, j].unsqueeze(0) + scores[:, i, j].unsqueeze(1)
                max_prob, max_tag = torch.max(prob, dim=1)
                new_prob[:, j] = max_prob
                new_tag[:, j, 0] = max_tag

            non_pad = mask[:, i].unsqueeze(-1)
            total_prob = non_pad * new_prob + (1 - non_pad) * total_prob
            non_pad=non_pad.unsqueeze(-1)
            temp_tag=torch.cat([torch.tensor(range(num_tags)).view(1, -1, 1) for _ in range(batch_size)], dim=0).cuda()
            append_tag = non_pad * temp_tag + (1 - non_pad) * torch.ones(batch_size, num_tags, 1).cuda() * self.pad_id

            new_tag=new_tag.long()
            pre_tag=tags[[ [i]*num_tags for i in range(batch_size)],new_tag[:,:,0],:]

            tags = torch.cat([pre_tag, append_tag], dim=-1)

        prob = total_prob + self.transition[:, self.end_id].unsqueeze(0)
        _, max_tag = torch.max(prob, dim=1)

        return tags[[ i for i in range(batch_size)], max_tag]