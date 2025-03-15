#  Copyright (c) 2022 Szymon Mikler

"""
Original:
https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

CODE FROM THE ABOVE LINK WAS REWRITTEN USING PYTORCH SYMBOLIC
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pytorch_symbolic import Input, SymbolicModel, add_to_graph


def run() -> None:
    """
    Example: An LSTM for Part-of-Speech Tagging

    Prepare data:
    """

    def prepare_sequence(seq, to_ix):
        idxs = [to_ix[w] for w in seq]
        return torch.tensor(idxs, dtype=torch.long)

    training_data = [
        # Tags are: DET - determiner; NN - noun; V - verb
        # For example, the word "The" is a determiner
        ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
        ("Everybody read that book".split(), ["NN", "V", "DET", "NN"]),
    ]
    word_to_ix: Dict[str, int] = {}
    # For each words-list (sentence) and tags-list in each tuple of training_data
    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_ix:  # word has not been assigned an index yet
                word_to_ix[word] = len(word_to_ix)  # Assign each word with a unique index
    print(word_to_ix)
    tag_to_ix = {"DET": 0, "NN": 1, "V": 2}  # Assign each tag with a unique index

    # These will usually be more like 32 or 64 dimensional.
    # We will keep them small, so we can see how the weights change as we train.
    EMBEDDING_DIM = 6
    HIDDEN_DIM = 6

    """
    Create the model:
    """

    class LSTMTagger(nn.Module):
        def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
            super(LSTMTagger, self).__init__()
            self.hidden_dim = hidden_dim

            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

            # The LSTM takes word embeddings as inputs, and outputs hidden states
            # with dimensionality hidden_dim.
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

            # The linear layer that maps from hidden state space to tag space
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        def forward(self, sentence):
            embeds = self.word_embeddings(sentence)
            lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
            tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
            tag_scores = F.log_softmax(tag_space, dim=1)
            return tag_scores

    def create_lstm_tagger(embedding_dim, hidden_dim, vocab_size, tagset_size):
        """Equivalent of LSTMTagger."""
        sentence = Input(batch_shape=(30,), dtype=torch.int)
        embeds = nn.Embedding(vocab_size, embedding_dim)(sentence)
        lstm_out, _ = nn.LSTM(embedding_dim, hidden_dim)(embeds.unsqueeze(1))
        tag_space = nn.Linear(hidden_dim, tagset_size)(lstm_out.squeeze())
        tag_scores = add_to_graph(F.log_softmax, tag_space, dim=1)
        model = SymbolicModel(sentence, tag_scores)
        return model

    model = create_lstm_tagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # See what the scores are before training
    # Note that element i,j of the output is the score for tag j for word i.
    # Here we don't need to train, so the code is wrapped in torch.no_grad()
    with torch.no_grad():
        inputs = prepare_sequence(training_data[0][0], word_to_ix)
        tag_scores = model(inputs)
        print(tag_scores.argmax(1).tolist())

    for _epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)

            # Step 3. Run our forward pass.
            tag_scores = model(sentence_in)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()

    # See what the scores are after training
    with torch.no_grad():
        inputs = prepare_sequence(training_data[0][0], word_to_ix)
        tag_scores = model(inputs)

        # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
        # for word i. The predicted tag is the maximum scoring tag.
        # Here, we can see the predicted sequence below is 0 1 2 0 1
        # since 0 is index of the maximum value of row 1,
        # 1 is the index of maximum value of row 2, etc.
        # Which is DET NOUN VERB DET NOUN, the correct sequence!
        print("PREDICT", tag_scores.argmax(1).tolist())

    print("CORRECT", [0, 1, 2, 0, 1])
