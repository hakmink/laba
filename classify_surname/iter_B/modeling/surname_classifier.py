import modeling.surname_common as sc
import torch.nn as nn
import torch
import json
import pprint


criterion = nn.NLLLoss()
LEARNING_RATE = 0.005

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

    
def train(rnn, category_tensor, surname_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(surname_tensor.size()[0]):
        output, hidden = rnn(surname_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-LEARNING_RATE, p.grad.data)

    return output, loss.item()  
    
    

def evaluate(rnn, line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

def predict(rnn, input_surname, n_predictions=3):
    with torch.no_grad():
        surname_tensor = sc.surname_to_tensor(input_surname)
        output = evaluate(rnn, surname_tensor)

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            predictions.append([value, category_index])
        return predictions 

def load_model():
    return torch.load('data/model/rnn.pickle')