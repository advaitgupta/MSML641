import torch
import torch.nn as nn

class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout,
                 model_type='lstm', activation='tanh', bidirectional=False):

        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0) 
        
        self.model_type = model_type.lower()
        self.bidirectional = bidirectional
        
        rnn_module = None
        if self.model_type == 'rnn':
            rnn_module = nn.RNN
        elif self.model_type == 'lstm':
            rnn_module = nn.LSTM
        elif self.model_type == 'bilstm':
            rnn_module = nn.LSTM
            self.bidirectional = True 
        else:
            raise ValueError("Unsupported model_type: choose 'rnn', 'lstm', or 'bilstm'")

        # Handle activation functions
        rnn_kwargs = {
            'input_size': embed_dim,
            'hidden_size': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout if num_layers > 1 else 0,
            'batch_first': True,
            'bidirectional': self.bidirectional
        }
        
        if self.model_type == 'rnn':
            if activation in ['tanh', 'relu', 'sigmoid']:
                rnn_kwargs['nonlinearity'] = activation 
            else:
                rnn_kwargs['nonlinearity'] = 'tanh' 
        
        self.rnn = rnn_module(**rnn_kwargs)
        
        self.dropout = nn.Dropout(dropout)
        
        fc_input_dim = hidden_dim * 2 if self.bidirectional else hidden_dim
        
        self.fc = nn.Linear(fc_input_dim, 1)

    def forward(self, text):
        
        embedded = self.dropout(self.embedding(text))
        
        if self.model_type == 'rnn':
            output, hidden = self.rnn(embedded)
        else: 
            output, (hidden, cell) = self.rnn(embedded)
        
        if self.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
                    
        prediction = self.fc(hidden)
        
        return prediction