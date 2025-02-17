"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error
import torch

from utils import create_test_dataset
from models import DeepSets, LSTM

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
batch_size = 64
embedding_dim = 128
hidden_dim = 64

# Generates test data
X_test, y_test = create_test_dataset()
cards = [X_test[i].shape[1] for i in range(len(X_test))]
n_samples_per_card = X_test[0].shape[0]
n_digits = 11

# Retrieves DeepSets model
deepsets = DeepSets(n_digits, embedding_dim, hidden_dim).to(device)
print("Loading DeepSets checkpoint!")
checkpoint = torch.load('model_deepsets.pth.tar')
deepsets.load_state_dict(checkpoint['state_dict'])
deepsets.eval()

# Retrieves LSTM model
lstm = LSTM(n_digits, embedding_dim, hidden_dim).to(device)
print("Loading LSTM checkpoint!")
checkpoint = torch.load('model_lstm.pth.tar')
lstm.load_state_dict(checkpoint['state_dict'])
lstm.eval()

# Dict to store the results
results = {'deepsets': {'acc':[], 'mae':[]}, 'lstm': {'acc':[], 'mae':[]}}

for i in range(len(cards)):
    y_pred_deepsets = list()
    y_pred_lstm = list()
    n_samples_per_card=X_test[i].shape[0]
   
    for j in range(0, n_samples_per_card, batch_size):
        
        ############## Task 6
    
        ##################
        
        x_batch=X_test[i][j:min(j+batch_size,n_samples_per_card),:]
        
        y_batch=y_test[i][j:min(j+batch_size,n_samples_per_card)]
        x_batch=torch.LongTensor(x_batch).to(device)
        y_batch=torch.LongTensor(y_batch).to(device)
        output_deepsets = deepsets(x_batch)
        
        output_lstm=lstm(x_batch)
        y_pred_deepsets.append(output_deepsets)
        y_pred_lstm.append(output_lstm)
        ##################
        
    y_pred_deepsets = torch.cat(y_pred_deepsets)
    y_pred_deepsets = y_pred_deepsets.detach().cpu().numpy()
    y_pred_deepsets = np.floor(y_pred_deepsets).astype(int)

    acc_deepsets = accuracy_score(y_test[i],y_pred_deepsets)
    mae_deepsets = mean_absolute_error(y_test[i],y_pred_deepsets)
    results['deepsets']['acc'].append(acc_deepsets)
    results['deepsets']['mae'].append(mae_deepsets)
    
    y_pred_lstm = torch.cat(y_pred_lstm)
    y_pred_lstm = y_pred_lstm.detach().cpu().numpy()
    
    y_pred_lstm = np.floor(y_pred_lstm).astype(int)
    acc_lstm = accuracy_score(y_test[i],y_pred_lstm)
    mae_lstm = mean_absolute_error(y_test[i],y_pred_lstm)
    results['lstm']['acc'].append(acc_lstm)
    results['lstm']['mae'].append(mae_lstm)
    


############## Task 7
    
##################
plt.plot(np.linspace(0,len(cards),len(cards)),np.array(results['deepsets']['acc']),label='deepsets')
plt.plot(np.linspace(0,len(cards),len(cards)),np.array(results['lstm']['acc']),label='lstm')
plt.xlabel('cardinality')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_vs_cardinality.png') 
plt.show()
##################
