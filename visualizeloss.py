import pickle
import torch
import matplotlib.pyplot as plt
resultsdir = 'results/trial6/training_loss_list.pkl'
with open(resultsdir, 'rb') as f:
	training_loss_list = pickle.load(f)
print('training_loss_list: ', training_loss_list)
epoch_list = list(range(len(training_loss_list)))
print('epoch_list: ', epoch_list)
plt.plot(epoch_list, training_loss_list)
plt.xlabel('epochs')
plt.ylabel('training loss')
# giving a title to my graph
plt.title('training loss vs epochs')
plt.show()
