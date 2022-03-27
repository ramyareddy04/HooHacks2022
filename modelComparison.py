import matplotlib.pyplot as plt

val_accuracy_Categorical =  [0.0, 0.0, 0.0083, 0.0271, 0.0521, 0.1542, 0.3313, 0.7708, 0.5583, 0.6667, 0.5917, 0.7396, 0.7229, 0.6958, 0.5625, 0.7104, 0.6792, 0.7854, 0.7083, 0.75, 0.7063, 0.7937, 0.8188, 0.7417, 0.8062, 0.8479, 0.8208, 0.7583, 0.7937, 0.8396, 0.8, 0.8104, 0.7583, 0.8292, 0.8042, 0.8354, 0.8396, 0.8667, 0.8146, 0.7229, 0.85, 0.85, 0.7229, 0.8229, 0.7583, 0.6687, 0.7625, 0.7437, 0.8167, 0.8062]
val_accuracy_Binary = [0.7667, 0.8813, 0.8979, 0.925, 0.9458, 0.9729, 0.9146, 0.9625, 0.9604, 0.9812, 0.9708, 0.95, 0.9583, 0.9646, 0.9604, 0.9688, 0.9583, 0.95, 0.9604, 0.9604, 0.975, 0.9792, 0.9625, 0.9646, 0.975, 0.975, 0.9729, 0.9729, 0.975, 0.9688, 0.9937, 0.9771, 0.9458, 0.9583, 0.9708, 0.9604, 0.9625, 0.9729, 0.9729, 0.9688, 0.9688, 0.9708, 0.9688, 0.9708, 0.9812, 0.9667, 0.9792, 0.9875, 0.9708, 0.9563]

plt.plot(val_accuracy_Categorical, 'g', label='Categorical Validation Accuracy')
plt.plot(val_accuracy_Binary, 'b', label='Binary Validation Accuracy')
plt.title('Training and Validation Accuracy for Binary and Categorical Tests')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()