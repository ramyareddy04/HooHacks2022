import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

txt = """Epoch 1/50
136/136 [==============================] - 9s 58ms/step - loss: 0.6644 - accuracy: 0.5747 - val_loss: 0.5736 - val_accuracy: 0.7667
Epoch 2/50
136/136 [==============================] - 7s 52ms/step - loss: 0.4483 - accuracy: 0.7996 - val_loss: 0.2909 - val_accuracy: 0.8813
Epoch 3/50
136/136 [==============================] - 7s 53ms/step - loss: 0.3461 - accuracy: 0.8573 - val_loss: 0.2749 - val_accuracy: 0.8979
Epoch 4/50
136/136 [==============================] - 8s 56ms/step - loss: 0.3166 - accuracy: 0.8901 - val_loss: 0.2095 - val_accuracy: 0.9250
Epoch 5/50
136/136 [==============================] - 7s 54ms/step - loss: 0.2883 - accuracy: 0.8954 - val_loss: 0.1845 - val_accuracy: 0.9458
Epoch 6/50
136/136 [==============================] - 7s 54ms/step - loss: 0.2386 - accuracy: 0.9242 - val_loss: 0.1129 - val_accuracy: 0.9729
Epoch 7/50
136/136 [==============================] - 7s 55ms/step - loss: 0.2473 - accuracy: 0.9221 - val_loss: 0.2212 - val_accuracy: 0.9146
Epoch 8/50
136/136 [==============================] - 7s 52ms/step - loss: 0.2165 - accuracy: 0.9223 - val_loss: 0.1732 - val_accuracy: 0.9625
Epoch 9/50
136/136 [==============================] - 7s 52ms/step - loss: 0.2079 - accuracy: 0.9356 - val_loss: 0.1545 - val_accuracy: 0.9604
Epoch 10/50
136/136 [==============================] - 7s 53ms/step - loss: 0.1912 - accuracy: 0.9351 - val_loss: 0.0893 - val_accuracy: 0.9812
Epoch 11/50
136/136 [==============================] - 7s 54ms/step - loss: 0.1754 - accuracy: 0.9434 - val_loss: 0.1051 - val_accuracy: 0.9708
Epoch 12/50
136/136 [==============================] - 8s 58ms/step - loss: 0.1728 - accuracy: 0.9492 - val_loss: 0.1597 - val_accuracy: 0.9500
Epoch 13/50
136/136 [==============================] - 7s 54ms/step - loss: 0.1588 - accuracy: 0.9463 - val_loss: 0.1479 - val_accuracy: 0.9583
Epoch 14/50
136/136 [==============================] - 7s 54ms/step - loss: 0.1541 - accuracy: 0.9450 - val_loss: 0.1186 - val_accuracy: 0.9646
Epoch 15/50
136/136 [==============================] - 8s 56ms/step - loss: 0.1569 - accuracy: 0.9443 - val_loss: 0.1234 - val_accuracy: 0.9604
Epoch 16/50
136/136 [==============================] - 8s 56ms/step - loss: 0.1440 - accuracy: 0.9506 - val_loss: 0.1019 - val_accuracy: 0.9688
Epoch 17/50
136/136 [==============================] - 8s 58ms/step - loss: 0.1426 - accuracy: 0.9541 - val_loss: 0.1344 - val_accuracy: 0.9583
Epoch 18/50
136/136 [==============================] - 8s 56ms/step - loss: 0.1634 - accuracy: 0.9457 - val_loss: 0.1619 - val_accuracy: 0.9500
Epoch 19/50
136/136 [==============================] - 7s 54ms/step - loss: 0.1336 - accuracy: 0.9561 - val_loss: 0.1311 - val_accuracy: 0.9604
Epoch 20/50
136/136 [==============================] - 8s 55ms/step - loss: 0.1413 - accuracy: 0.9518 - val_loss: 0.1459 - val_accuracy: 0.9604
Epoch 21/50
136/136 [==============================] - 7s 53ms/step - loss: 0.1376 - accuracy: 0.9554 - val_loss: 0.0829 - val_accuracy: 0.9750
Epoch 22/50
136/136 [==============================] - 8s 59ms/step - loss: 0.1158 - accuracy: 0.9552 - val_loss: 0.0873 - val_accuracy: 0.9792
Epoch 23/50
136/136 [==============================] - 8s 61ms/step - loss: 0.1136 - accuracy: 0.9590 - val_loss: 0.1331 - val_accuracy: 0.9625
Epoch 24/50
136/136 [==============================] - 8s 56ms/step - loss: 0.1352 - accuracy: 0.9480 - val_loss: 0.1340 - val_accuracy: 0.9646
Epoch 25/50
136/136 [==============================] - 8s 58ms/step - loss: 0.1180 - accuracy: 0.9626 - val_loss: 0.0999 - val_accuracy: 0.9750
Epoch 26/50
136/136 [==============================] - 8s 58ms/step - loss: 0.1150 - accuracy: 0.9613 - val_loss: 0.0940 - val_accuracy: 0.9750
Epoch 27/50
136/136 [==============================] - 8s 59ms/step - loss: 0.1140 - accuracy: 0.9599 - val_loss: 0.1156 - val_accuracy: 0.9729
Epoch 28/50
136/136 [==============================] - 8s 60ms/step - loss: 0.1378 - accuracy: 0.9540 - val_loss: 0.1042 - val_accuracy: 0.9729
Epoch 29/50
136/136 [==============================] - 8s 56ms/step - loss: 0.1062 - accuracy: 0.9605 - val_loss: 0.1150 - val_accuracy: 0.9750
Epoch 30/50
136/136 [==============================] - 7s 54ms/step - loss: 0.1127 - accuracy: 0.9598 - val_loss: 0.1285 - val_accuracy: 0.9688
Epoch 31/50
136/136 [==============================] - 7s 54ms/step - loss: 0.1070 - accuracy: 0.9663 - val_loss: 0.0370 - val_accuracy: 0.9937
Epoch 32/50
136/136 [==============================] - 7s 54ms/step - loss: 0.1251 - accuracy: 0.9597 - val_loss: 0.0813 - val_accuracy: 0.9771
Epoch 33/50
136/136 [==============================] - 7s 53ms/step - loss: 0.1191 - accuracy: 0.9566 - val_loss: 0.2133 - val_accuracy: 0.9458
Epoch 34/50
136/136 [==============================] - 7s 54ms/step - loss: 0.1151 - accuracy: 0.9601 - val_loss: 0.1498 - val_accuracy: 0.9583
Epoch 35/50
136/136 [==============================] - 7s 53ms/step - loss: 0.0841 - accuracy: 0.9737 - val_loss: 0.1004 - val_accuracy: 0.9708
Epoch 36/50
136/136 [==============================] - 8s 57ms/step - loss: 0.0998 - accuracy: 0.9654 - val_loss: 0.1705 - val_accuracy: 0.9604
Epoch 37/50
136/136 [==============================] - 8s 56ms/step - loss: 0.0985 - accuracy: 0.9659 - val_loss: 0.1376 - val_accuracy: 0.9625
Epoch 38/50
136/136 [==============================] - 8s 60ms/step - loss: 0.0850 - accuracy: 0.9728 - val_loss: 0.1173 - val_accuracy: 0.9729
Epoch 39/50
136/136 [==============================] - 8s 56ms/step - loss: 0.0859 - accuracy: 0.9731 - val_loss: 0.0995 - val_accuracy: 0.9729
Epoch 40/50
136/136 [==============================] - 8s 55ms/step - loss: 0.0901 - accuracy: 0.9701 - val_loss: 0.1241 - val_accuracy: 0.9688
Epoch 41/50
136/136 [==============================] - 7s 55ms/step - loss: 0.0961 - accuracy: 0.9662 - val_loss: 0.1541 - val_accuracy: 0.9688
Epoch 42/50
136/136 [==============================] - 7s 54ms/step - loss: 0.0808 - accuracy: 0.9730 - val_loss: 0.1000 - val_accuracy: 0.9708
Epoch 43/50
136/136 [==============================] - 7s 55ms/step - loss: 0.0898 - accuracy: 0.9722 - val_loss: 0.1300 - val_accuracy: 0.9688
Epoch 44/50
136/136 [==============================] - 7s 53ms/step - loss: 0.0937 - accuracy: 0.9667 - val_loss: 0.1192 - val_accuracy: 0.9708
Epoch 45/50
136/136 [==============================] - 8s 56ms/step - loss: 0.0771 - accuracy: 0.9768 - val_loss: 0.0730 - val_accuracy: 0.9812
Epoch 46/50
136/136 [==============================] - 7s 55ms/step - loss: 0.0897 - accuracy: 0.9726 - val_loss: 0.1402 - val_accuracy: 0.9667
Epoch 47/50
136/136 [==============================] - 7s 55ms/step - loss: 0.0863 - accuracy: 0.9732 - val_loss: 0.0840 - val_accuracy: 0.9792
Epoch 48/50
136/136 [==============================] - 7s 55ms/step - loss: 0.0874 - accuracy: 0.9733 - val_loss: 0.0821 - val_accuracy: 0.9875
Epoch 49/50
136/136 [==============================] - 7s 53ms/step - loss: 0.0853 - accuracy: 0.9721 - val_loss: 0.1334 - val_accuracy: 0.9708
Epoch 50/50
136/136 [==============================] - 7s 53ms/step - loss: 0.0808 - accuracy: 0.9719 - val_loss: 0.1515 - val_accuracy: 0.9563"""

data = txt.split(" - ")

loss = []
accuracy = []
val_loss = []
val_accuracy = []
for i in data:
    if i.find("loss: ") != -1 and i.find("val_loss: ") != 0:
        loss.append(float(i[6:]))
    elif i.find("val_loss: ") == 0:
        val_loss.append(float(i[10:]))
    elif i.find("accuracy: ") != -1 and i.find("val_accuracy: ") != 0:
        accuracy.append(float(i[10:]))
    elif i.find("val_accuracy: ") == 0:
        val_accuracy.append(float(i[14:20]))


print(len(loss), len(val_loss), len(accuracy), len(val_accuracy))
print("loss: " + str(loss))
print("val_loss: " + str(val_loss))
print("accuracy: " + str(accuracy))
print("val_accuracy: " + str(val_accuracy))
fig, (ax1,ax2) = plt.subplots(1,2)

ax1.plot(loss, 'g', label='Training loss')
ax1.plot(val_loss, 'b', label='validation loss')
ax1.set_title('Training and Validation loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()


ax2.plot(accuracy, 'g', label='Training accuracy')
ax2.plot(val_accuracy, 'b', label='validation accuracy')
ax2.set_title('Training and Validation accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()
plt.show()


