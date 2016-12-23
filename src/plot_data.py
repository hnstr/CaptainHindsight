# file used for plotting and data analysis

import numpy as np
import matplotlib.pyplot as plt

f = open('weights.txt')

loss = []

for line in f:
        loss.append(line)

loss.reverse()
del loss[:3]
t = np.arange(len(loss))

plt.plot(t, loss, color='black', linewidth=2)
plt.ylabel('Training loss')
plt.xlabel('Rank')
plt.title('LSTM loss, sorted')
plt.grid(b=True)
plt.show()


'''
plot the improvement in BLEU-n metric compared to beginning of training stage.
'''

# 2-gram
m2 = 0.297869240454
s2 = 0.13980173794816966

# 3-gram
m3 = 0.388426194332
s3 = 0.16968486944449018

# 4-gram
m4 = 0.442276023404
s4 = 0.187628458674268

# plot BLEU-2
plt.plot(1, m2, 'ks', markersize=10)
plt.errorbar(1, m2, yerr=s2, fmt='', color='k', linewidth=2)

# plot BLEU-3
plt.plot(2, m3, 'ks', markersize=10)
axes = plt.errorbar(2, m3, yerr=s3, fmt='', color='k', linewidth=2)

# plot BLEU-4
plt.plot(3, m4, 'ks', markersize=10)
axes = plt.errorbar(3, m4, yerr=s4, fmt='', color='k', linewidth=2)

plt.xticks(range(4), ['', 'BLEU-2', 'BLEU-3', 'BLEU-4'], color='black')
plt.xlim((0.5, 3.5))
plt.ylim((0, 1))
plt.grid(b=True)
plt.ylabel('BLEU (normalised)')
plt.title('n-gram BLEU score on validation set')
plt.show()

