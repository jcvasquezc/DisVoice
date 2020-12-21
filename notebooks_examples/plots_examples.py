
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import metrics
import itertools

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(5,5))
    cm = metrics.confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, np.round(cm[i, j],2),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('Real class')
    plt.xlabel('Predicted class')
    plt.tight_layout()
    plt.show()

def plot_ROC(ytest, score_test):
    plt.figure(figsize=(5,5))
    
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)
    fprs, tprs, thresholds = metrics.roc_curve(ytest, score_test)
    AUC=metrics.auc(fprs, tprs)
    plt.plot(fprs, tprs, color='k',
         label=r'Avg. ROC (AUC = %0.3f)' % (AUC),
         lw=2, alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()
    
    plt.show()

def plot_histogram(ytest, score_test, name_clases):
    plt.figure(figsize=(5,5))
    p1=np.where(np.hstack(ytest)==0)[0]
    p2=np.where(np.hstack(ytest)==1)[0]
    sns.distplot(np.hstack(score_test)[p1], label=name_clases[0])
    sns.distplot(np.hstack(score_test)[p2], label=name_clases[1], color="k")
    plt.grid()
    plt.legend()
    plt.xlabel("Classification threshold")
    plt.ylabel('Normalized count')
    plt.tight_layout()
    plt.show()