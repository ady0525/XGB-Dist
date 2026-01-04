import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.metrics import (
    accuracy_score,
    auc,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)

def plot_confusion_matrix(y_true, y_pred, labels=["Predicted 1", "Predicted 0"], columns=["Actual 1", "Actual 0"]):
    # print(confusion_matrix(y_true, y_pred))
    confusion_matrix_df = pd.DataFrame(
        confusion_matrix(y_true, y_pred)[::-1,::-1].T, index=labels, columns=columns
    )
    cmap = sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True)
    # res = sns.heatmap(confusion_matrix_df, annot=True, vmin=0.0, fmt=".2f", cmap=cmap)
    res = sns.heatmap(confusion_matrix_df, annot=True, vmin=0.0, fmt=".0f", cmap=cmap)
    plt.yticks(rotation=0)
    plt.title("Confusion Matrix")
    plt.show()


def plot_roc(y_true, prob):
    # Area under the curve(AUC) and Receiver Operating Characteristic(ROC)
    fpr, tpr, thresholds = roc_curve(y_true, prob)

    # Chosing the best threshold (the one who maximize tpr - fpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # auc = roc_auc_score(y_test_ub, prob)
    area = auc(fpr, tpr)

    # Gini coefficient
    GINI = (2 * area) - 1

    nearest_value_index = (np.abs(thresholds - 0.5)).argmin()

    # Plot ROC curve
    plt.figure(figsize=(6, 5.8))
    # plt.title(" ROC, AUC and Gini")
    plt.plot(
        # fpr, tpr, label="gini=" + str(round(GINI * 100, 2)), color="black", zorder=2
        fpr, tpr, label='ROC', color="black", zorder=2
    )
    plt.hlines(
        y=tpr[optimal_idx],
        xmin=0,
        xmax=1,
        colors="grey",
        linestyles="dotted",
        lw=2,
        zorder=1,
    )
    plt.vlines(
        x=fpr[optimal_idx],
        ymin=0,
        ymax=1,
        colors="grey",
        linestyles="dotted",
        lw=2,
        zorder=1,
    )
    plt.scatter(
        fpr[optimal_idx],
        tpr[optimal_idx],
        color="red",
        label="Best threshold: " + str(round(optimal_threshold, 2)),
        zorder=10,
    )
    plt.hlines(
        y=tpr[nearest_value_index],
        xmin=0,
        xmax=1,
        colors="grey",
        linestyles="dotted",
        lw=2,
        zorder=1,
    )
    plt.vlines(
        x=fpr[nearest_value_index],
        ymin=0,
        ymax=1,
        colors="grey",
        linestyles="dotted",
        lw=2,
        zorder=1,
    )
    plt.scatter(
        fpr[nearest_value_index],
        tpr[nearest_value_index],
        color="green",
        label="Threshold: 0.5",
        zorder=10,
    )
    # plt.plot(fpr_cv, tpr_cv, label="gini cv="+str(GINI_cv), color='red')
    plt.plot([0, 1], [0, 1], "--", color="red")
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.xticks(fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.yticks(fontsize=12)
    plt.legend(loc=4)
    # plt.axis('equal')
    plt.show()

    print("Gini coefficient: {}".format(round(GINI, 2)))
    print("AUC-ROC: {}".format(round(area, 2)))


def plot_outcome_prob_relation(y_true, y_pred, prob):
  N = prob.shape[0]

  prob_TP = []
  prob_FP = []
  prob_FN = []
  prob_TN = []

  for i in range(N):
    if y_pred[i] == 1 and y_true[i] == 1:
      prob_TP.append(prob[i])
    elif y_pred[i] == 1 and y_true[i] == 0:
      prob_FP.append(prob[i])
    elif y_pred[i] == 0 and y_true[i] == 1:
      prob_FN.append(prob[i])
    elif y_pred[i] == 0 and y_true[i] == 0:
      prob_TN.append(prob[i])

  bins = 20
  plt.figure(figsize=(7, 4.5))
  plt.hist([prob_FN, prob_TN], bins=bins, stacked=True, label=['FN', 'TN'], edgecolor = 'black',
          color=[mcolors.CSS4_COLORS['lightsteelblue'], mcolors.TABLEAU_COLORS['tab:blue']])
  plt.hist([prob_FP, prob_TP], bins=bins, stacked=True, label=['FP', 'TP'], edgecolor = 'black',
          color=[mcolors.CSS4_COLORS['lightcoral'], mcolors.TABLEAU_COLORS['tab:red']])
  plt.legend()
  plt.xlabel('Probability', fontsize=14)
  plt.xticks(np.arange(0, 1.1, step=0.1))
  plt.xticks(fontsize=12)
  plt.ylabel('Counts', fontsize=14)
  plt.yticks(fontsize=12)
  plt.show()


def plot_feature_importance(imp, num_names, cat_names):
  I_sorted = np.argsort(-imp)
  imp_sorted = imp[I_sorted]

  names = np.array(num_names + cat_names)
  names_sorted = names[I_sorted]

  colors_sorted = []
  for name in names_sorted:
    if name in num_names:
      colors_sorted.append('blue')
    elif name in cat_names:
      colors_sorted.append('green')

  imp_perc_sorted = imp_sorted / np.sum(imp_sorted) * 100
  num_disp = 20

  plt.figure(figsize=(8, 4))
  plt.bar(names_sorted[:num_disp], imp_perc_sorted[:num_disp], color=colors_sorted[:num_disp], capsize=3)
  plt.xticks(fontsize=12)
  plt.xticks(rotation='vertical')
  plt.ylabel('Feature Importances (%)', fontsize=14)
  plt.yticks(fontsize=12)
  plt.show()
  
  return imp_perc_sorted[:num_disp], names_sorted[:num_disp], colors_sorted[:num_disp]
