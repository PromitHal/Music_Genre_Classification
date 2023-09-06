from sklearn.metrics import f1_score,accuracy_score,confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class Evaluate:
    def __init__(self,true_labels,preds,display_labels=None,save_path=None):
        self.save_path=save_path
        self.true_labels=true_labels
        self.preds=preds
        self.display_labels=display_labels

    def calc_eval(self):
        acc=accuracy_score(self.true_labels,self.preds)
        f1=f1_score(self.true_labels,self.preds,average='macro')
        cm=confusion_matrix(self.true_labels,self.preds)
        disp=ConfusionMatrixDisplay(cm,display_labels=self.display_labels)
        disp.plot()
        plt.show()
        print('Acc :{:.2f}\nF1 {:.2f}'.format(acc,f1))
        return [acc,f1]
