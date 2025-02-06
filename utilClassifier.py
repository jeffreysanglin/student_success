from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score, RocCurveDisplay, accuracy_score
from numpy import log1p, arange, where
from pandas import Series, DataFrame

import seaborn as sns
import matplotlib.pyplot as plt

class Classifier():
    def __init__(self, data, X, y, estimator, gparams_dict, normalize=False, scale_cols=None, log_cols=None, SD=42):
        """
        Create a classifier object where we can plug in all our settings and hyperparameters.
        """
        self.data = data.copy()
        #self.X = X
        #self.y = y
        self.estimator = estimator
        self.grid_params = gparams_dict
        self.scale_cols=scale_cols
        self.log_cols=log_cols
        self.SD = SD
        
        if normalize:
            scaler = StandardScaler()
            self.data[self.scale_cols] = scaler.fit_transform(X=self.data[self.scale_cols])
            
            lg_transformer = FunctionTransformer(log1p)
            self.data[self.log_cols] = scaler.fit_transform(X=self.data[self.log_cols])
        
        #Assign vars
        self.X = self.data.drop(y, inplace=False, axis=1)
        self.y = self.data[y]
        
        #Train Test Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.33, stratify=self.y, random_state=self.SD)
        
        #Instantiate model and hyperparams
        self.estimator=estimator(random_state=self.SD)
        
        self.gscv_ = GridSearchCV(self.estimator, 
                            param_grid=self.grid_params, scoring="neg_mean_squared_error",
                            cv=5, verbose=True)
    
    def fitModel(self):
        """
        Fitting the classifier model
        """
        self.gscv_.fit(self.X_train, self.y_train)
        return(self)
    
    def showMetrics(self):
        """
        Printing the metrics of the model
        """
        self.y_preds = self.gscv_.predict(self.X_test)
        print(f"RMSE: {mean_squared_error(self.y_test, self.y_preds)**(1/2)}")
        print(f"Accuracy Score: {accuracy_score(self.y_test, self.y_preds)}")
        self.y_proba = self.gscv_.predict_proba(self.X_test)[:,1]
        
        # Identify misclassified examples
        self.misclassified_idx = where(self.y_test != self.y_preds)[0]
        
        # Get the incorrect predictions with the lowest confidence in their true class
        #misses = [(i, self.y_test[i], self.y_preds[i], self.y_proba[i].max()) for i in self.misclassified_idx]
        #self.sorted_misses = sorted(misses, key=lambda x: x[3])[:10]  # Least confident misses

        #for i, true_label, pred_label, confidence in self.sorted_misses:
        #    print(f"Index: {i}, True: {true_label}, Predicted: {pred_label}, Confidence: {confidence}")
            
        #print(f"Roc AUC Score: {roc_auc_score(self.y_test, self.y_preds, multi_class='ovr')}")
        #RocCurveDisplay(
        return(self)
    
    def showFeatureImportance(self):
        """
        Chart the feature importance of the classifier
        """
        self.best_est = self.gscv_.best_estimator_
        self.features = dict(zip(list(self.data.columns[:-1]), self.best_est.feature_importances_))
        self.features = DataFrame(Series(self.features))
        self.features.columns = ['importance']
        self.features = self.features.sort_values(by='importance', ascending=False)
        #display(features)

        ax = sns.heatmap(self.features, annot=True)
        ax.set_yticks(arange(len(self.features)) + 0.5)
        ax.set_yticklabels(self.features.index, rotation=0)
        plt.show()
        return(self)