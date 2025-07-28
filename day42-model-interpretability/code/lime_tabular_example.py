 
import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target
model = RandomForestClassifier().fit(X, y)

explainer = lime.lime_tabular.LimeTabularExplainer(X, feature_names=iris.feature_names, class_names=iris.target_names)
exp = explainer.explain_instance(X[0], model.predict_proba)
exp.show_in_notebook()
