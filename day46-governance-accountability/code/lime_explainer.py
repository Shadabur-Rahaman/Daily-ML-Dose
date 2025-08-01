from lime.lime_tabular import LimeTabularExplainer
explainer = LimeTabularExplainer(X_train, feature_names=features)
explanation = explainer.explain_instance(X_test[0], model.predict_proba)
explanation.save_to_file('lime_explanation.html')
