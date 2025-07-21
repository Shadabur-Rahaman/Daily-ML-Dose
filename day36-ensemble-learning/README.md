# 🧩 Day 36 – Ensemble Learning Techniques

Welcome to **Day 36** of the #DailyMLDose challenge!

Today, we dive deep into the fascinating world of **Ensemble Learning** — combining multiple models to build one superior predictive model.

---

## 🔍 What is Ensemble Learning?

Ensemble Learning involves using multiple learning algorithms to obtain better predictive performance than any individual model. The core idea is **wisdom of the crowd**.

---

## 📚 Key Techniques Covered

| Technique       | Core Idea                                      | Example Models         |
|-----------------|------------------------------------------------|------------------------|
| 🧺 **Bagging**    | Train models in parallel on bootstrapped data | Random Forest          |
| 🚀 **Boosting**   | Sequentially train models to fix predecessors | XGBoost, AdaBoost      |
| 🧠 **Stacking**   | Train base models & meta-model over outputs   | Any combo of models    |
| ⚖️ **Voting**     | Combine model predictions via voting          | Hard / Soft Voting     |
| 🔄 **Blending**   | Similar to stacking, but on a holdout set     | Gradient + CatBoost    |

---

## 📁 Folder Structure

```css
📁 day36-ensemble-learning/
├── code/
│   ├── bagging_random_forest.py
│   ├── xgboost_vs_stacking.py 
│   ├── catboost_classifier_blending.py 
│   └── voting_classifier_comparison.py
|       
├── images/
│   ├── bagging_vs_boosting.png
│   ├── random_forest_bagging_architecture.png
│   ├── stacking_architecture_diagram.png
│   ├── xgboost_vs_stacking_chart.png
│   ├── blending_vs_stacking_visual.png
│   ├── catboost_blending_flow.png
│   ├── voting_classifier_types.png
│   └── ensemble_methods_overview.png
└── README.md
```
🖼️ Visual Understanding
🔹 Bagging with Random Forest

🔹 Boosting vs Bagging

🔹 Stacking Architecture

🔹 Blending vs Stacking

🔹 Voting Classifier (Hard vs Soft)

🧪 Code Highlights
✅ Bagging with Random Forest
```python
Copy
Edit
model = RandomForestClassifier(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)
```
✅ Stacking Models
```python
estimators = [
    ('lr', LogisticRegression()),
    ('svc', SVC(probability=True))
]
clf = StackingClassifier(estimators=estimators, final_estimator=GradientBoostingClassifier())
```
✅ Blending with CatBoost
```python
cat = CatBoostClassifier(verbose=0)
cat.fit(X_train_blend, y_train)
📊 Why Use Ensembles?
✅ Reduce Variance
✅ Reduce Bias
✅ Boost Accuracy
✅ Improve Generalization
```
🔗 Previous Topics
![🔁 Day 35: Model Optimization & Tuning](https://github.com/Shadabur-Rahaman/Daily-ML-Dose/edit/main/day35-model-optimization)
![🧠 Day 34 → Advanced Boosting (XGBoost, CatBoost)](https://github.com/Shadabur-Rahaman/Daily-ML-Dose/tree/main/day34-advanced-boosting)
![💡 Day 33 → Variational Autoencoders (VAEs)](https://github.com/Shadabur-Rahaman/Daily-ML-Dose/tree/main/day33-variational-autoencoders)


🔗 GitHub Anchor
📁 day36-ensemble-learning

🔥 Summary
Ensemble methods are among the most powerful tools in the ML toolbox. Whether you're building competition-grade models or optimizing production pipelines — mastering them is a must! 💪

📅 Stay Tuned
📌 Next Up: Day 37 – 🧠 Explainable AI (XAI): Building Trust in ML Models

#️⃣ #MachineLearning #EnsembleLearning #RandomForest #Boosting #Stacking #Blending #VotingClassifier #DataScience #DailyMLDose
