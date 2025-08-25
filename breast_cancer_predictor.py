import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

data = load_breast_cancer()
X = data.data
y = data.target
df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y

df

X_train, X_test, y_train, y_test = train_test_split(X, y)

print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)

print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_test: {y_test.shape}")
print(f"X_val: {X_val.shape}")
print(f"y_val: {y_val.shape}")

for depth in range(1, 11):
  for min in range(2, 20):

    model = DecisionTreeClassifier(max_depth=depth, min_samples_split=min)
    model.fit(X_train, y_train)

    y_pred_val = model.predict(X_val)
    print(f"depth: {depth}, min_value: {min}, accuracy: {accuracy_score(y_val, y_pred_val)}")

    #depth = 5, min value=1, min_value_split = 6

X_final_train = np.vstack([X_train, X_val])
y_final_train = np.concatenate([y_train, y_val])

model = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_leaf=1, min_samples_split=6)
model.fit(X_final_train, y_final_train)

y_pred = model.predict(X_test)
print(f"model accuracy: {accuracy_score(y_test, y_pred)}")

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(35,25))
plot_tree(model, feature_names=data.feature_names, class_names=data.target_names, filled=True, impurity=True)
plt.show()

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(18, 12))  
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

from sklearn.model_selection import GridSearchCV

param_grid = {'max_depth': [2,3,4,5,6,7,8,9,10,11],
              'min_samples_leaf': [2,3,4,5,6,7,8,9,10,11],
              'criterion': ['gini', 'entropy']

}
grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    scoring='recall',
)

grid_search.fit(X_train, y_train)

best_model = grid_search.predict(X_test)

print(f"accuracy = {accuracy_score(y_test, best_model)}")

print(classification_report(y_test, best_model))

print(classification_report(y_test, y_pred))

feature_imp_df = pd.DataFrame({'feature': data.feature_names,'importance': model.feature_importances_})

feature_imp_df = feature_imp_df.sort_values(by='importance', ascending=False)

top_ten = feature_imp_df['feature'].head(10).tolist()
top_ten

X_df = pd.DataFrame(X, columns=data.feature_names)
X_df_top = X_df[top_ten]
X_df_top

X_train_top, X_test_top, y_train_top, y_test_top = train_test_split(X_df_top, y, test_size=0.2, random_state=42)

X_train_top, X_val_top, y_train_top, y_val_top = train_test_split(X_train_top, y_train_top,test_size=0.2, random_state=42)

print(f"X_train_top: {X_train_top.shape}")
print(f"y_train_top: {y_train_top.shape}")
print(f"X_val_top: {X_val_top.shape}")
print(f"y_val_top: {y_val_top.shape}")
print(f"X_test_top: {X_test_top.shape}")
print(f"y_test_top: {y_test_top.shape}")

from sklearn.metrics import recall_score

for depth in range(1, 11):
  for min in range(2, 11):
    model = DecisionTreeClassifier(max_depth = depth, min_samples_leaf=min)
    model.fit(X_train_top, y_train_top)

    y_pred = model.predict(X_test_top)
    recall = recall_score(y_test_top, y_pred)
    acc = accuracy_score(y_test_top, y_pred)

    print(f"depth: {depth}, min:{min}, recall: {recall}, acc: {acc}")

    #depth 5, min 3

X_train_top_final = np.vstack([X_train_top, X_test_top])
y_train_top_final = np.concatenate([y_train_top, y_test_top])

model_top = DecisionTreeClassifier(max_depth=4, min_samples_leaf=2)
model_top.fit(X_train_top_final, y_train_top_final)

y_pred_top_final = model_top.predict(X_val_top)

print(accuracy_score(y_val_top, y_pred_top_final))
print(classification_report(y_val_top, y_pred_top_final))

from sklearn.ensemble import RandomForestClassifier
model_random_forest = RandomForestClassifier(bootstrap=True, oob_score=True, min_samples_split=2, max_features='sqrt', class_weight='balanced')
model_random_forest.fit(X_train_top, y_train_top)

model_random_forest.oob_score_

#feat_importances = pd.Series(model_random_forest.feature_importances_, index=data.feature_names)
#feat_importances.sort_values(ascending=False).plot(kind='bar')

y_pred_random_forest = model_random_forest.predict(X_test_top)
print(classification_report(y_test_top, y_pred_random_forest))

y_val_pred_random_forest = model_random_forest.predict(X_val_top)
print(classification_report(y_val_top, y_val_pred_random_forest))

from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2,5,10],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, scoring='recall', n_jobs=-1, cv=5)
grid.fit(X_train_top, y_train_top)

print("Best params:", grid.best_params_)
print("Best score:", grid.best_score_)

#import pickle

#with open('cancer_model.pkl', 'wb') as f:
  #pickle.dump(model_top, f)

from sklearn.ensemble import GradientBoostingClassifier

model_gradient_boost = GradientBoostingClassifier()
model_gradient_boost.fit(X_train_top, y_train_top)

y_val_pred_gradient_boost = model.predict(X_val_top)

from sklearn.metrics import classification_report
print(classification_report(y_val_top, y_val_pred_gradient_boost))

y_pred_gradient_boost = model.predict(X_test_top)

print(classification_report(y_test_top, y_pred_gradient_boost))

y_pred_random_forest = model_random_forest.predict(X_test_top)
print(classification_report(y_test_top, y_pred_random_forest))

from xgboost import XGBClassifier

xgb_model = XGBClassifier()

xgb_model.fit(X_train_top, y_train_top)

y_val_pred_xgboost = xgb_model.predict(X_val_top)
print(classification_report(y_val_top, y_val_pred_xgboost))

y_pred_xgboost = xgb_model.predict(X_test_top)
print(classification_report(y_test_top, y_pred_xgboost))
