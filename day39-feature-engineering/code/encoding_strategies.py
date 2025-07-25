# Encoding categorical variables: One-Hot, Label, Ordinal
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder

# Sample categorical data
df = pd.DataFrame({
    'city': ['Delhi', 'Mumbai', 'Delhi', 'Bangalore', 'Chennai'],
    'experience_level': ['Junior', 'Senior', 'Mid', 'Mid', 'Senior']
})

# One-Hot Encoding
ohe = OneHotEncoder(sparse=False)
one_hot = ohe.fit_transform(df[['city']])
df_ohe = pd.DataFrame(one_hot, columns=ohe.get_feature_names_out(['city']))
print("One-Hot Encoded:\n", df_ohe)

# Label Encoding (suitable for target)
le = LabelEncoder()
df['experience_encoded'] = le.fit_transform(df['experience_level'])
print("\nLabel Encoded:\n", df[['experience_level', 'experience_encoded']])

# Ordinal Encoding (if order matters)
ordinal_map = [['Junior', 'Mid', 'Senior']]
ord_enc = OrdinalEncoder(categories=ordinal_map)
df['experience_ordinal'] = ord_enc.fit_transform(df[['experience_level']])
print("\nOrdinal Encoded:\n", df[['experience_level', 'experience_ordinal']])
