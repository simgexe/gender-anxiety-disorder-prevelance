import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler

# Veri setini yükle
file_path = 'prevalence-of-anxiety-disorders-males-vs-females.csv'
data = pd.read_csv(file_path)

# Veri setinin ilk birkaç satırını görüntüleyerek yapısını anlamak
data_head = data.head()
print(data_head)

# Eksik verileri kontrol etme
missing_data = data.isnull().sum()

# Eksik verileri görüntüleme
print(missing_data)

# Eksik değerleri içeren satırları çıkarma
cleaned_data = data.dropna(subset=[
    'Prevalence - Anxiety disorders - Sex: Male - Age: Age-standardized (Percent)',
    'Prevalence - Anxiety disorders - Sex: Female - Age: Age-standardized (Percent)'
])

# Eksik değerlerin tekrar kontrol edilmesi
cleaned_missing_data = cleaned_data.isnull().sum()
print(cleaned_missing_data)


cleaned_data = cleaned_data.drop(columns=['Code', 'Continent'])

# Eksik "Population (historical estimates)" değerlerini medyan ile doldurma
cleaned_data['Population (historical estimates)'] = cleaned_data['Population (historical estimates)'].fillna(cleaned_data['Population (historical estimates)'].median())

print(cleaned_data.head())

# Entity (ülke) sütununu one-hot encoding ile dönüştürme
encoder = OneHotEncoder()
entity_encoded = encoder.fit_transform(cleaned_data[['Entity']]).toarray()

# Yeni sütun isimlerini alma
encoded_columns = encoder.get_feature_names_out(['Entity'])

# One-hot encoding ile oluşan sütunları veri setine ekleme
encoded_df = pd.DataFrame(entity_encoded, columns=encoded_columns)
cleaned_data = pd.concat([cleaned_data.reset_index(drop=True), encoded_df], axis=1)

# Gereksiz sütunları çıkarma
cleaned_data = cleaned_data.drop(columns=['Entity', 'index'])

# Bağımsız ve bağımlı değişkenleri belirleme
X = cleaned_data.drop(columns=[
    'Prevalence - Anxiety disorders - Sex: Male - Age: Age-standardized (Percent)',
    'Prevalence - Anxiety disorders - Sex: Female - Age: Age-standardized (Percent)'
])
y_male = cleaned_data['Prevalence - Anxiety disorders - Sex: Male - Age: Age-standardized (Percent)']
y_female = cleaned_data['Prevalence - Anxiety disorders - Sex: Female - Age: Age-standardized (Percent)']


X_train, X_test, y_male_train, y_male_test, y_female_train, y_female_test = train_test_split(X, y_male, y_female, test_size=0.2, random_state=42)


# Lineer regresyon modelini oluştur ve eğit
model_male = LinearRegression()
model_male.fit(X_train, y_male_train)

model_female = LinearRegression()
model_female.fit(X_train, y_female_train)

# Test seti üzerinde tahmin yap ve hata metriklerini hesapla
y_male_pred = model_male.predict(X_test)
y_female_pred = model_female.predict(X_test)

mae_male = mean_absolute_error(y_male_test, y_male_pred)
mae_female = mean_absolute_error(y_female_test, y_female_pred)

print(mae_male, mae_female)
# Erkek Anksiyete yaygınlığı MAE =0.050
# Kadın Anksiyete yaygınlığı MAE =0.089

#############################DERİN ÖĞRENME MODELİ##################################

# Veriyi ölçeklendirme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim ve test setlerine tekrar ayırma
X_train, X_test, y_male_train, y_male_test, y_female_train, y_female_test = train_test_split(X_scaled, y_male, y_female, test_size=0.2, random_state=42)

# Modeli oluşturma
def create_model():
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# Erkek anksiyete yaygınlığı için model
model_male = create_model()
history_male = model_male.fit(X_train, y_male_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Kadın anksiyete yaygınlığı için model
model_female = create_model()
history_female = model_female.fit(X_train, y_female_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Test seti üzerinde tahmin yapma ve hata metriklerini hesaplama
y_male_pred = model_male.predict(X_test)
y_female_pred = model_female.predict(X_test)

mae_male = mean_absolute_error(y_male_test, y_male_pred)
mae_female = mean_absolute_error(y_female_test, y_female_pred)

print(mae_male, mae_female)
