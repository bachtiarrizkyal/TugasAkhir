import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ===== DATA EXPLORE =====
github_url = "https://raw.githubusercontent.com/bachtiarrizkyal/TugasAkhir/refs/heads/main/USDAnew.xlsx"
df_usda = pd.read_excel(github_url)
#df_usda = pd.read_excel('/content/USDAnew.xlsx')

# Melihat informasi dataset
print("Informasi Dataset:")
print(df_usda.info())

# Melihat jumlah data dan kolom
print("\nBentuk Dataset:", df_usda.shape)

# Melihat beberapa data awal
print("\nSample Data:")
print(df_usda.head())

# Melihat statistik deskriptif
print("\nStatistik Deskriptif:")
print(df_usda.describe())

# Melihat nilai yang hilang
print("\nJumlah nilai yang hilang per kolom:")
print(df_usda.isnull().sum())

# Melihat duplikat berdasarkan FoodName
print("\nJumlah duplikat berdasarkan FoodName:", df_usda.duplicated(subset=['FoodName']).sum())

# Melihat distribusi kategori makanan
print("\nDistribusi Kategori Makanan:")
print(df_usda['FoodCategory'].value_counts())

# Melihat distribusi kelompok makanan
print("\nDistribusi Kelompok Makanan:")
print(df_usda['FoodGroups'].value_counts())

# ===== DATA PREPROCESSING ======
def preprocess_data(df):
    # Menghapus duplikat berdasarkan FoodName
    df_clean = df.drop_duplicates(subset=['FoodName'])
    print(f"Jumlah data setelah menghapus duplikat: {df_clean.shape[0]}")

    # Mengisi nilai kosong dengan 0
    df_clean.fillna(0, inplace=True)

    # Memilih kolom yang akan digunakan
    selected_columns = ['FoodID', 'FoodGroups', 'FoodName', 'Energi', 'Protein',
                       'Serat', 'VitaminC', 'Kalium', 'Magnesium', 'Kalsium',
                       'Besi', 'GulaTotal', 'Natrium', 'LemakJenuh', 'Kolesterol']

    df_clean = df_clean[selected_columns]

    # Melihat statistik setelah preprocessing
    print("\nStastistik setelah preprocessing:")
    print(df_clean.describe())

    # Simpan salinan data asli sebelum normalisasi
    df_original = df_clean.copy()

    # Normalisasi data nutrisi
    non_numeric_cols = ['FoodID', 'FoodGroups', 'FoodName']
    numeric_cols = [col for col in df_clean.columns if col not in non_numeric_cols]

    scaler = MinMaxScaler()
    df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])

    print("\nData setelah normalisasi (MinMaxScaler):")
    print(df_clean[numeric_cols].describe())

    return df_clean, scaler, df_original, numeric_cols

df_preprocessed, scaler, df_original, nutrient_cols = preprocess_data(df_usda)

# ===== USER DATA PROCESSING =====
def calculate_bmi(weight, height):
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    return bmi

def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Berat Badan Kurang (Underweight)"
    elif 18.5 <= bmi < 23:
        return "Berat Badan Normal"
    elif 23 <= bmi < 25:
        return "Berat Badan Lebih (Overweight)"
    elif 25 <= bmi < 30:
        return "Obesitas I"
    else:
        return "Obesitas II"

def calculate_ideal_weight(gender, height):
    if gender.lower() == 'laki-laki':
        return (height - 100) - ((height - 100) * 0.1)
    else:
        return (height - 100) - ((height - 100) * 0.15)

def calculate_energy_needs(gender, age, height, ideal_weight):
    if gender.lower() == 'laki-laki':
        bmr = 66.5 + (13.75 * ideal_weight) + (5.003 * height) - (6.75 * age)
    else:
        bmr = 655.1 + (9.563 * ideal_weight) + (1.850 * height) - (4.676 * age)

    energy_needs = bmr * 1.55
    return energy_needs

def get_age_category(age):
    if 10 <= age <= 18:
        return "remaja"
    elif 19 <= age <= 64:
        return "dewasa"
    else:
        return "lansia"

def get_nutrition_requirements(gender, age, energy_needs, diseases):
    # Mendapatkan kategori usia
    age_category = get_age_category(age)

    # Indeks untuk nilai AKG berdasarkan jenis kelamin dan usia
    idx = 0

    # Remaja laki-laki (10-18 tahun)
    if gender.lower() == 'laki-laki' and age_category == 'remaja':
        if 10 <= age <= 12:
            idx = 0
        elif 13 <= age <= 15:
            idx = 1
        elif 16 <= age <= 18:
            idx = 2

    # Dewasa laki-laki (19-64 tahun)
    elif gender.lower() == 'laki-laki' and age_category == 'dewasa':
        if 19 <= age <= 29:
            idx = 3
        elif 30 <= age <= 49:
            idx = 4
        elif 50 <= age <= 64:
            idx = 5

    # Lansia laki-laki (>64 tahun)
    elif gender.lower() == 'laki-laki' and age_category == 'lansia':
        if 65 <= age <= 80:
            idx = 6
        else:
            idx = 7

    # Remaja perempuan (10-18 tahun)
    elif gender.lower() == 'perempuan' and age_category == 'remaja':
        if 10 <= age <= 12:
            idx = 8
        elif 13 <= age <= 15:
            idx = 9
        elif 16 <= age <= 18:
            idx = 10

    # Dewasa perempuan (19-64 tahun)
    elif gender.lower() == 'perempuan' and age_category == 'dewasa':
        if 19 <= age <= 29:
            idx = 11
        elif 30 <= age <= 49:
            idx = 12
        elif 50 <= age <= 64:
            idx = 13

    # Lansia perempuan (>64 tahun)
    elif gender.lower() == 'perempuan' and age_category == 'lansia':
        if 65 <= age <= 80:
            idx = 14
        else:
            idx = 15

    # AKG Kondisi Normal
    nutrition_req = {
        'GulaTotal': (10 * energy_needs / 100) / 4,
        'Serat': [28, 34, 37, 37, 36, 30, 25, 22, 27, 29, 29, 32, 30, 25, 22, 20][idx],
        'Protein': [50, 70, 75, 65, 65, 65, 64, 64, 55, 65, 65, 60, 60, 60, 58, 58][idx],
        'LemakJenuh': [30, 30, 30, 30, 30, 30, 30, 30, 20, 20, 20, 20, 20, 20, 20, 20][idx],
        'VitaminC': [50, 75, 90, 90, 90, 90, 90, 90, 50, 65, 75, 75, 75, 75, 75, 75][idx],
        'Magnesium': [160, 225, 270, 360, 360, 360, 350, 350, 170, 220, 230, 330, 340, 340, 320, 320][idx],
        'Natrium': [1300, 1500, 1700, 1500, 1500, 1300, 1100, 1000, 1400, 1500, 1600, 1500, 1500, 1400, 1200, 1000][idx],
        'Kalsium': [1200, 1200, 1200, 1000, 1000, 1200, 1200, 1200, 1200, 1200, 1200, 1000, 1000, 1200, 1200, 1200][idx],
        'Kalium': [3900, 4800, 5300, 4700, 4700, 4700, 4700, 4700, 4400, 4800, 5000, 4700, 4700, 4700, 4700, 4700][idx],
        'Besi': [8, 11, 11, 9, 9, 9, 9, 9, 8, 15, 15, 18, 18, 8, 8, 8][idx],
        'Kolesterol': [300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300][idx],
        'Energi': energy_needs
    }

    # AKG Pengidap Penyakit
    if diseases:
        # Pengidap Diabetes
        if 'diabetes' in diseases and 'hipertensi' not in diseases and 'cardiovascular disease' not in diseases:
            nutrition_req.update({
                'Serat': [32.2, 39.1, 42.55, 42.55, 41.4, 34.5, 28.75, 25.3, 31.05, 33.35, 33.35, 36.8, 34.5, 28.75, 25.3, 23][idx],
                'Protein': (20 * energy_needs / 100) / 4,
                'VitaminC': [57.5, 86.25, 103.5, 103.5, 103.5, 103.5, 103.5, 103.5, 57.5, 74.75, 86.25, 86.25, 86.25, 86.25, 86.25, 86.25][idx],
                'GulaTotal': (5 * energy_needs / 100) / 4,
                'LemakJenuh': (7 * energy_needs / 100) / 9,
                'Natrium': [1105, 1275, 1445, 1275, 1275, 1105, 935, 850, 1190, 1275, 1360, 1275, 1275, 1190, 1020, 850][idx],
                'Kolesterol': 200
            })

        # Pengidap Hipertensi
        elif 'hipertensi' in diseases and 'diabetes' not in diseases and 'cardiovascular disease' not in diseases:
            nutrition_req.update({
                'Serat': [32.2, 39.1, 42.55, 42.55, 41.4, 34.5, 28.75, 25.3, 31.05, 33.35, 33.35, 36.8, 34.5, 28.75, 25.3, 23][idx],
                'Magnesium': [184, 258.75, 310.5, 414, 414, 414, 402.5, 402.5, 195.5, 253, 264.5, 379.5, 391, 391, 368, 368][idx],
                'Kalsium': [1380, 1380, 1380, 1150, 1150, 1380, 1380, 1380, 1380, 1380, 1380, 1150, 1150, 1380, 1380, 1380][idx],
                'Kalium': [4485, 5520, 6095, 5405, 5405, 5405, 5405, 5405, 5060, 5520, 5750, 5405, 5405, 5405, 5405, 5405][idx],
                'LemakJenuh': (7 * energy_needs / 100) / 9,
                'Natrium': [1105, 1275, 1445, 1275, 1275, 1105, 935, 850, 1190, 1275, 1360, 1275, 1275, 1190, 1020, 850][idx],
                'Kolesterol': 200
            })

        # Pengidap Cardiovascular Disease
        elif 'cardiovascular disease' in diseases and 'diabetes' not in diseases and 'hipertensi' not in diseases:
            nutrition_req.update({
                'Serat': [32.2, 39.1, 42.55, 42.55, 41.4, 34.5, 28.75, 25.3, 31.05, 33.35, 33.35, 36.8, 34.5, 28.75, 25.3, 23][idx],
                'Protein': (20 * energy_needs / 100) / 4,
                'Besi': [9.2, 12.65, 12.65, 10.35, 10.35, 10.35, 10.35, 10.35, 9.2, 17.25, 17.25, 20.7, 20.7, 9.2, 9.2, 9.2][idx],
                'LemakJenuh': (7 * energy_needs / 100) / 9,
                'Natrium': [1105, 1275, 1445, 1275, 1275, 1105, 935, 850, 1190, 1275, 1360, 1275, 1275, 1190, 1020, 850][idx],
                'Kolesterol': 200
            })

        # Pengidap Diabetes dan Hipertensi
        elif 'diabetes' in diseases and 'hipertensi' in diseases and 'cardiovascular disease' not in diseases:
            nutrition_req.update({
                'Serat': [32.2, 39.1, 42.55, 42.55, 41.4, 34.5, 28.75, 25.3, 31.05, 33.35, 33.35, 36.8, 34.5, 28.75, 25.3, 23][idx],
                'Protein': (20 * energy_needs / 100) / 4,
                'VitaminC': [57.5, 86.25, 103.5, 103.5, 103.5, 103.5, 103.5, 103.5, 57.5, 74.75, 86.25, 86.25, 86.25, 86.25, 86.25, 86.25][idx],
                'Magnesium': [184, 258.75, 310.5, 414, 414, 414, 402.5, 402.5, 195.5, 253, 264.5, 379.5, 391, 391, 368, 368][idx],
                'Kalsium': [1380, 1380, 1380, 1150, 1150, 1380, 1380, 1380, 1380, 1380, 1380, 1150, 1150, 1380, 1380, 1380][idx],
                'Kalium': [4485, 5520, 6095, 5405, 5405, 5405, 5405, 5405, 5060, 5520, 5750, 5405, 5405, 5405, 5405, 5405][idx],
                'GulaTotal': (5 * energy_needs / 100) / 4,
                'LemakJenuh': (7 * energy_needs / 100) / 9,
                'Natrium': [1105, 1275, 1445, 1275, 1275, 1105, 935, 850, 1190, 1275, 1360, 1275, 1275, 1190, 1020, 850][idx],
                'Kolesterol': 200
            })

        # Pengidap Diabetes dan Cardiovascular Disease
        elif 'diabetes' in diseases and 'cardiovascular disease' in diseases and 'hipertensi' not in diseases:
            nutrition_req.update({
                'Serat': [32.2, 39.1, 42.55, 42.55, 41.4, 34.5, 28.75, 25.3, 31.05, 33.35, 33.35, 36.8, 34.5, 28.75, 25.3, 23][idx],
                'Protein': (20 * energy_needs / 100) / 4,
                'VitaminC': [57.5, 86.25, 103.5, 103.5, 103.5, 103.5, 103.5, 103.5, 57.5, 74.75, 86.25, 86.25, 86.25, 86.25, 86.25, 86.25][idx],
                'Besi': [9.2, 12.65, 12.65, 10.35, 10.35, 10.35, 10.35, 10.35, 9.2, 17.25, 17.25, 20.7, 20.7, 9.2, 9.2, 9.2][idx],
                'GulaTotal': (5 * energy_needs / 100) / 4,
                'LemakJenuh': (7 * energy_needs / 100) / 9,
                'Natrium': [1105, 1275, 1445, 1275, 1275, 1105, 935, 850, 1190, 1275, 1360, 1275, 1275, 1190, 1020, 850][idx],
                'Kolesterol': 200
            })

        # Pengidap Hipertensi dan Cardiovascular Disease
        elif 'hipertensi' in diseases and 'cardiovascular disease' in diseases and 'diabetes' not in diseases:
            nutrition_req.update({
                'Serat': [32.2, 39.1, 42.55, 42.55, 41.4, 34.5, 28.75, 25.3, 31.05, 33.35, 33.35, 36.8, 34.5, 28.75, 25.3, 23][idx],
                'Protein': (20 * energy_needs / 100) / 4,
                'Magnesium': [184, 258.75, 310.5, 414, 414, 414, 402.5, 402.5, 195.5, 253, 264.5, 379.5, 391, 391, 368, 368][idx],
                'Kalsium': [1380, 1380, 1380, 1150, 1150, 1380, 1380, 1380, 1380, 1380, 1380, 1150, 1150, 1380, 1380, 1380][idx],
                'Kalium': [4485, 5520, 6095, 5405, 5405, 5405, 5405, 5405, 5060, 5520, 5750, 5405, 5405, 5405, 5405, 5405][idx],
                'Besi': [9.2, 12.65, 12.65, 10.35, 10.35, 10.35, 10.35, 10.35, 9.2, 17.25, 17.25, 20.7, 20.7, 9.2, 9.2, 9.2][idx],
                'LemakJenuh': (7 * energy_needs / 100) / 9,
                'Natrium': [1105, 1275, 1445, 1275, 1275, 1105, 935, 850, 1190, 1275, 1360, 1275, 1275, 1190, 1020, 850][idx],
                'Kolesterol': 200
            })

        # Pengidap Diabetes, Hipertensi, dan Cardiovascular Disease
        elif 'diabetes' in diseases and 'hipertensi' in diseases and 'cardiovascular disease' in diseases:
            nutrition_req.update({
                'Serat': [32.2, 39.1, 42.55, 42.55, 41.4, 34.5, 28.75, 25.3, 31.05, 33.35, 33.35, 36.8, 34.5, 28.75, 25.3, 23][idx],
                'Protein': (20 * energy_needs / 100) / 4,
                'VitaminC': [57.5, 86.25, 103.5, 103.5, 103.5, 103.5, 103.5, 103.5, 57.5, 74.75, 86.25, 86.25, 86.25, 86.25, 86.25, 86.25][idx],
                'Magnesium': [184, 258.75, 310.5, 414, 414, 414, 402.5, 402.5, 195.5, 253, 264.5, 379.5, 391, 391, 368, 368][idx],
                'Kalsium': [1380, 1380, 1380, 1150, 1150, 1380, 1380, 1380, 1380, 1380, 1380, 1150, 1150, 1380, 1380, 1380][idx],
                'Kalium': [4485, 5520, 6095, 5405, 5405, 5405, 5405, 5405, 5060, 5520, 5750, 5405, 5405, 5405, 5405, 5405][idx],
                'Besi': [9.2, 12.65, 12.65, 10.35, 10.35, 10.35, 10.35, 10.35, 9.2, 17.25, 17.25, 20.7, 20.7, 9.2, 9.2, 9.2][idx],
                'GulaTotal': (5 * energy_needs / 100) / 4,
                'LemakJenuh': (7 * energy_needs / 100) / 9,
                'Natrium': [1105, 1275, 1445, 1275, 1275, 1105, 935, 850, 1190, 1275, 1360, 1275, 1275, 1190, 1020, 850][idx],
                'Kolesterol': 200
            })

    return nutrition_req

def process_user_data(gender, age, height, weight, diseases):
    # Mendapatkan kategori usia
    age_category = get_age_category(age)

    # Menggabungkan usia dengan kategori
    age_with_category = f"{age} tahun ({age_category})"

    # Menghitung BMI
    bmi = calculate_bmi(weight, height)
    bmi_category = get_bmi_category(bmi)

    # Menghitung Berat Badan Ideal
    ideal_weight = calculate_ideal_weight(gender, height)

    # Menghitung kebutuhan energi
    energy_needs = calculate_energy_needs(gender, age, height, ideal_weight)

    # Mendapatkan kebutuhan nutrisi
    nutrition_req = get_nutrition_requirements(gender, age, energy_needs, diseases)

    # Menyusun informasi pengguna
    user_info = {
        'gender': gender,
        'age': age,
        'age_category': age_category,
        'age_with_category': age_with_category,
        'height': height,
        'weight': weight,
        'diseases': diseases,
        'bmi': bmi,
        'bmi_category': bmi_category,
        'ideal_weight': ideal_weight,
        'energy_needs': energy_needs,
        'nutrition_req': nutrition_req
    }

    return user_info

# ===== ITEM DATA PROCESSING =====
def label_food_by_nutrition(df, diseases, df_original):
    # Ambil data nutrisi dari df
    df_nutrition = df.copy()

    # Tambahkan kolom untuk nilai nutrisi asli
    for col in nutrient_cols:
        df_nutrition[f'original_{col}'] = df_original[col].values

    # Mendefinisikan batas untuk nutrisi tinggi
    high_thresholds = {
        'Protein': 10,
        'Serat': 6,
        'VitaminC': 15,
        'Kalium': 600,
        'Magnesium': 48,
        'Kalsium': 300,
        'Besi': 2.4
    }

    # Mendefinisikan batas untuk nutrisi rendah
    low_thresholds = {
        'GulaTotal': 15,
        'Natrium': 600,
        'LemakJenuh': 5,
        'Kolesterol': 100
    }

    # Mengidentifikasi makanan tinggi nutrisi tertentu
    df_nutrition['high_protein'] = df_nutrition['original_Protein'] >= high_thresholds['Protein']
    df_nutrition['high_fiber'] = df_nutrition['original_Serat'] >= high_thresholds['Serat']
    df_nutrition['high_vitaminc'] = df_nutrition['original_VitaminC'] >= high_thresholds['VitaminC']
    df_nutrition['high_potassium'] = df_nutrition['original_Kalium'] >= high_thresholds['Kalium']
    df_nutrition['high_magnesium'] = df_nutrition['original_Magnesium'] >= high_thresholds['Magnesium']
    df_nutrition['high_calcium'] = df_nutrition['original_Kalsium'] >= high_thresholds['Kalsium']
    df_nutrition['high_iron'] = df_nutrition['original_Besi'] >= high_thresholds['Besi']

    # Mengidentifikasi makanan rendah nutrisi tertentu
    df_nutrition['low_sugar'] = df_nutrition['original_GulaTotal'] <= low_thresholds['GulaTotal']
    df_nutrition['low_sodium'] = df_nutrition['original_Natrium'] <= low_thresholds['Natrium']
    df_nutrition['low_sat_fat'] = df_nutrition['original_LemakJenuh'] <= low_thresholds['LemakJenuh']
    df_nutrition['low_cholesterol'] = df_nutrition['original_Kolesterol'] <= low_thresholds['Kolesterol']

    # Label makanan berdasarkan riwayat penyakit
    if not diseases:
        # Tidak ada riwayat penyakit, makanan sehat secara umum
        df_nutrition['suitable'] = (
            df_nutrition['low_sugar'] &
            df_nutrition['low_sodium'] &
            df_nutrition['low_sat_fat'] &
            df_nutrition['low_cholesterol']
        ).astype(int)
    else:
        # Pengidap Diabetes
        if 'diabetes' in diseases and 'hipertensi' not in diseases and 'cardiovascular disease' not in diseases:
            df_nutrition['suitable'] = (
                (df_nutrition['high_protein'] | df_nutrition['high_fiber'] | df_nutrition['high_vitaminc']) &
                (df_nutrition['low_sugar'] & df_nutrition['low_sodium'] & df_nutrition['low_sat_fat'] & df_nutrition['low_cholesterol'])
            ).astype(int)

        # Pengidap Hipertensi
        elif 'hipertensi' in diseases and 'diabetes' not in diseases and 'cardiovascular disease' not in diseases:
            df_nutrition['suitable'] = (
                (df_nutrition['high_fiber'] | df_nutrition['high_potassium'] | df_nutrition['high_magnesium'] | df_nutrition['high_calcium']) &
                (df_nutrition['low_sodium'] & df_nutrition['low_sat_fat'] & df_nutrition['low_cholesterol'])
            ).astype(int)

        # Pengidap Cardiovascular Disease
        elif 'cardiovascular disease' in diseases and 'diabetes' not in diseases and 'hipertensi' not in diseases:
            df_nutrition['suitable'] = (
                (df_nutrition['high_protein'] | df_nutrition['high_fiber'] | df_nutrition['high_iron']) &
                (df_nutrition['low_sodium'] & df_nutrition['low_sat_fat'] & df_nutrition['low_cholesterol'])
            ).astype(int)

        # Pengidap Diabetes dan Hipertensi
        elif 'diabetes' in diseases and 'hipertensi' in diseases and 'cardiovascular disease' not in diseases:
            df_nutrition['suitable'] = (
                (df_nutrition['high_protein'] | df_nutrition['high_fiber'] | df_nutrition['high_potassium'] |
                 df_nutrition['high_magnesium'] | df_nutrition['high_calcium'] | df_nutrition['high_vitaminc']) &
                (df_nutrition['low_sugar'] & df_nutrition['low_sodium'] & df_nutrition['low_sat_fat'] & df_nutrition['low_cholesterol'])
            ).astype(int)

        # Pengidap Diabetes dan Cardiovascular Disease
        elif 'diabetes' in diseases and 'cardiovascular disease' in diseases and 'hipertensi' not in diseases:
            df_nutrition['suitable'] = (
                (df_nutrition['high_protein'] | df_nutrition['high_fiber'] | df_nutrition['high_iron'] | df_nutrition['high_vitaminc']) &
                (df_nutrition['low_sugar'] & df_nutrition['low_sodium'] & df_nutrition['low_sat_fat'] & df_nutrition['low_cholesterol'])
            ).astype(int)

        # Pengidap Hipertensi dan Cardiovascular Disease
        elif 'hipertensi' in diseases and 'cardiovascular disease' in diseases and 'diabetes' not in diseases:
            df_nutrition['suitable'] = (
                (df_nutrition['high_fiber'] | df_nutrition['high_protein'] | df_nutrition['high_iron'] |
                 df_nutrition['high_potassium'] | df_nutrition['high_magnesium'] | df_nutrition['high_calcium']) &
                (df_nutrition['low_sodium'] & df_nutrition['low_sat_fat'] & df_nutrition['low_cholesterol'])
            ).astype(int)

        # Pengidap Diabetes, Hipertensi, dan Cardiovascular Disease
        elif 'diabetes' in diseases and 'hipertensi' in diseases and 'cardiovascular disease' in diseases:
            df_nutrition['suitable'] = (
                (df_nutrition['high_protein'] | df_nutrition['high_fiber'] | df_nutrition['high_vitaminc'] |
                 df_nutrition['high_potassium'] | df_nutrition['high_magnesium'] | df_nutrition['high_calcium'] |
                 df_nutrition['high_iron']) &
                (df_nutrition['low_sugar'] & df_nutrition['low_sodium'] & df_nutrition['low_sat_fat'] & df_nutrition['low_cholesterol'])
            ).astype(int)

    return df_nutrition

def prepare_food_data(df, user_info, df_original):
    # Label makanan berdasarkan nutrisi dan riwayat penyakit
    df_labeled = label_food_by_nutrition(df, user_info['diseases'], df_original)

    # Mengelompokkan makanan berdasarkan kelompok makanan
    df_main = df_labeled[df_labeled['FoodGroups'] == 'Makanan Utama']
    df_side = df_labeled[df_labeled['FoodGroups'] == 'Makanan Pendamping']
    df_drink_fruit = df_labeled[df_labeled['FoodGroups'] == 'Minuman dan Buah']

    return df_labeled, df_main, df_side, df_drink_fruit

def train_knn_model(df, user_info):
    # Menyiapkan fitur (X) dan target (y) - menggunakan nilai yang sudah dinormalisasi untuk pelatihan model
    X = df[['Energi', 'Protein', 'Serat', 'VitaminC', 'Kalium', 'Magnesium',
            'Kalsium', 'Besi', 'GulaTotal', 'Natrium', 'LemakJenuh', 'Kolesterol']]
    y = df['suitable']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'metric': ['euclidean']
    }

    knn_model_instance = KNeighborsClassifier()

    scorers = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
    }

    grid_search = GridSearchCV(estimator=knn_model_instance, param_grid=param_grid, cv=5,
                               scoring=scorers, refit='recall', n_jobs=-1, verbose=1)

    grid_search.fit(X_train, y_train)

    knn = grid_search.best_estimator_

    print(f"\n--- Hasil Tuning KNN ---")
    print(f"Hyperparameter terbaik (berdasarkan recall): {grid_search.best_params_}")
    print(f"Nilai recall terbaik pada cross-validation: {grid_search.best_score_:.4f}")
    print(f"Akurasi terbaik pada test set: {knn.score(X_test, y_test):.4f}")
    print(f"Precision terbaik pada test set: {precision_score(y_test, knn.predict(X_test), zero_division=0):.4f}")
    print(f"Recall terbaik pada test set: {recall_score(y_test, knn.predict(X_test), zero_division=0):.4f}")
    return knn, X, y, X_train, X_test, y_train, y_test

def get_recommendation(knn, df_labeled, user_info, df_original, n_recommendations=15):
    X_predict = df_labeled[['Energi', 'Protein', 'Serat', 'VitaminC', 'Kalium',
                        'Magnesium', 'Kalsium', 'Besi', 'GulaTotal', 'Natrium',
                        'LemakJenuh', 'Kolesterol']]

    # Prediksi dengan model KNN
    knn_predictions = knn.predict(X_predict)

    # Tambahkan hasil prediksi ke DataFrame
    df_labeled['knn_suitable'] = knn_predictions

    # Filter makanan yang diprediksi sesuai oleh KNNnutri
    suitable_foods = df_labeled[df_labeled['knn_suitable'] == 1].copy()

    if suitable_foods.empty:
        print("KNN tidak menemukan makanan yang sesuai. Menggunakan semua makanan...")
        suitable_foods = df_labeled.copy()

    # Mendapatkan kebutuhan nutrisi pengguna
    nutrition_req = user_info['nutrition_req']

    # Menghitung nilai per kalori untuk setiap makanan berdasarkan nilai nutrisi asli
    suitable_foods['value_per_calorie'] = 0

    # Menghitung nilai berdasarkan nutrisi yang dibutuhkan (semakin tinggi semakin baik)
    for nutrient in ['Protein', 'Serat', 'VitaminC', 'Kalium', 'Magnesium', 'Kalsium', 'Besi']:
        if nutrient in nutrition_req:
            original_nutrient = f'original_{nutrient}'
            # Menambah nilai jika nutrisi baik ada dan kebutuhan nutrisi > 0
            suitable_foods['value_per_calorie'] += suitable_foods[original_nutrient] / nutrition_req[nutrient] if nutrition_req[nutrient] > 0 else 0

    # Pengurangan nilai untuk nutrisi yang ingin diminimalkan (semakin tinggi semakin buruk)
    for nutrient in ['GulaTotal', 'Natrium', 'LemakJenuh', 'Kolesterol']:
        if nutrient in nutrition_req:
            original_nutrient = f'original_{nutrient}'
            # Mengurangi nilai jika nutrisi buruk ada dan kebutuhan nutrisi > 0
            suitable_foods['value_per_calorie'] -= suitable_foods[original_nutrient] / nutrition_req[nutrient] if nutrition_req[nutrient] > 0 else 0

    # Normalisasi nilai per kalori dengan membagi dengan energi asli makanan
    # Ini memberikan bobot nilai per unit energi
    suitable_foods['value_per_calorie'] = suitable_foods['value_per_calorie'] / suitable_foods['original_Energi']

    # Urutkan makanan berdasarkan nilai per kalori (dari tertinggi ke terendah)
    suitable_foods = suitable_foods.sort_values('value_per_calorie', ascending=False)

    # Batasan energi total harian - DIUBAH KE 80%-120%
    energy_limit_max_daily = nutrition_req['Energi'] * 1.2
    energy_limit_min_daily = nutrition_req['Energi'] * 0.8

    # Batasan nutrisi lain total harian - BATAS KHUSUS UNTUK NUTRISI TERTENTU
    nutrient_limits_max_daily = {}
    nutrient_limits_min_daily = {}
    moderate_restricted_nutrients = ['GulaTotal', 'Natrium']
    highly_restricted_nutrients = ['LemakJenuh', 'Kolesterol']

    for nutrient, value in nutrition_req.items():
        if nutrient != 'Energi':
            nutrient_limits_max_daily[nutrient] = value * 1.2
            if nutrient in moderate_restricted_nutrients:
                nutrient_limits_min_daily[nutrient] = value * 0.6
            elif nutrient in highly_restricted_nutrients:
                nutrient_limits_min_daily[nutrient] = value * 0.4
            else:
                nutrient_limits_min_daily[nutrient] = value * 0.8

    # Menyiapkan Dictionary untuk menyimpan rekomendasi per waktu makan
    meal_plan = {
        'Sarapan': {'Makanan Utama': [], 'Makanan Pendamping': [], 'Minuman dan Buah': []},
        'Makan Siang': {'Makanan Utama': [], 'Makanan Pendamping': [], 'Minuman dan Buah': []},
        'Makan Malam': {'Makanan Utama': [], 'Makanan Pendamping': [], 'Minuman dan Buah': []}
    }

    # Menyiapkan Dictionary untuk menyimpan total nutrisi per waktu makan
    meal_nutrition = {
        'Sarapan': {nutrient: 0.0 for nutrient in nutrition_req},
        'Makan Siang': {nutrient: 0.0 for nutrient in nutrition_req},
        'Makan Malam': {nutrient: 0.0 for nutrient in nutrition_req}
    }

    # Fungsi untuk menambahkan makanan ke rencana makan dan memperbarui total nutrisi (nested function)
    def add_food_to_meal(food, meal_time, food_group):
        meal_plan[meal_time][food_group].append({
            'FoodID': food['FoodID'], 'FoodName': food['FoodName'], 'Energi': food['original_Energi'],
            'Protein': food['original_Protein'], 'Serat': food['original_Serat'], 'VitaminC': food['original_VitaminC'],
            'Kalium': food['original_Kalium'], 'Magnesium': food['original_Magnesium'], 'Kalsium': food['original_Kalsium'],
            'Besi': food['original_Besi'], 'GulaTotal': food['original_GulaTotal'], 'Natrium': food['original_Natrium'],
            'LemakJenuh': food['original_LemakJenuh'], 'Kolesterol': food['original_Kolesterol']
        })
        for nutrient_key in nutrition_req:
            original_key = f'original_{nutrient_key}'
            if original_key in food:
                meal_nutrition[meal_time][nutrient_key] += food[original_key]

    # Menyiapkan taboo list untuk mencegah duplikasi makanan dalam rencana makan
    taboo_list = set()

    # Mendistribusikan makanan ke dalam rencana makan
    meal_times = ['Sarapan', 'Makan Siang', 'Makan Malam']
    food_groups = ['Makanan Utama', 'Makanan Pendamping', 'Minuman dan Buah']

    # Membagi target energi dan nutrisi per waktu makan
    energy_per_meal_max = {
        'Sarapan': energy_limit_max_daily * 0.33, 'Makan Siang': energy_limit_max_daily * 0.44, 'Makan Malam': energy_limit_max_daily * 0.33
    }
    energy_per_meal_min = {
        'Sarapan': energy_limit_min_daily * 0.24, 'Makan Siang': energy_limit_min_daily * 0.32, 'Makan Malam': energy_limit_min_daily * 0.24
    }

    nutrient_per_meal_max = {}
    nutrient_per_meal_min = {}
    for meal in meal_times:
        nutrient_per_meal_max[meal] = {}
        nutrient_per_meal_min[meal] = {}
        for nutrient, limit_max_daily in nutrient_limits_max_daily.items():
            limit_min_daily = nutrient_limits_min_daily[nutrient]
            if meal == 'Sarapan':
                nutrient_per_meal_max[meal][nutrient] = limit_max_daily * 0.3
                if nutrient in moderate_restricted_nutrients: nutrient_per_meal_min[meal][nutrient] = limit_min_daily * 0.18
                elif nutrient in highly_restricted_nutrients: nutrient_per_meal_min[meal][nutrient] = limit_min_daily * 0.12
                else: nutrient_per_meal_min[meal][nutrient] = limit_min_daily * 0.24
            elif meal == 'Makan Siang':
                nutrient_per_meal_max[meal][nutrient] = limit_max_daily * 0.4
                if nutrient in moderate_restricted_nutrients: nutrient_per_meal_min[meal][nutrient] = limit_min_daily * 0.24
                elif nutrient in highly_restricted_nutrients: nutrient_per_meal_min[meal][nutrient] = limit_min_daily * 0.16
                else: nutrient_per_meal_min[meal][nutrient] = limit_min_daily * 0.32
            else: # Makan Malam
                nutrient_per_meal_max[meal][nutrient] = limit_max_daily * 0.3
                if nutrient in moderate_restricted_nutrients: nutrient_per_meal_min[meal][nutrient] = limit_min_daily * 0.18
                elif nutrient in highly_restricted_nutrients: nutrient_per_meal_min[meal][nutrient] = limit_min_daily * 0.12
                else: nutrient_per_meal_min[meal][nutrient] = limit_min_daily * 0.24

    # Mengisi rencana makan dengan Fractional Knapsack approach
    for meal_time in meal_times:
        # Dapatkan target energi min/max untuk waktu makan saat ini
        target_energy_max_meal = energy_per_meal_max[meal_time]
        target_energy_min_meal = energy_per_meal_min[meal_time]

        for food_group in food_groups:
            group_foods = suitable_foods[suitable_foods['FoodGroups'] == food_group].copy()

            # Mengecualikan makanan yang sudah ada dalam taboo list
            # Gunakan copy dari taboo_list untuk iterasi agar tidak ada masalah modifikasi saat iterasi
            group_foods = group_foods[~group_foods['FoodName'].isin(taboo_list)]

            # Jika tidak ada makanan dalam kelompok, lanjut ke kelompok berikutnya
            if group_foods.empty:
                continue

            for _, food in group_foods.iterrows():
                # Dapatkan energi saat ini untuk waktu makan ini (dari total meal_nutrition)
                current_meal_energy = meal_nutrition[meal_time]['Energi']

                # Cek apakah menambahkan makanan ini melebihi batas energi MAKSIMUM untuk waktu makan ini
                if current_meal_energy + food['original_Energi'] <= target_energy_max_meal:
                    exceeds_nutrient_limit = False
                    for nutrient, limit_max in nutrient_per_meal_max[meal_time].items():
                        original_nutrient_key = f'original_{nutrient}'
                        # Pastikan nutrisi ada di food (row DataFrame) dan di meal_nutrition (tracking dict)
                        if original_nutrient_key in food and nutrient in meal_nutrition[meal_time]:
                            if meal_nutrition[meal_time][nutrient] + food[original_nutrient_key] > limit_max:
                                exceeds_nutrient_limit = True
                                break

                    if not exceeds_nutrient_limit:
                        # Tambahkan makanan ke rencana makan
                        add_food_to_meal(food, meal_time, food_group)

                        # Tambahkan ke taboo list untuk mencegah duplikasi di seluruh rencana makan
                        taboo_list.add(food['FoodName'])

                        # Cek apakah semua kriteria nutrisi sudah terpenuhi dalam rentang minimal-maksimal
                        all_nutrients_in_target_range = True
                        current_meal_nutrition_snapshot = meal_nutrition[meal_time] # Mengambil snapshot saat ini

                        # Cek Energi (juga merupakan nutrisi yang harus dipenuhi)
                        if current_meal_nutrition_snapshot['Energi'] < target_energy_min_meal or \
                           current_meal_nutrition_snapshot['Energi'] > target_energy_max_meal:
                            all_nutrients_in_target_range = False

                        # Cek Nutrisi Lainnya (jika energi sudah dalam rentang)
                        if all_nutrients_in_target_range: # Lanjutkan cek jika energi OK
                            for nutrient_key_check in nutrition_req:
                                if nutrient_key_check == 'Energi': # Energi sudah dicek
                                    continue

                                current_nutrient_value_check = current_meal_nutrition_snapshot.get(nutrient_key_check, 0.0)
                                min_limit_meal_check = nutrient_per_meal_min[meal_time].get(nutrient_key_check, 0.0)
                                max_limit_meal_check = nutrient_per_meal_max[meal_time].get(nutrient_key_check, float('inf'))

                                if nutrient_key_check in highly_restricted_nutrients or nutrient_key_check in moderate_restricted_nutrients:
                                    # Untuk nutrisi yang dibatasi, cukup pastikan TIDAK melebihi batas maksimalnya.
                                    if current_nutrient_value_check > max_limit_meal_check:
                                        all_nutrients_in_target_range = False
                                        break
                                else: # Untuk nutrisi yang harus dipenuhi (Protein, Serat, Vit C, Kalium, dll.)
                                    if current_nutrient_value_check < min_limit_meal_check or current_nutrient_value_check > max_limit_meal_check:
                                        all_nutrients_in_target_range = False
                                        break

                        # Hanya break jika SEMUA nutrisi sudah dalam rentang target
                        if all_nutrients_in_target_range:
                            break

            all_nutrients_in_target_range_group_level = True
            current_meal_nutrition_group_level_snapshot = meal_nutrition[meal_time]

            # Cek Energi
            if current_meal_nutrition_group_level_snapshot['Energi'] < target_energy_min_meal or \
               current_meal_nutrition_group_level_snapshot['Energi'] > target_energy_max_meal:
                all_nutrients_in_target_range_group_level = False

            if all_nutrients_in_target_range_group_level:
                for nutrient_key_check_group in nutrition_req:
                    if nutrient_key_check_group == 'Energi':
                        continue

                    current_nutrient_value_check_group = current_meal_nutrition_group_level_snapshot.get(nutrient_key_check_group, 0.0)
                    min_limit_meal_check_group = nutrient_per_meal_min[meal_time].get(nutrient_key_check_group, 0.0)
                    max_limit_meal_check_group = nutrient_per_meal_max[meal_time].get(nutrient_key_check_group, float('inf'))

                    if nutrient_key_check_group in highly_restricted_nutrients or nutrient_key_check_group in moderate_restricted_nutrients:
                        if current_nutrient_value_check_group > max_limit_meal_check_group:
                            all_nutrients_in_target_range_group_level = False
                            break
                    else:
                        if current_nutrient_value_check_group < min_limit_meal_check_group or current_nutrient_value_check_group > max_limit_meal_check_group:
                            all_nutrients_in_target_range_group_level = False
                            break

            if all_nutrients_in_target_range_group_level:
                break

    # Validasi apakah total nutrisi memenuhi batas minimal harian
    total_nutrition_check = {nutrient: 0.0 for nutrient in nutrition_req}

    # Hitung total nutrisi dari semua waktu makan
    for meal_time in meal_times:
        for nutrient in nutrition_req:
            total_nutrition_check[nutrient] += meal_nutrition[meal_time][nutrient]

    # Cek apakah memenuhi batas minimal harian
    meets_minimum_overall = True
    if total_nutrition_check['Energi'] < energy_limit_min_daily:
        meets_minimum_overall = False
        print(f"Peringatan: Total energi ({total_nutrition_check['Energi']:.2f} kkal) kurang dari batas minimal harian ({energy_limit_min_daily:.2f} kkal atau 80% dari kebutuhan). Tambahkan suplemen untuk menambah jumlah kalori yang masuk ke dalam tubuh.")

    for nutrient, min_limit in nutrient_limits_min_daily.items():
        if total_nutrition_check[nutrient] < min_limit:
            meets_minimum_overall = False
            percentage = "60%" if nutrient in moderate_restricted_nutrients else ("40%" if nutrient in highly_restricted_nutrients else "80%")
            print(f"Peringatan: Total {nutrient} ({total_nutrition_check[nutrient]:.2f}) kurang dari batas minimal harian ({min_limit:.2f} atau {percentage} dari kebutuhan). Tambahkan suplemen untuk menambah jumlah {nutrient} yang masuk ke dalam tubuh.")

    # Cek apakah melebihi batas maksimal harian (120%)
    if total_nutrition_check['Energi'] > energy_limit_max_daily:
        print(f"Peringatan: Total energi ({total_nutrition_check['Energi']:.2f} kkal) melebihi batas maksimal harian ({energy_limit_max_daily:.2f} kkal atau 120% dari kebutuhan). Pertimbangkan untuk mengurangi porsi makanan.")

    for nutrient, max_limit in nutrient_limits_max_daily.items():
        if total_nutrition_check[nutrient] > max_limit:
            print(f"Peringatan: Total {nutrient} ({total_nutrition_check[nutrient]:.2f}) melebihi batas maksimal harian ({max_limit:.2f} atau 120% dari kebutuhan). Pertimbangkan untuk mengurangi konsumsi makanan tinggi {nutrient}.")

    # Konversi rencana makan ke DataFrame
    recommended_foods = []
    for meal_time, groups in meal_plan.items():
        for food_group, foods in groups.items():
            for food in foods:
                food['MealTime'] = meal_time
                food['FoodGroup'] = food_group
                recommended_foods.append(food)

    # Jika tidak ada rekomendasi yang memenuhi semua kriteria, pilih beberapa makanan terbaik
    if not recommended_foods:
        print("Tidak dapat menyusun rencana makan dengan batasan yang diberikan. Merekomendasikan makanan terbaik...")
        top_foods = suitable_foods.head(n_recommendations)
        for _, food in top_foods.iterrows():
            recommended_foods.append({
                'FoodID': food['FoodID'],
                'FoodName': food['FoodName'],
                'FoodGroup': food['FoodGroups'],
                'MealTime': 'Tidak Ditentukan', # Menandakan tidak ada alokasi waktu makan spesifik
                'Energi': food['original_Energi'],
                'Protein': food['original_Protein'],
                'Serat': food['original_Serat'],
                'VitaminC': food['original_VitaminC'],
                'Kalium': food['original_Kalium'],
                'Magnesium': food['original_Magnesium'],
                'Kalsium': food['original_Kalsium'],
                'Besi': food['original_Besi'],
                'GulaTotal': food['original_GulaTotal'],
                'Natrium': food['original_Natrium'],
                'LemakJenuh': food['original_LemakJenuh'],
                'Kolesterol': food['original_Kolesterol']
            })

    # Konversi hasil rekomendasi ke DataFrame Pandas
    recommended_df = pd.DataFrame(recommended_foods)

    return recommended_df

# ===== MODEL EVALUATION ======
def evaluate_model(knn, X_test, y_test):
    # Prediksi data uji
    y_pred = knn.predict(X_test)

    # Menghitung metrik evaluasi
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    # Menghitung confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Mendapatkan nilai TP, FP, FN, TN
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tp = np.sum((y_test == 1) & (y_pred == 1))
        fp = np.sum((y_test == 0) & (y_pred == 1))
        fn = np.sum((y_test == 1) & (y_pred == 0))
        tn = np.sum((y_test == 0) & (y_pred == 0))

    # Menyusun hasil evaluasi
    evaluation = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }

    return evaluation

def visualize_confusion_matrix(cm, model_name='KNN'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Suitable', 'Suitable'],
                yticklabels=['Not Suitable', 'Suitable'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

def visualize_nutrition_comparison(recommended_df, user_info):
    # Menghitung total nutrisi dari makanan yang direkomendasikan
    total_nutrition = {}
    for nutrient in ['Energi', 'Protein', 'Serat', 'VitaminC', 'Kalium', 'Magnesium',
                    'Kalsium', 'Besi', 'GulaTotal', 'Natrium', 'LemakJenuh', 'Kolesterol']:
        total_nutrition[nutrient] = recommended_df[nutrient].sum()

    # Kebutuhan nutrisi pengguna
    nutrition_req = user_info['nutrition_req']

    # Membuat DataFrame untuk perbandingan
    comparison = pd.DataFrame({
        'Nutrisi': list(total_nutrition.keys()),
        'Rekomendasi': list(total_nutrition.values()),
        'Kebutuhan': [nutrition_req.get(nutrient, 0) for nutrient in total_nutrition.keys()]
    })

    # Menghitung persentase terpenuhi
    comparison['Persentase'] = comparison['Rekomendasi'] / comparison['Kebutuhan'] * 100
    comparison['Persentase'] = comparison['Persentase'].fillna(0)

    # Membuat heatmap perbandingan nutrisi
    plt.figure(figsize=(12, 8))
    comparison_pivot = comparison.pivot_table(index='Nutrisi', values='Persentase', aggfunc='sum')
    sns.heatmap(comparison_pivot, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title('Persentase Pemenuhan Kebutuhan Nutrisi')
    plt.ylabel('Nutrisi')
    plt.tight_layout()
    plt.show()

    # Membuat barplot perbandingan
    plt.figure(figsize=(14, 10))
    bar_comparison = comparison.melt(id_vars='Nutrisi', value_vars=['Rekomendasi', 'Kebutuhan'], var_name='Tipe', value_name='Nilai')
    sns.barplot(x='Nutrisi', y='Nilai', hue='Tipe', data=bar_comparison)
    plt.xticks(rotation=45, ha='right')
    plt.title('Perbandingan Nutrisi: Rekomendasi vs Kebutuhan')
    plt.tight_layout()
    plt.show()

    return comparison

# ===== MAIN FUNCTION =====
def display_user_info(user_info):
    print("\n========== PROFIL PENGGUNA ==========")
    print(f"Jenis Kelamin: {user_info['gender'].capitalize()}")
    print(f"Usia: {user_info['age_with_category']}")
    print(f"Tinggi Badan: {user_info['height']} cm")
    print(f"Berat Badan: {user_info['weight']} kg")

    # Menampilkan riwayat penyakit
    if user_info['diseases']:
        diseases_str = ", ".join(disease.capitalize() for disease in user_info['diseases'])
        print(f"Riwayat Penyakit: {diseases_str}")
    else:
        print("Riwayat Penyakit: Tidak Ada")

    print("\n========== INFORMASI KESEHATAN ==========")
    print(f"BMI: {user_info['bmi']:.2f}")
    print(f"Kategori BMI: {user_info['bmi_category']}")
    print(f"Berat Badan Ideal: {user_info['ideal_weight']:.2f} kg")
    print(f"Kebutuhan Energi: {user_info['energy_needs']:.2f} kkal")

    print("\n========== KEBUTUHAN NUTRISI ==========")
    nutrition_req = user_info['nutrition_req']
    print(f"Energi: {nutrition_req['Energi']:.2f} kkal")
    print(f"Protein: {nutrition_req['Protein']:.2f} g")
    print(f"Serat: {nutrition_req['Serat']:.2f} g")
    print(f"Gula Total: {nutrition_req['GulaTotal']:.2f} g")
    print(f"Lemak Jenuh: {nutrition_req['LemakJenuh']:.2f} g")
    print(f"Vitamin C: {nutrition_req['VitaminC']:.2f} mg")
    print(f"Magnesium: {nutrition_req['Magnesium']:.2f} mg")
    print(f"Natrium: {nutrition_req['Natrium']:.2f} mg")
    print(f"Kalsium: {nutrition_req['Kalsium']:.2f} mg")
    print(f"Kalium: {nutrition_req['Kalium']:.2f} mg")
    print(f"Besi: {nutrition_req['Besi']:.2f} mg")
    print(f"Kolesterol: {nutrition_req['Kolesterol']:.2f} mg")

def display_food_recommendations(recommended_df):
    print("\n========== REKOMENDASI MAKANAN ==========")

    # Buat copy dari DataFrame untuk menghindari mengubah data asli
    df_display = recommended_df.copy()

    # Definisikan urutan yang diinginkan
    meal_order = ['Sarapan', 'Makan Siang', 'Makan Malam']
    food_group_order = ['Makanan Utama', 'Makanan Pendamping', 'Minuman dan Buah']

    # Konversi ke categorical dengan urutan custom
    df_display['MealTime'] = pd.Categorical(
        df_display['MealTime'],
        categories=meal_order,
        ordered=True
    )
    df_display['FoodGroup'] = pd.Categorical(
        df_display['FoodGroup'],
        categories=food_group_order,
        ordered=True
    )

    # Urutkan DataFrame berdasarkan categorical columns
    df_display = df_display.sort_values(['MealTime', 'FoodGroup'])

    # Mengelompokkan rekomendasi berdasarkan waktu makan dan kelompok makanan
    meal_groups = df_display.groupby(['MealTime', 'FoodGroup'])

    for (meal_time, food_group), foods in meal_groups:
        print(f"\n----- {meal_time}: {food_group} -----")
        for _, food in foods.iterrows():
            print(f"\nNama Makanan: {food['FoodName']}")
            print(f"Energi: {food['Energi']:.2f} kkal")
            print(f"Protein: {food['Protein']:.2f} g")
            print(f"Serat: {food['Serat']:.2f} g")
            print(f"Gula Total: {food['GulaTotal']:.2f} g")
            print(f"Lemak Jenuh: {food['LemakJenuh']:.2f} g")
            print(f"Vitamin C: {food['VitaminC']:.2f} mg")
            print(f"Magnesium: {food['Magnesium']:.2f} mg")
            print(f"Natrium: {food['Natrium']:.2f} mg")
            print(f"Kalsium: {food['Kalsium']:.2f} mg")
            print(f"Kalium: {food['Kalium']:.2f} mg")
            print(f"Besi: {food['Besi']:.2f} mg")
            print(f"Kolesterol: {food['Kolesterol']:.2f} mg")

    # Menghitung total nutrisi
    total_nutrition = recommended_df.sum()

    print("\n========== TOTAL NUTRISI REKOMENDASI ==========")
    print(f"Total Energi: {total_nutrition['Energi']:.2f} kkal")
    print(f"Total Protein: {total_nutrition['Protein']:.2f} g")
    print(f"Total Serat: {total_nutrition['Serat']:.2f} g")
    print(f"Total Gula: {total_nutrition['GulaTotal']:.2f} g")
    print(f"Total Lemak Jenuh: {total_nutrition['LemakJenuh']:.2f} g")
    print(f"Total Vitamin C: {total_nutrition['VitaminC']:.2f} mg")
    print(f"Total Magnesium: {total_nutrition['Magnesium']:.2f} mg")
    print(f"Total Natrium: {total_nutrition['Natrium']:.2f} mg")
    print(f"Total Kalsium: {total_nutrition['Kalsium']:.2f} mg")
    print(f"Total Kalium: {total_nutrition['Kalium']:.2f} mg")
    print(f"Total Besi: {total_nutrition['Besi']:.2f} mg")
    print(f"Total Kolesterol: {total_nutrition['Kolesterol']:.2f} mg")

# ===== CASE NORMAL =====
def main():
    # Memuat data
    github_url = "https://raw.githubusercontent.com/bachtiarrizkyal/TugasAkhir/refs/heads/main/USDA.xlsx"
    df_usda = pd.read_excel(github_url)

    # Melakukan preprocessing data
    df_preprocessed, scaler, df_original, nutrient_cols = preprocess_data(df_usda)

    # Memproses data pengguna (contoh)
    gender = 'laki-laki'
    age = 22
    height = 170
    weight = 75
    diseases = []

    user_info = process_user_data(gender, age, height, weight, diseases)

    # Tampilkan informasi pengguna
    display_user_info(user_info)

    # Menyiapkan data makanan
    df_labeled, df_main, df_side, df_drink_fruit = prepare_food_data(df_preprocessed, user_info, df_original)

    # Melatih model KNN
    knn, X, y, X_train, X_test, y_train, y_test = train_knn_model(df_labeled, user_info)

    # Evaluasi model
    evaluation = evaluate_model(knn, X_test, y_test)

    print("\n========== EVALUASI MODEL ==========")
    print(f"Accuracy: {evaluation['accuracy']:.4f}")
    print(f"Precision: {evaluation['precision']:.4f}")
    print(f"Recall: {evaluation['recall']:.4f}")
    print(f"True Positive (TP): {evaluation['tp']}")
    print(f"False Positive (FP): {evaluation['fp']}")
    print(f"False Negative (FN): {evaluation['fn']}")
    print(f"True Negative (TN): {evaluation['tn']}")

    # Visualisasi confusion matrix
    visualize_confusion_matrix(evaluation['confusion_matrix'])

    # Rekomendasi makanan dengan Fractional Knapsack
    recommended_df = get_recommendation(knn, df_labeled, user_info, df_original)

    # Tampilkan rekomendasi
    display_food_recommendations(recommended_df)

    # Visualisasi perbandingan nutrisi
    comparison = visualize_nutrition_comparison(recommended_df, user_info)

    # Tampilkan persentase pemenuhan nutrisi
    print("\n========== PERSENTASE PEMENUHAN KEBUTUHAN NUTRISI ==========")
    for _, row in comparison.iterrows():
        print(f"{row['Nutrisi']}: {row['Persentase']:.2f}% dari kebutuhan")

if __name__ == "__main__":
    main()