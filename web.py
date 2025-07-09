import streamlit as st
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

# Set page configuration
st.set_page_config(
    page_title="Rekomendasi Makanan",
    page_icon="🍱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #43a047;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1e88e5;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the USDA food data"""
    try:
        github_url = "https://raw.githubusercontent.com/bachtiarrizkyal/TugasAkhir/refs/heads/main/USDAnew.xlsx"
        df_usda = pd.read_excel(github_url)
        return df_usda
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    # 1. Memilih makanan relevan
    df_filtered_foodname = df[~df['FoodName'].str.contains('mentah|tidak siap|alkohol', case=False, na=False)]
    df_clean = df_filtered_foodname[~df_filtered_foodname['FoodCategory'].str.contains('babi', case=False, na=False)]

    # 2. Membuat kolom baru 'FoodGroup' berdasarkan 'FoodCategory'
    food_group_mapping = {
        'Babi': 'Makanan Utama',
        'Daging dan Sosis': 'Makanan Utama',
        'Daging Merah dan Hasil Buruan': 'Makanan Utama',
        'Ikan dan Kerang': 'Makanan Utama',
        'Makanan Cepat Saji': 'Makanan Utama',
        'Sereal dan Pasta': 'Makanan Utama',
        'Unggas': 'Makanan Utama',
        'Keju, Susu, dan Telur': 'Makanan Pendamping',
        'Makanan Pembuka dan Lauk': 'Makanan Pendamping',
        'Produk Panggang': 'Makanan Pendamping',
        'Sayuran': 'Makanan Pendamping',
        'Sup dan Kaldu': 'Makanan Pendamping',
        'Buah dan Jus': 'Minuman dan Buah',
        'Minuman': 'Minuman dan Buah'
    }
    df_clean['FoodGroup'] = df_clean['FoodCategory'].map(food_group_mapping)

    # Memindahkan kolom FoodGroup agar berada sebelum FoodName
    cols = df_clean.columns.tolist()
    food_name_idx = cols.index('FoodName')

    # Hapus FoodGroup dari posisi aslinya (jika sudah ada, yang seharusnya tidak jika baru dibuat)
    if 'FoodGroup' in cols:
        cols.remove('FoodGroup')
    # Masukkan FoodGroup pada posisi sebelum FoodName
    cols.insert(food_name_idx, 'FoodGroup')
    df_clean = df_clean[cols]

    # Mengisi nilai kosong dengan 0
    df_clean.fillna(0, inplace=True)

    # Menghapus duplikat berdasarkan FoodName
    df_clean = df_clean.drop_duplicates(subset=['FoodName'])

    # Memilih kolom yang akan digunakan
    selected_columns = [
        'FoodID', 'FoodCategory', 'FoodGroup', 'FoodName',
        'Energi', 'Protein', 'Serat', 'VitaminC', 'Kalium', 'Magnesium',
        'Kalsium', 'Besi', 'GulaTotal', 'Natrium', 'LemakJenuh', 'Kolesterol'
    ]
    df_clean = df_clean[selected_columns]

    # Simpan salinan data asli sebelum normalisasi
    df_original = df_clean.copy()

    # Normalisasi data nutrisi
    non_numeric_cols = ['FoodID', 'FoodCategory', 'FoodGroup', 'FoodName']
    numeric_cols = [col for col in df_clean.columns if col not in non_numeric_cols]
    scaler = MinMaxScaler()
    df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])

    return df_clean, scaler, df_original, numeric_cols

def calculate_bmi(weight, height):
    """Calculate BMI"""
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    return bmi

def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Berat Badan Kurang (Underweight)"
    elif 18.5 <= bmi < 25:
        return "Berat Badan Normal"
    elif 25 <= bmi < 30:
        return "Berat Badan Berlebih (Overweight)"
    elif 30 <= bmi < 35:
        return "Obesitas I"
    elif 35 <= bmi < 40:
        return "Obesitas II"
    else: # bmi >= 40
        return "Obesitas III"

def calculate_ideal_weight(gender, height):
    """Calculate ideal weight"""
    if gender.lower() == 'laki-laki':
        return (height - 100) - ((height - 100) * 0.1)
    else:
        return (height - 100) - ((height - 100) * 0.15)

def calculate_energy_needs(gender, age, height, ideal_weight):
    """Calculate energy needs"""
    if gender.lower() == 'laki-laki':
        bmr = 66.5 + (13.75 * ideal_weight) + (5.003 * height) - (6.75 * age)
    else:
        bmr = 655.1 + (9.563 * ideal_weight) + (1.850 * height) - (4.676 * age)
    
    energy_needs = bmr * 1.55
    return energy_needs

def get_age_category(age):
    """Get age category"""
    if 10 <= age <= 18:
        return "remaja"
    elif 19 <= age <= 64:
        return "dewasa"
    else:
        return "lansia"

def get_nutrition_requirements(gender, age, energy_needs, diseases):
    """Get nutrition requirements based on user profile"""
    age_category = get_age_category(age)
    idx = 0
    
    # Age and gender-based index calculation
    if gender.lower() == 'laki-laki' and age_category == 'remaja':
        if 10 <= age <= 12:
            idx = 0
        elif 13 <= age <= 15:
            idx = 1
        elif 16 <= age <= 18:
            idx = 2
    elif gender.lower() == 'laki-laki' and age_category == 'dewasa':
        if 19 <= age <= 29:
            idx = 3
        elif 30 <= age <= 49:
            idx = 4
        elif 50 <= age <= 64:
            idx = 5
    elif gender.lower() == 'laki-laki' and age_category == 'lansia':
        if 65 <= age <= 80:
            idx = 6
        else:
            idx = 7
    elif gender.lower() == 'perempuan' and age_category == 'remaja':
        if 10 <= age <= 12:
            idx = 8
        elif 13 <= age <= 15:
            idx = 9
        elif 16 <= age <= 18:
            idx = 10
    elif gender.lower() == 'perempuan' and age_category == 'dewasa':
        if 19 <= age <= 29:
            idx = 11
        elif 30 <= age <= 49:
            idx = 12
        elif 50 <= age <= 64:
            idx = 13
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

def process_user_data(gender, age, height, weight, diseases, preferred_food_category):
    """Process user data and calculate health metrics"""
    age_category = get_age_category(age)
    age_with_category = f"{age} tahun ({age_category})"
    
    bmi = calculate_bmi(weight, height)
    bmi_category = get_bmi_category(bmi)
    
    ideal_weight = calculate_ideal_weight(gender, height)
    energy_needs = calculate_energy_needs(gender, age, height, ideal_weight)
    nutrition_req = get_nutrition_requirements(gender, age, energy_needs, diseases)
    
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
        'nutrition_req': nutrition_req,
        'preferred_food_category': preferred_food_category
    }
    
    return user_info

def label_food_by_nutrition(df, diseases, df_original):
    """Label food based on nutrition and disease conditions"""
    df_nutrition = df.copy()
    
    # Add original nutrition values
    nutrient_cols = ['Energi', 'Protein', 'Serat', 'VitaminC', 'Kalium', 'Magnesium',
                    'Kalsium', 'Besi', 'GulaTotal', 'Natrium', 'LemakJenuh', 'Kolesterol']
    
    for col in nutrient_cols:
        df_nutrition[f'original_{col}'] = df_original[col].values
    
    # Define thresholds for high nutrients
    high_thresholds = {
        'Protein': 10,
        'Serat': 6,
        'VitaminC': 15,
        'Kalium': 250,
        'Magnesium': 48,
        'Kalsium': 150,
        'Besi': 2.4
    }
    
    # Define thresholds for low nutrients
    low_thresholds = {
        'GulaTotal': 15,
        'Natrium': 600,
        'LemakJenuh': 5,
        'Kolesterol': 100
    }
    
    # Identify foods high in certain nutrients
    df_nutrition['high_protein'] = df_nutrition['original_Protein'] >= high_thresholds['Protein']
    df_nutrition['high_fiber'] = df_nutrition['original_Serat'] >= high_thresholds['Serat']
    df_nutrition['high_vitaminc'] = df_nutrition['original_VitaminC'] >= high_thresholds['VitaminC']
    df_nutrition['high_potassium'] = df_nutrition['original_Kalium'] >= high_thresholds['Kalium']
    df_nutrition['high_magnesium'] = df_nutrition['original_Magnesium'] >= high_thresholds['Magnesium']
    df_nutrition['high_calcium'] = df_nutrition['original_Kalsium'] >= high_thresholds['Kalsium']
    df_nutrition['high_iron'] = df_nutrition['original_Besi'] >= high_thresholds['Besi']
    
    # Identify foods low in certain nutrients
    df_nutrition['low_sugar'] = df_nutrition['original_GulaTotal'] <= low_thresholds['GulaTotal']
    df_nutrition['low_sodium'] = df_nutrition['original_Natrium'] <= low_thresholds['Natrium']
    df_nutrition['low_sat_fat'] = df_nutrition['original_LemakJenuh'] <= low_thresholds['LemakJenuh']
    df_nutrition['low_cholesterol'] = df_nutrition['original_Kolesterol'] <= low_thresholds['Kolesterol']
    
    # Label makanan berdasarkan riwayat penyakit
    if not diseases or 'normal' in diseases:
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
        
        # Fallback jika tidak ada kondisi yang cocok
        else:
            df_nutrition['suitable'] = 1
    
    return df_nutrition

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

    return knn, X, y, X_train, X_test, y_train, y_test

def get_recommendation(knn, df_labeled, user_info, df_original, n_recommendations=15):
    X_predict = df_labeled[['Energi', 'Protein', 'Serat', 'VitaminC', 'Kalium',
                            'Magnesium', 'Kalsium', 'Besi', 'GulaTotal', 'Natrium',
                            'LemakJenuh', 'Kolesterol']]

    knn_predictions = knn.predict(X_predict)
    df_labeled['knn_suitable'] = knn_predictions
    suitable_foods = df_labeled[df_labeled['knn_suitable'] == 1].copy()

    if suitable_foods.empty:
        print("KNN tidak menemukan makanan yang sesuai. Menggunakan semua makanan...")
        suitable_foods = df_labeled.copy()

    nutrition_req = user_info['nutrition_req']
    preferred_category = user_info.get('preferred_food_category')

    suitable_foods['value_per_calorie'] = 0
    for nutrient in ['Protein', 'Serat', 'VitaminC', 'Kalium', 'Magnesium', 'Kalsium', 'Besi']:
        if nutrient in nutrition_req:
            original_nutrient = f'original_{nutrient}'
            suitable_foods['value_per_calorie'] += suitable_foods[original_nutrient] / nutrition_req[nutrient] if nutrition_req[nutrient] > 0 else 0

    for nutrient in ['GulaTotal', 'Natrium', 'LemakJenuh', 'Kolesterol']:
        if nutrient in nutrition_req:
            original_nutrient = f'original_{nutrient}'
            suitable_foods['value_per_calorie'] -= suitable_foods[original_nutrient] / nutrition_req[nutrient] if nutrition_req[nutrient] > 0 else 0

    suitable_foods['value_per_calorie'] = suitable_foods['value_per_calorie'] / suitable_foods['original_Energi']
    suitable_foods = suitable_foods.sort_values('value_per_calorie', ascending=False)

    energy_limit_max_daily = nutrition_req['Energi'] * 1.2
    energy_limit_min_daily = nutrition_req['Energi'] * 0.8

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

    meal_plan = {
        'Sarapan': {'Makanan Utama': [], 'Makanan Pendamping': [], 'Minuman dan Buah': []},
        'Makan Siang': {'Makanan Utama': [], 'Makanan Pendamping': [], 'Minuman dan Buah': []},
        'Makan Malam': {'Makanan Utama': [], 'Makanan Pendamping': [], 'Minuman dan Buah': []}
    }

    meal_nutrition = {
        'Sarapan': {nutrient: 0.0 for nutrient in nutrition_req},
        'Makan Siang': {nutrient: 0.0 for nutrient in nutrition_req},
        'Makan Malam': {nutrient: 0.0 for nutrient in nutrition_req}
    }

    def add_food_to_meal(food, meal_time, food_group):
        meal_plan[meal_time][food_group].append({
            'FoodID': food['FoodID'], 'FoodName': food['FoodName'], 'FoodCategory': food['FoodCategory'], 'Energi': food['original_Energi'],
            'Protein': food['original_Protein'], 'Serat': food['original_Serat'], 'VitaminC': food['original_VitaminC'],
            'Kalium': food['original_Kalium'], 'Magnesium': food['original_Magnesium'], 'Kalsium': food['original_Kalsium'],
            'Besi': food['original_Besi'], 'GulaTotal': food['original_GulaTotal'], 'Natrium': food['original_Natrium'],
            'LemakJenuh': food['original_LemakJenuh'], 'Kolesterol': food['original_Kolesterol']
        })
        for nutrient_key in nutrition_req:
            original_key = f'original_{nutrient_key}'
            if original_key in food:
                meal_nutrition[meal_time][nutrient_key] += food[original_key]

    taboo_list = set()
    preferred_category_added_once = False

    meal_times = ['Sarapan', 'Makan Siang', 'Makan Malam']
    food_groups = ['Makanan Utama', 'Makanan Pendamping', 'Minuman dan Buah']

    energy_per_meal_max = {
        'Sarapan': energy_limit_max_daily * 0.36, 'Makan Siang': energy_limit_max_daily * 0.48, 'Makan Malam': energy_limit_max_daily * 0.36
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

    # --- Bagian Prioritas Makanan dari Kategori yang Disukai (Sekali Saja, Bebas Waktu Makan, Mengikuti FoodGroup Asli) ---
    if preferred_category and not preferred_category_added_once:
        best_preferred_food = None
        
        # Filter kandidat makanan dari kategori yang disukai, sudah diurutkan berdasarkan value_per_calorie
        preferred_foods_candidates = suitable_foods[
            (suitable_foods['FoodCategory'] == preferred_category) &
            (suitable_foods['knn_suitable'] == 1) &
            (~suitable_foods['FoodName'].isin(taboo_list))
        ].sort_values('value_per_calorie', ascending=False)

        chosen_meal_time = None
        chosen_food = None

        # Prioritaskan waktu makan secara berurutan (Sarapan, Makan Siang, Makan Malam)
        for current_meal_time_attempt in meal_times:
            target_energy_max_meal = energy_per_meal_max[current_meal_time_attempt]
            current_meal_nutrition_snapshot = meal_nutrition[current_meal_time_attempt]

            for _, food in preferred_foods_candidates.iterrows():
                # Cek apakah makanan ini bisa ditambahkan tanpa melanggar batas MAX
                if food['FoodGroup'] not in meal_plan[current_meal_time_attempt]:
                    # Jika FoodGroup makanan ini belum ada di struktur meal_plan untuk waktu makan ini
                    # (ini adalah pengecekan defensif, seharusnya food_groups sudah lengkap)
                    continue

                if current_meal_nutrition_snapshot['Energi'] + food['original_Energi'] <= target_energy_max_meal:
                    exceeds_nutrient_limit = False
                    for nutrient, limit_max in nutrient_per_meal_max[current_meal_time_attempt].items():
                        original_nutrient_key = f'original_{nutrient}'
                        if original_nutrient_key in food and nutrient in current_meal_nutrition_snapshot:
                            if current_meal_nutrition_snapshot[nutrient] + food[original_nutrient_key] > limit_max:
                                exceeds_nutrient_limit = True
                                break
                    
                    if not exceeds_nutrient_limit:
                        chosen_meal_time = current_meal_time_attempt
                        chosen_food = food
                        break # Ambil yang pertama terbaik yang ditemukan untuk waktu makan ini
            if chosen_food is not None: # Jika makanan ditemukan untuk waktu makan ini, hentikan pencarian
                break

        if chosen_food is not None:
            # Gunakan FoodGroup asli dari chosen_food saat menambahkannya!
            add_food_to_meal(chosen_food, chosen_meal_time, chosen_food['FoodGroup'])
            taboo_list.add(chosen_food['FoodName'])
            preferred_category_added_once = True
            print(f"Makanan dari kategori favorit '{preferred_category}' berhasil ditambahkan pada waktu makan: {chosen_meal_time} ({chosen_food['FoodGroup']}).")
        else:
            print(f"Peringatan: Tidak dapat menemukan makanan dari kategori favorit '{preferred_category}' yang sesuai dan memenuhi batasan untuk rencana makan harian.")


    # --- Lanjutkan dengan logika pengisian rencana makan yang ada (Algoritma Fractional Knapsack) ---
    for meal_time in meal_times:
        target_energy_max_meal = energy_per_meal_max[meal_time]
        target_energy_min_meal = energy_per_meal_min[meal_time]

        for food_group in food_groups:
            group_foods = suitable_foods[suitable_foods['FoodGroup'] == food_group].copy()
            group_foods = group_foods[~group_foods['FoodName'].isin(taboo_list)]

            if group_foods.empty:
                continue

            for _, food in group_foods.iterrows():
                current_meal_energy = meal_nutrition[meal_time]['Energi']

                if current_meal_energy + food['original_Energi'] <= target_energy_max_meal:
                    exceeds_nutrient_limit = False
                    for nutrient, limit_max in nutrient_per_meal_max[meal_time].items():
                        original_nutrient_key = f'original_{nutrient}'
                        if original_nutrient_key in food and nutrient in meal_nutrition[meal_time]:
                            if meal_nutrition[meal_time][nutrient] + food[original_nutrient_key] > limit_max:
                                exceeds_nutrient_limit = True
                                break

                    if not exceeds_nutrient_limit:
                        add_food_to_meal(food, meal_time, food_group)
                        taboo_list.add(food['FoodName'])

                        all_nutrients_in_target_range = True
                        current_meal_nutrition_snapshot = meal_nutrition[meal_time]

                        if current_meal_nutrition_snapshot['Energi'] < target_energy_min_meal or \
                           current_meal_nutrition_snapshot['Energi'] > target_energy_max_meal:
                            all_nutrients_in_target_range = False

                        if all_nutrients_in_target_range:
                            for nutrient_key_check in nutrition_req:
                                if nutrient_key_check == 'Energi':
                                    continue

                                current_nutrient_value_check = current_meal_nutrition_snapshot.get(nutrient_key_check, 0.0)
                                min_limit_meal_check = nutrient_per_meal_min[meal_time].get(nutrient_key_check, 0.0)
                                max_limit_meal_check = nutrient_per_meal_max[meal_time].get(nutrient_key_check, float('inf'))

                                if nutrient_key_check in highly_restricted_nutrients or nutrient_key_check in moderate_restricted_nutrients:
                                    if current_nutrient_value_check > max_limit_meal_check:
                                        all_nutrients_in_target_range = False
                                        break
                                else:
                                    if current_nutrient_value_check < min_limit_meal_check or current_nutrient_value_check > max_limit_meal_check:
                                        all_nutrients_in_target_range = False
                                        break

                        if all_nutrients_in_target_range:
                            break

            all_nutrients_in_target_range_group_level = True
            current_meal_nutrition_group_level_snapshot = meal_nutrition[meal_time]

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

    total_nutrition_check = {nutrient: 0.0 for nutrient in nutrition_req}

    for meal_time in meal_times:
        for nutrient in nutrition_req:
            total_nutrition_check[nutrient] += meal_nutrition[meal_time][nutrient]

    meets_minimum_overall = True
    if total_nutrition_check['Energi'] < energy_limit_min_daily:
        meets_minimum_overall = False
        print(f"Peringatan: Total energi ({total_nutrition_check['Energi']:.2f} kkal) kurang dari batas minimal harian ({energy_limit_min_daily:.2f} kkal atau 80% dari kebutuhan). Tambahkan suplemen untuk menambah jumlah kalori yang masuk ke dalam tubuh.")

    for nutrient, min_limit in nutrient_limits_min_daily.items():
        if total_nutrition_check[nutrient] < min_limit:
            meets_minimum_overall = False
            percentage = "60%" if nutrient in moderate_restricted_nutrients else ("40%" if nutrient in highly_restricted_nutrients else "80%")
            print(f"Peringatan: Total {nutrient} ({total_nutrition_check[nutrient]:.2f}) kurang dari batas minimal harian ({min_limit:.2f} atau {percentage} dari kebutuhan). Tambahkan suplemen untuk menambah jumlah {nutrient} yang masuk ke dalam tubuh.")

    if total_nutrition_check['Energi'] > energy_limit_max_daily:
        print(f"Peringatan: Total energi ({total_nutrition_check['Energi']:.2f} kkal) melebihi batas maksimal harian ({energy_limit_max_daily:.2f} kkal atau 120% dari kebutuhan). Pertimbangkan untuk mengurangi porsi makanan.")

    for nutrient, max_limit in nutrient_limits_max_daily.items():
        if total_nutrition_check[nutrient] > max_limit:
            print(f"Peringatan: Total {nutrient} ({total_nutrition_check[nutrient]:.2f}) melebihi batas maksimal harian ({max_limit:.2f} atau 120% dari kebutuhan). Pertimbangkan untuk mengurangi konsumsi makanan tinggi {nutrient}.")

    recommended_foods = []
    for meal_time, groups in meal_plan.items():
        for food_group, foods in groups.items():
            for food in foods:
                food['MealTime'] = meal_time
                food['FoodGroup'] = food_group
                recommended_foods.append(food)

    if not recommended_foods:
        print("Tidak dapat menyusun rencana makan dengan batasan yang diberikan. Merekomendasikan makanan terbaik...")
        top_foods = suitable_foods.head(n_recommendations)
        for _, food in top_foods.iterrows():
            recommended_foods.append({
                'FoodID': food['FoodID'],
                'FoodName': food['FoodName'],
                'FoodCategory': food['FoodCategory'],
                'FoodGroup': food['FoodGroup'],
                'MealTime': 'Tidak Ditentukan',
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

    recommended_df = pd.DataFrame(recommended_foods)

    return recommended_df

def main():
    st.markdown('<h1 class="main-header">🍱 Rekomendasi Makanan Berbasis Data USDA</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Memuat data makanan...'):
        df_usda = load_data()
    
    if df_usda is None:
        st.error("Gagal memuat data. Silakan coba lagi.")
        return
    
    # Preprocess data
    df_preprocessed, scaler, df_original, nutrient_cols = preprocess_data(df_usda)
    
    # Sidebar for user input
    st.sidebar.markdown("## 📋 Input Data Pengguna")
    
    # Gender selection
    gender = st.sidebar.selectbox(
        "Jenis Kelamin:",
        options=["Laki-laki", "Perempuan"],
        index=0
    )
    
    # Age input
    age = st.sidebar.number_input(
        "Usia (tahun):",
        min_value=10,
        max_value=99,
        value=25,
        step=1
    )
    
    # Height input
    height = st.sidebar.number_input(
        "Tinggi Badan (cm): (minimal 150 cm)",
        min_value=150,
        max_value=300,
        value=170,
        step=1
    )
    
    # Weight input
    weight = st.sidebar.number_input(
        "Berat Badan (kg):",
        min_value=30.0,
        max_value=200.0,
        value=70.0,
        step=0.1
    )
    
    # Health conditions
    st.sidebar.markdown("### Kondisi Kesehatan:")
    
    # Normal condition
    normal = st.sidebar.checkbox("Normal (Tidak ada penyakit)")
    
    # Disease conditions (disabled if normal is selected)
    diabetes = st.sidebar.checkbox("Diabetes", disabled=normal)
    hipertensi = st.sidebar.checkbox("Hipertensi", disabled=normal)
    cardiovascular = st.sidebar.checkbox("Cardiovascular Disease", disabled=normal)
    
    # Prepare diseases list
    diseases = []
    if normal:
        diseases.append('normal')
    else:
        if diabetes:
            diseases.append('diabetes')
        if hipertensi:
            diseases.append('hipertensi')
        if cardiovascular:
            diseases.append('cardiovascular disease')

    st.sidebar.markdown("### Preferensi Makanan:")
    
    # Mengambil kategori makanan dari data asli untuk dropdown
    food_categories_from_data = sorted(df_original['FoodCategory'].unique().tolist())
    
    # Tambahkan opsi "Tidak ada preferensi" di awal dropdown
    preferred_category_options = ["Tidak ada preferensi"] + food_categories_from_data
    
    selected_preferred_category = st.sidebar.selectbox(
        "Pilih kategori makanan yang Anda sukai (opsional):",
        options=preferred_category_options,
        index=0 # Default ke "Tidak ada preferensi"
    )
    
    # Set preferred_category berdasarkan pilihan dropdown
    preferred_category = None
    if selected_preferred_category != "Tidak ada preferensi":
        preferred_category = selected_preferred_category
    
    # Process button
    if st.sidebar.button("🔍 Dapatkan Rekomendasi", type="primary"):
        with st.spinner('Menyusun rekomendasi makanan'):
            # Process user data
            user_info = process_user_data(gender.lower(), age, height, weight, diseases,  preferred_category)
            
            # Display user profile
            st.markdown('<h2 class="sub-header">👤 Profil Pengguna</h2>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("BMI", f"{user_info['bmi']:.1f}")
                st.caption(user_info['bmi_category'])
            
            with col2:
                st.metric("Berat Ideal", f"{user_info['ideal_weight']:.1f} kg")
            
            with col3:
                st.metric("Kebutuhan Energi", f"{user_info['energy_needs']:.0f} kkal")
            
            with col4:
                st.metric("Kelompok Usia", user_info['age_category'].title())
                st.caption(f"{user_info['age']} tahun")
            
            # Kondisi kesehatan dalam baris terpisah untuk memberikan ruang lebih
            diseases_str = ", ".join([d.title() for d in diseases]) if diseases else "Tidak Ada"
            st.metric("Kondisi Kesehatan", diseases_str)

            if user_info.get('preferred_food_category'):
                st.metric("Preferensi Makanan", user_info['preferred_food_category'].title())
            else:
                st.metric("Preferensi Makanan", "Tidak Ada/Tidak Ditentukan")
            
            # Label food data
            df_labeled = label_food_by_nutrition(df_preprocessed, diseases, df_original)

            # Melatih model KNN
            knn, X, y, X_train, X_test, y_train, y_test = train_knn_model(df_labeled, user_info)

            # Get recommendations
            recommended_df = get_recommendation(knn, df_labeled, user_info, df_original)
            
            if not recommended_df.empty:
                # Display meal plan
                st.markdown('<h2 class="sub-header">🍽️ Rencana Makan Harian</h2>', unsafe_allow_html=True)
                st.markdown('<h6 class="sub-header">Hasil Rekomendasi Makanan adalah per 100 gram "Bagian yang Dapat Dimakan" (BDD)</h6>', unsafe_allow_html=True)
                
                meal_times = ['Sarapan', 'Makan Siang', 'Makan Malam']
                
                for meal_time in meal_times:
                    st.markdown(f"### {meal_time}")
                    meal_foods = recommended_df[recommended_df['MealTime'] == meal_time]
                    
                    if not meal_foods.empty:
                        food_groups = ['Makanan Utama', 'Makanan Pendamping', 'Minuman dan Buah']
                        
                        cols = st.columns(3)
                        for i, food_group in enumerate(food_groups):
                            with cols[i]:
                                st.markdown(f"**{food_group}**")
                                group_foods = meal_foods[meal_foods['FoodGroup'] == food_group]
                                
                                if not group_foods.empty:
                                    for _, food in group_foods.iterrows():
                                        with st.expander(f"🍴 {food['FoodName']}"):
                                            st.write(f"**Kategori:** {food['FoodCategory']}")
                                            col_a, col_b = st.columns(2)
                                            with col_a:
                                                st.write(f"**Energi:** {food['Energi']:.1f} kkal")
                                                st.write(f"**Protein:** {food['Protein']:.1f} g")
                                                st.write(f"**Serat:** {food['Serat']:.1f} g")
                                                st.write(f"**Vitamin C:** {food['VitaminC']:.1f} mg")
                                                st.write(f"**Kalsium:** {food['Kalsium']:.1f} mg")
                                                st.write(f"**Besi:** {food['Besi']:.1f} mg")
                                            with col_b:
                                                st.write(f"**Gula Total:** {food['GulaTotal']:.1f} g")
                                                st.write(f"**Lemak Jenuh:** {food['LemakJenuh']:.1f} g")
                                                st.write(f"**Natrium:** {food['Natrium']:.1f} mg")
                                                st.write(f"**Kalium:** {food['Kalium']:.1f} mg")
                                                st.write(f"**Magnesium:** {food['Magnesium']:.1f} mg")
                                                st.write(f"**Kolesterol:** {food['Kolesterol']:.1f} mg")
                                else:
                                    st.write("Tidak ada rekomendasi")
                    else:
                        st.write("Tidak ada rekomendasi untuk waktu makan ini")
                
                # Display total nutrition
                st.markdown('<h2 class="sub-header">📊 Total Nutrisi Rekomendasi</h2>', unsafe_allow_html=True)
                
                total_nutrition = recommended_df.sum()
                nutrition_req = user_info['nutrition_req']
                
                # Create comparison dataframe
                nutrients = ['Energi', 'Protein', 'Serat', 'VitaminC', 'Kalium', 'Magnesium',
                           'Kalsium', 'Besi', 'GulaTotal', 'Natrium', 'LemakJenuh', 'Kolesterol']
                
                comparison_data = []
                for nutrient in nutrients:
                    recommended_value = total_nutrition.get(nutrient, 0)
                    required_value = nutrition_req.get(nutrient, 0)
                    percentage = (recommended_value / required_value * 100) if required_value > 0 else 0
                    
                    comparison_data.append({
                        'Nutrisi': nutrient,
                        'Rekomendasi': recommended_value,
                        'Kebutuhan': required_value,
                        'Persentase': percentage
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                
                # Display as metrics
                cols = st.columns(4)
                for i, row in comparison_df.iterrows():
                    with cols[i % 4]:
                        unit = "kkal" if row['Nutrisi'] == 'Energi' else ("g" if row['Nutrisi'] in ['Protein', 'Serat', 'GulaTotal', 'LemakJenuh'] else "mg")
                        st.metric(
                            label=row['Nutrisi'],
                            value=f"{row['Rekomendasi']:.1f} {unit}",
                            delta=f"{row['Persentase']:.0f}% dari kebutuhan"
                        )
                # Visualization
                def get_custom_color(nutrient, percentage):
                    if nutrient in ['Energi', 'Protein', 'Serat', 'VitaminC', 'Kalium', 'Magnesium', 'Kalsium', 'Besi']:
                        if percentage < 80:
                            return 'red'
                        elif 80 <= percentage <= 100:
                            return 'orange'
                        elif percentage > 100:
                            return 'green'
                    elif nutrient in ['GulaTotal', 'Natrium']:
                        if percentage < 60:
                            return 'red'
                        elif 60 <= percentage <= 100:
                            return 'orange'
                        elif percentage > 100:
                            return 'green'
                    elif nutrient in ['LemakJenuh', 'Kolesterol']:
                        if percentage < 40:
                            return 'red'
                        elif 40 <= percentage <= 100:
                            return 'orange'
                        elif percentage > 100:
                            return 'green'
                
                st.markdown('<h2 class="sub-header">📈 Visualisasi Pemenuhan Nutrisi</h2>', unsafe_allow_html=True)

                fig1, ax1 = plt.subplots(figsize=(10, 6))
                x = np.arange(len(comparison_df))
                width = 0.35

                ax1.bar(x - width/2, comparison_df['Rekomendasi'], width, label='Rekomendasi', color='orange', alpha=0.8)
                ax1.bar(x + width/2, comparison_df['Kebutuhan'], width, label='Kebutuhan', color='green', alpha=0.8)
                ax1.set_xlabel('Nutrisi')
                ax1.set_ylabel('Jumlah')
                ax1.set_title('Perbandingan Nutrisi: Rekomendasi vs Kebutuhan')
                ax1.set_xticks(x)
                ax1.set_xticklabels(comparison_df['Nutrisi'], rotation=45, ha='right')
                ax1.legend()
                ax1.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig1)

                st.markdown('---')

                st.markdown('<h3 style="text-align: center;">Persentase Pemenuhan Kebutuhan Nutrisi</h3>', unsafe_allow_html=True)

                fig_combined, (ax_80, ax_60, ax_40) = plt.subplots(1, 3, figsize=(18, 6), sharey=True) 

                # Kelompok Nutrisi: Minimal 80% Pemenuhan
                nutrients_80_percent = ['Energi', 'Protein', 'Serat', 'VitaminC', 'Kalium', 'Magnesium', 'Kalsium', 'Besi']
                df_80 = comparison_df[comparison_df['Nutrisi'].isin(nutrients_80_percent)]

                if not df_80.empty:
                    colors_80 = [get_custom_color(row['Nutrisi'], row['Persentase']) for index, row in df_80.iterrows()]
                    ax_80.bar(df_80['Nutrisi'], df_80['Persentase'], color=colors_80, alpha=0.7)

                    # Garis referensi untuk kelompok 80%
                    ax_80.axhline(y=120, color='red', linestyle='--', alpha=0.7, label='Batas Toleransi Maksimal (120%)')
                    ax_80.axhline(y=100, color='blue', linestyle='-', alpha=0.7, label='Target Optimal (100%)')
                    ax_80.axhline(y=80, color='green', linestyle='--', alpha=0.7, label='Minimal Target (80%)')
                    
                    ax_80.set_xlabel('Nutrisi')
                    ax_80.set_ylabel('Persentase Pemenuhan (%)')
                    ax_80.set_title('Minimal 80%')
                    ax_80.set_ylim(0, 130)
                    ax_80.set_xticklabels(df_80['Nutrisi'], rotation=45, ha='right')
                    ax_80.legend(loc='lower right', fontsize='small')
                    ax_80.grid(axis='y', alpha=0.3)
                else:
                    ax_80.text(0.5, 0.5, "Tidak ada data", horizontalalignment='center', verticalalignment='center', transform=ax_80.transAxes)
                    ax_80.set_title('Minimal 80%')
                    ax_80.set_xticks([])
                    ax_80.set_yticks([])


                # Kelompok Nutrisi: Minimal 60% Pemenuhan
                nutrients_60_percent = ['GulaTotal', 'Natrium']
                df_60 = comparison_df[comparison_df['Nutrisi'].isin(nutrients_60_percent)]

                if not df_60.empty:
                    colors_60 = [get_custom_color(row['Nutrisi'], row['Persentase']) for index, row in df_60.iterrows()]
                    ax_60.bar(df_60['Nutrisi'], df_60['Persentase'], color=colors_60, alpha=0.7)

                    # Garis referensi untuk kelompok 60%
                    ax_60.axhline(y=120, color='red', linestyle='--', alpha=0.7, label='Batas Toleransi Maksimal (120%)')
                    ax_60.axhline(y=100, color='blue', linestyle='-', alpha=0.7, label='Target Optimal (100%)')
                    ax_60.axhline(y=60, color='green', linestyle='--', alpha=0.7, label='Minimal Target (60%)')
                    
                    ax_60.set_xlabel('Nutrisi')
                    # ax_60.set_ylabel('Persentase Pemenuhan (%)') # Tidak perlu label Y jika sharey=True
                    ax_60.set_title('Minimal 60%')
                    ax_60.set_ylim(0, 130)
                    ax_60.set_xticklabels(df_60['Nutrisi'], rotation=45, ha='right')
                    ax_60.legend(loc='lower right', fontsize='small')
                    ax_60.grid(axis='y', alpha=0.3)
                else:
                    ax_60.text(0.5, 0.5, "Tidak ada data", horizontalalignment='center', verticalalignment='center', transform=ax_60.transAxes)
                    ax_60.set_title('Minimal 60%')
                    ax_60.set_xticks([])
                    ax_60.set_yticks([])


                # Kelompok Nutrisi: Minimal 40% Pemenuhan
                nutrients_40_percent = ['LemakJenuh', 'Kolesterol']
                df_40 = comparison_df[comparison_df['Nutrisi'].isin(nutrients_40_percent)]

                if not df_40.empty:
                    colors_40 = [get_custom_color(row['Nutrisi'], row['Persentase']) for index, row in df_40.iterrows()]
                    ax_40.bar(df_40['Nutrisi'], df_40['Persentase'], color=colors_40, alpha=0.7)

                    # Garis referensi untuk kelompok 40%
                    ax_40.axhline(y=120, color='red', linestyle='--', alpha=0.7, label='Batas Toleransi Maksimal (120%)')
                    ax_40.axhline(y=100, color='blue', linestyle='-', alpha=0.7, label='Target Optimal (100%)')
                    ax_40.axhline(y=40, color='green', linestyle='--', alpha=0.7, label='Minimal Target (40%)')
                    
                    ax_40.set_xlabel('Nutrisi')
                    # ax_40.set_ylabel('Persentase Pemenuhan (%)') # Tidak perlu label Y jika sharey=True
                    ax_40.set_title('Minimal 40%')
                    ax_40.set_ylim(0, 130)
                    ax_40.set_xticklabels(df_40['Nutrisi'], rotation=45, ha='right')
                    ax_40.legend(loc='lower right', fontsize='small')
                    ax_40.grid(axis='y', alpha=0.3)
                else:
                    ax_40.text(0.5, 0.5, "Tidak ada data", horizontalalignment='center', verticalalignment='center', transform=ax_40.transAxes)
                    ax_40.set_title('Minimal 40%')
                    ax_40.set_xticks([])
                    ax_40.set_yticks([])

                plt.tight_layout()
                st.pyplot(fig_combined)

                st.markdown("""
                ##### Keterangan Warna Barplot

                * **Merah**: Persentase pemenuhan nutrisi masih di bawah target minimal. Perlu perhatian lebih untuk memenuhi kebutuhan nutrisi ini.
                * **Oranye**: Persentase pemenuhan nutrisi sudah di atas target minimal, namun masih di bawah target optimal (100%). Cukup baik, tetapi ada ruang untuk peningkatan.
                * **Hijau**: Persentase pemenuhan nutrisi sudah mencapai atau melebihi target optimal (100%). Menunjukkan pemenuhan nutrisi yang baik.
                """)
                
                # Summary table
                st.markdown('<h2 class="sub-header">📋 Ringkasan Nutrisi</h2>', unsafe_allow_html=True)
                
                # Format the comparison dataframe for display
                display_df = comparison_df.copy()
                display_df['Rekomendasi'] = display_df['Rekomendasi'].round(1)
                display_df['Kebutuhan'] = display_df['Kebutuhan'].round(1)
                display_df['Persentase'] = display_df['Persentase'].round(1).astype(str) + '%'
                
                st.dataframe(display_df, use_container_width=True)
                
                # Health recommendations based on nutrition analysis
                st.markdown('<h2 class="sub-header">💡 Saran Kesehatan</h2>', unsafe_allow_html=True)
                
                recommendations = []
                
                # Check for deficiencies
                min_thresholds_for_deficiency = {
                    'Energi': 80,
                    'Protein': 80,
                    'Serat': 80,
                    'VitaminC': 80,
                    'Kalium': 80,
                    'Magnesium': 80,
                    'Kalsium': 80,
                    'Besi': 80,
                    'GulaTotal': 60,
                    'Natrium': 60,
                    'LemakJenuh': 40,
                    'Kolesterol': 40
                }

                max_thresholds_for_excess = {
                    nutrient: 100 for nutrient in comparison_df['Nutrisi'].unique()
                }

                low_nutrients_found = False
                for _, nutrient in comparison_df.iterrows():
                    nutrient_name = nutrient['Nutrisi']
                    percentage = nutrient['Persentase']

                    threshold = min_thresholds_for_deficiency[nutrient_name] 

                    if percentage < threshold:
                        if not low_nutrients_found:
                            recommendations.append("⚠️ **Nutrisi yang Perlu Ditingkatkan:**")
                            low_nutrients_found = True
                        recommendations.append(f"  - Hasil rekomendasi {nutrient_name}: {percentage:.1f}% dari kebutuhan")
                        recommendations.append(f"    Untuk membantu memenuhi kebutuhan harian minimal akan {nutrient_name.lower()} sebesar {threshold}%, Anda bisa mengubah porsi makanan dengan menambahkan asupan yang kaya {nutrient_name.lower()}. Atau konsumsi suplemen yang dapat menambah asupan {nutrient_name.lower()} namun dengan konsultasi dokter/ahli gizi.")

                if low_nutrients_found:
                    recommendations.append("")
                
                # Check for excess
                high_nutrients_found = False
                for _, nutrient in comparison_df.iterrows():
                    nutrient_name = nutrient['Nutrisi']
                    percentage = nutrient['Persentase']

                    threshold = max_thresholds_for_excess[nutrient_name]

                    if percentage > threshold:
                        if not high_nutrients_found:
                            recommendations.append("⚠️ **Nutrisi yang Berlebihan:**")
                            high_nutrients_found = True
                        recommendations.append(f"  - Hasil rekomendasi {nutrient_name}: {percentage:.1f}% dari kebutuhan")
                        recommendations.append(f"    Nutrisi {nutrient_name} dari hasil rekomendasi masih dalam batas toleransi maksimal 120% meskipun melebihi {threshold}%.")

                if high_nutrients_found:
                    recommendations.append("")
                
                # General recommendations based on diseases
                if 'diabetes' in diseases:
                    recommendations.extend([
                        "🩺 **Saran untuk Diabetes:**",
                        " - Konsumsi makanan tinggi Protein, Serat, Vitamin C",
                        " - Konsumsi makanan rendah Gula, Natrium, Lemak Jenuh, Kolesterol",
                        " - Cek selalu gula darah",
                        " - Kontrol rutin",
                        ""  # Add spacing after this section
                    ])
                
                if 'hipertensi' in diseases:
                    recommendations.extend([
                        "🩺 **Saran untuk Hipertensi:**",
                        " - Konsumsi makanan tinggi Serat, Kalium, Magnesium, Kalsium",
                        " - Konsumsi makanan rendah Natrium, Lemak Jenuh, Kolesterol",
                        " - Jaga berat badan ideal",
                        " - Pantau tekanan darah",
                        ""  # Add spacing after this section
                    ])
                
                if 'cardiovascular disease' in diseases:
                    recommendations.extend([
                        "🩺 **Saran untuk Cardiovascular Disease:**",
                        " - Konsumsi maknan tinggi Protein, Serat, Zat Besi",
                        " - Konsumsi makanan rendah Natrium, Lemak Jenuh, Kolesterol",
                        " - Hindari aktivitas berat",
                        " - Kontrol faktor risiko CVD",
                        ""  # Add spacing after this section
                    ])
                
                if not recommendations:
                    recommendations = [
                        "✅ **Status Nutrisi Baik!**",
                        " Periksa kesehatan rutin",
                        " Jaga pola makan seimbang",
                        " Istirahat cukup",
                        " Aktivitas fisik teratur",
                    ]
                
                for rec in recommendations:
                    st.markdown(rec)
                
            else:
                st.warning("Tidak dapat membuat rekomendasi makanan. Silakan coba dengan parameter yang berbeda.")
    
    # Information about the app
    with st.expander("ℹ️ Tentang Website"):
        st.markdown("""
                
        Website ini menggunakan algoritma K-Nearest Neighbors (KNN) dan pendekatan Multiple Constraint 0/1 Knapsack Problem untuk merekomendasikan 
        makanan yang sesuai berdasarkan:
        
        - **Profil Pengguna**: Usia, jenis kelamin, tinggi badan, berat badan
        - **Kondisi Kesehatan**: Diabetes, hipertensi, Cardiovascular Disease (CVD)
        - **Kebutuhan Nutrisi**: Berdasarkan Angka Kecukupan Gizi (AKG) Indonesia yang disesuaikan dengan kondisi pengguna
        - **Distribusi Waktu Makan**: Sarapan (30%), Makan Siang (40%), Makan Malam (30%)
        
        **Sumber Data**: United States Department of Agriculture (USDA) FoodCentral Database. Dataset diakses melalui https://www.andrafarm.com/. Data USDA dipilih karena merupakan basis data nutrisi terbesar dan terlengkap di dunia, menyediakan detail nutrisi yang lengkap, dan sering digunakan dalam penelitian serta pedoman diet untuk penyakit kronis seperti diabetes, hipertensi, dan CVD.
        
        Website ini disusun oleh:
        - Bachtiar Rizky Alamsyah
        - Dosen Pembimbing: Retno Aulia Vinarti, S.Kom., M.Kom., Ph.D.
        - Ahli Gizi: Tia Monica Wulan Sari, Amd. Gz.
        
        **Catatan**: Rekomendasi ini bersifat umum dan tidak menggantikan konsultasi dengan ahli gizi atau dokter.
        """)

if __name__ == "__main__":
    main()