import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

PROVINSI_LIST = [
    'Aceh', 'Sumatera Utara', 'Sumatera Barat', 'Riau', 'Kepulauan Riau',
    'Jambi', 'Bengkulu', 'Sumatera Selatan', 'Kepulauan Bangka Belitung',
    'Lampung', 'Banten', 'Jawa Barat', 'DKI Jakarta', 'Jawa Tengah',
    'DI Yogyakarta', 'Jawa Timur', 'Bali', 'Nusa Tenggara Barat',
    'Nusa Tenggara Timur', 'Kalimantan Barat', 'Kalimantan Selatan',
    'Kalimantan Tengah', 'Kalimantan Timur', 'Kalimantan Utara', 'Gorontalo',
    'Sulawesi Selatan', 'Sulawesi Tenggara', 'Sulawesi Tengah',
    'Sulawesi Utara', 'Sulawesi Barat', 'Maluku', 'Maluku Utara',
    'Papua', 'Papua Barat'
]

FEATURE_COLS = [
    'provinsi_encoded', 'tahun', 'bulan', 'minggu', 'day_of_year',
    'lag_1', 'lag_2', 'lag_4', 'rolling_mean_4', 'rolling_std_4', 'harga_diff'
]


def load_data(filepath="data/Price_Rice_In_Indonesia_2021-2024.csv"):
    df_wide = pd.read_csv(filepath, sep=';')
    df_wide['Tanggal'] = pd.to_datetime(df_wide['Tanggal'], dayfirst=True)

    # Wide → Long (melt per provinsi)
    provinsi_cols = [c for c in df_wide.columns if c != 'Tanggal']
    df = df_wide.melt(id_vars='Tanggal', value_vars=provinsi_cols,
                      var_name='provinsi', value_name='harga')
    df = df.dropna(subset=['harga'])
    df = df.sort_values(['provinsi', 'Tanggal']).reset_index(drop=True)
    df = df.rename(columns={'Tanggal': 'tanggal'})
    return df


def feature_engineering(df):
    df = df.copy()

    # Encode provinsi
    provinsi_map = {p: i for i, p in enumerate(sorted(df['provinsi'].unique()))}
    df['provinsi_encoded'] = df['provinsi'].map(provinsi_map)

    # Fitur waktu
    df['tahun'] = df['tanggal'].dt.year
    df['bulan'] = df['tanggal'].dt.month
    df['minggu'] = df['tanggal'].dt.isocalendar().week.astype(int)
    df['day_of_year'] = df['tanggal'].dt.dayofyear

    # Lag features per provinsi
    for lag in [1, 2, 4]:
        df[f'lag_{lag}'] = df.groupby('provinsi')['harga'].shift(lag)

    # Rolling statistics per provinsi
    df['rolling_mean_4'] = df.groupby('provinsi')['harga'].transform(
        lambda x: x.shift(1).rolling(4).mean()
    )
    df['rolling_std_4'] = df.groupby('provinsi')['harga'].transform(
        lambda x: x.shift(1).rolling(4).std()
    )
    df['harga_diff'] = df.groupby('provinsi')['harga'].diff()

    df = df.dropna().reset_index(drop=True)
    return df, provinsi_map


def train_models(df_feat, test_ratio=0.2):
    split_train, split_test = [], []

    for prov in df_feat['provinsi'].unique():
        subset = df_feat[df_feat['provinsi'] == prov].copy()
        n = len(subset)
        idx = int(n * (1 - test_ratio))
        split_train.append(subset.iloc[:idx])
        split_test.append(subset.iloc[idx:])

    train = pd.concat(split_train).reset_index(drop=True)
    test = pd.concat(split_test).reset_index(drop=True)

    X_train = train[FEATURE_COLS]
    y_train = train['harga']
    X_test = test[FEATURE_COLS]
    y_test = test['harga']

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    rf = RandomForestRegressor(
        n_estimators=200, max_depth=10,
        min_samples_split=5, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    return lr, rf, X_train, X_test, y_train, y_test, train, test


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    return {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R²': round(r2, 4),
        'MAPE (%)': round(mape, 2),
        'y_pred': y_pred
    }


def predict_future(model, df_feat, provinsi_map, provinsi, n_steps=8):
    subset = df_feat[df_feat['provinsi'] == provinsi].copy()
    last_prices = list(subset['harga'].values[-4:])
    last_date = subset['tanggal'].iloc[-1]
    enc = provinsi_map[provinsi]

    predictions = []
    for i in range(1, n_steps + 1):
        next_date = last_date + pd.Timedelta(days=7 * i)
        lag1 = last_prices[-1]
        lag2 = last_prices[-2] if len(last_prices) >= 2 else lag1
        lag4 = last_prices[-4] if len(last_prices) >= 4 else lag1
        roll_mean = np.mean(last_prices[-4:])
        roll_std  = np.std(last_prices[-4:])
        diff = last_prices[-1] - last_prices[-2] if len(last_prices) >= 2 else 0

        row = {
            'provinsi_encoded': enc,
            'tahun': next_date.year,
            'bulan': next_date.month,
            'minggu': next_date.isocalendar()[1],
            'day_of_year': next_date.timetuple().tm_yday,
            'lag_1': lag1, 'lag_2': lag2, 'lag_4': lag4,
            'rolling_mean_4': roll_mean, 'rolling_std_4': roll_std,
            'harga_diff': diff,
        }
        pred = model.predict(pd.DataFrame([row]))[0]
        predictions.append({'tanggal': next_date, 'harga_prediksi': round(pred, 0)})
        last_prices.append(pred)

    return pd.DataFrame(predictions)


def get_feature_importance(rf_model):
    return pd.DataFrame({
        'Fitur': FEATURE_COLS,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
