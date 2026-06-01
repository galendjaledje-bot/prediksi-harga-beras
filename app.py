import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from model import (
    load_data, feature_engineering, train_models,
    evaluate_model, predict_future, get_feature_importance, PROVINSI_LIST
)

st.set_page_config(
    page_title="Prediksi Harga Beras Indonesia",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.main { background-color: #FAFAF8; }
.section-title { font-size: 18px; font-weight: 700; color: #1a1a1a; margin-bottom: 4px; }
.section-sub   { font-size: 13px; color: #888; margin-bottom: 20px; }
.badge-rf { display:inline-block; background:#EFF6EC; color:#2D6A2A; border-radius:20px; padding:3px 12px; font-size:12px; font-weight:600; }
.badge-lr { display:inline-block; background:#EEF2FF; color:#3730A3; border-radius:20px; padding:3px 12px; font-size:12px; font-weight:600; }
[data-testid="stSidebar"] { background: white; border-right: 1px solid #E8E6E0; }
div[data-testid="metric-container"] { background: white; border: 1px solid #E8E6E0; border-radius: 12px; padding: 10px 14px; min-height: 105px; }
   div[data-testid="metric-container"] [data-testid="stMetricLabel"] { font-size: 13px !important; }
   div[data-testid="metric-container"] [data-testid="stMetricValue"] { font-size: 18px !important; font-weight: 700; white-space: nowrap; }
.stTabs [data-baseweb="tab-list"] { gap:4px; background:#F3F2EE; border-radius:10px; padding:4px; }
.stTabs [data-baseweb="tab"] { border-radius:8px; padding:8px 20px; font-weight:500; font-size:14px; }
.stTabs [aria-selected="true"] { background:white !important; color:#1a1a1a !important; }
hr { border-color: #E8E6E0; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_train(filepath="data/Price_Rice_In_Indonesia_2021-2024.csv"):
    df = load_data(filepath)
    df_feat, provinsi_map = feature_engineering(df)
    lr, rf, X_train, X_test, y_train, y_test, train, test = train_models(df_feat)
    eval_lr = evaluate_model(lr, X_test, y_test)
    eval_rf = evaluate_model(rf, X_test, y_test)
    fi_df   = get_feature_importance(rf)
    return df, df_feat, provinsi_map, lr, rf, X_test, y_test, train, test, eval_lr, eval_rf, fi_df


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 Harga Beras")
    st.markdown("Prediksi harga beras **34 provinsi** di Indonesia menggunakan Regresi Linier & Random Forest.")
    st.markdown("---")

    uploaded_file = st.file_uploader("📂 Upload Data (CSV ; separator)", type=["csv"],
        help="Format: Tanggal ; Aceh ; Sumatera Utara ; ...")
    st.markdown("---")

    st.markdown("### ⚙️ Pengaturan")
    n_forecast   = st.slider("Minggu Prediksi ke Depan", 4, 24, 8)
    model_choice = st.radio("Model untuk Prediksi",
                            ["Random Forest 🌲", "Regresi Linier 📈"], index=0)
    st.markdown("---")
    st.caption("Data: Harga Beras Indonesia 2021–2024")


# ── Load ─────────────────────────────────────────────────────
if uploaded_file:
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        df, df_feat, provinsi_map, lr, rf, X_test, y_test, train, test, eval_lr, eval_rf, fi_df = load_and_train(tmp_path)
        st.sidebar.success("✅ Data berhasil dimuat!")
    except Exception as e:
        st.sidebar.error(f"❌ Error: {e}")
        df, df_feat, provinsi_map, lr, rf, X_test, y_test, train, test, eval_lr, eval_rf, fi_df = load_and_train()
    finally:
        os.unlink(tmp_path)
else:
    df, df_feat, provinsi_map, lr, rf, X_test, y_test, train, test, eval_lr, eval_rf, fi_df = load_and_train()

active_model = rf if "Random Forest" in model_choice else lr
active_eval  = eval_rf if "Random Forest" in model_choice else eval_lr
active_name  = "Random Forest" if "Random Forest" in model_choice else "Regresi Linier"
provinsi_list = sorted(df['provinsi'].unique().tolist())


# ── Header ────────────────────────────────────────────────────
st.markdown("## 🌾 Prediksi Harga Beras Indonesia")
st.markdown("Analisis harga beras **34 provinsi** · 2021–2024 · Regresi Linier & Random Forest")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard", "🔮 Prediksi", "🧪 Evaluasi Model", "📁 Data"
])


# ════════════════════════════════
# TAB 1 — Dashboard
# ════════════════════════════════
with tab1:
    avg_harga   = df['harga'].mean()
    max_harga   = df['harga'].max()
    min_harga   = df['harga'].min()
    prov_termahal = df.groupby('provinsi')['harga'].mean().idxmax()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rata-rata Nasional", f"Rp {avg_harga:,.0f}")
    c2.metric("Harga Tertinggi",    f"Rp {max_harga:,.0f}")
    c3.metric("Harga Terendah",     f"Rp {min_harga:,.0f}")
    c4.metric("Provinsi Termahal",  prov_termahal)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tren rata-rata nasional
    st.markdown('<div class="section-title">Tren Rata-rata Harga Nasional</div>', unsafe_allow_html=True)
    national = df.groupby('tanggal')['harga'].mean().reset_index()
    fig_nat = go.Figure()
    fig_nat.add_trace(go.Scatter(
        x=national['tanggal'], y=national['harga'],
        mode='lines', line=dict(color='#2D6A4F', width=2.5),
        fill='tozeroy', fillcolor='rgba(45,106,79,0.08)',
        hovertemplate="%{x|%d %b %Y}<br>Rp %{y:,.0f}<extra>Nasional</extra>"
    ))
    fig_nat.update_layout(
        plot_bgcolor='white', paper_bgcolor='white', height=300,
        yaxis=dict(title='Harga (Rp)', tickformat=',.0f', gridcolor='#F0EEE8'),
        xaxis=dict(gridcolor='#F0EEE8'),
        margin=dict(l=0,r=0,t=10,b=0)
    )
    st.plotly_chart(fig_nat, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-title">Tren per Provinsi (pilih)</div>', unsafe_allow_html=True)
        sel_provs = st.multiselect(
            "Pilih Provinsi", provinsi_list,
            default=['Jawa Barat', 'DKI Jakarta', 'Jawa Timur', 'Sulawesi Selatan']
        )
        fig_prov = go.Figure()
        palette = px.colors.qualitative.Set2
        for i, prov in enumerate(sel_provs):
            sub = df[df['provinsi'] == prov].sort_values('tanggal')
            fig_prov.add_trace(go.Scatter(
                x=sub['tanggal'], y=sub['harga'], name=prov,
                mode='lines', line=dict(width=1.8, color=palette[i % len(palette)]),
                hovertemplate=f"<b>{prov}</b><br>%{{x|%d %b %Y}}<br>Rp %{{y:,.0f}}<extra></extra>"
            ))
        fig_prov.update_layout(
            plot_bgcolor='white', paper_bgcolor='white', height=320,
            yaxis=dict(tickformat=',.0f', gridcolor='#F0EEE8'),
            xaxis=dict(gridcolor='#F0EEE8'),
            legend=dict(orientation='h', y=-0.25),
            margin=dict(l=0,r=0,t=10,b=0)
        )
        st.plotly_chart(fig_prov, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Rata-rata Harga per Provinsi (terbaru)</div>', unsafe_allow_html=True)
        latest_avg = df.groupby('provinsi')['harga'].mean().sort_values(ascending=True)
        fig_bar = go.Figure(go.Bar(
            x=latest_avg.values,
            y=latest_avg.index,
            orientation='h',
            marker_color='#2D6A4F',
            hovertemplate="%{y}<br>Rp %{x:,.0f}<extra></extra>"
        ))
        fig_bar.update_layout(
            plot_bgcolor='white', paper_bgcolor='white', height=600,
            xaxis=dict(tickformat=',.0f', gridcolor='#F0EEE8'),
            yaxis=dict(tickfont=dict(size=11)),
            margin=dict(l=0,r=0,t=10,b=0)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Heatmap harga rata2 per provinsi per tahun
    st.markdown('<div class="section-title">Heatmap Harga Rata-rata per Provinsi per Tahun</div>', unsafe_allow_html=True)
    df['tahun'] = df['tanggal'].dt.year
    heat_data = df.groupby(['tahun', 'provinsi'])['harga'].mean().unstack()
    fig_heat = px.imshow(
        heat_data.T,
        color_continuous_scale='YlGn',
        labels=dict(color='Harga (Rp)'),
        aspect='auto'
    )
    fig_heat.update_layout(
        plot_bgcolor='white', paper_bgcolor='white', height=500,
        margin=dict(l=0,r=0,t=10,b=0),
        coloraxis_colorbar=dict(tickformat=',.0f')
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ════════════════════════════════
# TAB 2 — Prediksi
# ════════════════════════════════
with tab2:
    st.markdown(f'<div class="section-title">Prediksi {n_forecast} Minggu ke Depan per Provinsi</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">Model aktif: <span class="badge-rf">{active_name}</span></div>', unsafe_allow_html=True)

    prov_sel = st.selectbox("Pilih Provinsi", provinsi_list, index=provinsi_list.index('Jawa Barat') if 'Jawa Barat' in provinsi_list else 0)

    pred_df = predict_future(active_model, df_feat, provinsi_map, prov_sel, n_steps=n_forecast)

    hist = df[df['provinsi'] == prov_sel].sort_values('tanggal').tail(90)

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=hist['tanggal'], y=hist['harga'],
        name='Historis', mode='lines',
        line=dict(color='#2D6A4F', width=2.5),
        hovertemplate="%{x|%d %b %Y}<br>Rp %{y:,.0f}<extra>Historis</extra>"
    ))
    fig_pred.add_trace(go.Scatter(
        x=pred_df['tanggal'], y=pred_df['harga_prediksi'],
        name='Prediksi', mode='lines+markers',
        line=dict(color='#E76F51', width=2.5, dash='dash'),
        marker=dict(size=7, color='#E76F51'),
        hovertemplate="%{x|%d %b %Y}<br>Rp %{y:,.0f}<extra>Prediksi</extra>"
    ))
    fig_pred.add_trace(go.Scatter(
        x=[hist['tanggal'].iloc[-1], pred_df['tanggal'].iloc[0]],
        y=[hist['harga'].iloc[-1], pred_df['harga_prediksi'].iloc[0]],
        mode='lines', line=dict(color='#E76F51', width=1.5, dash='dot'),
        showlegend=False, hoverinfo='skip'
    ))
    fig_pred.add_vrect(
        x0=pred_df['tanggal'].iloc[0], x1=pred_df['tanggal'].iloc[-1],
        fillcolor='#FFF3EE', opacity=0.5, line_width=0,
        annotation_text='Periode Prediksi', annotation_position='top left',
        annotation_font_size=11, annotation_font_color='#E76F51'
    )
    fig_pred.update_layout(
        plot_bgcolor='white', paper_bgcolor='white', height=400,
        yaxis=dict(title='Harga (Rp)', tickformat=',.0f', gridcolor='#F0EEE8'),
        xaxis=dict(gridcolor='#F0EEE8'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=0,r=0,t=30,b=0)
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    # Tabel prediksi
    st.markdown("#### Tabel Hasil Prediksi")
    pred_display = pred_df.copy()
    pred_display['tanggal'] = pred_display['tanggal'].dt.strftime('%d %B %Y')
    pred_display['harga_prediksi'] = pred_display['harga_prediksi'].apply(lambda x: f"Rp {x:,.0f}")
    pred_display.columns = ['Tanggal', 'Harga Prediksi']
    st.dataframe(pred_display, use_container_width=True, hide_index=True)

    # Prediksi semua provinsi
    st.markdown("---")
    st.markdown("#### Prediksi Minggu Depan — Semua Provinsi")
    next_week_rows = []
    for prov in provinsi_list:
        try:
            p = predict_future(active_model, df_feat, provinsi_map, prov, n_steps=1)
            last_h = df[df['provinsi']==prov]['harga'].iloc[-1]
            pred_h = p['harga_prediksi'].iloc[0]
            next_week_rows.append({
                'Provinsi': prov,
                'Harga Terakhir': f"Rp {last_h:,.0f}",
                'Prediksi Minggu Depan': f"Rp {pred_h:,.0f}",
                'Selisih': f"{'▲' if pred_h >= last_h else '▼'} Rp {abs(pred_h - last_h):,.0f}"
            })
        except:
            pass
    st.dataframe(pd.DataFrame(next_week_rows), use_container_width=True, hide_index=True)


# ════════════════════════════════
# TAB 3 — Evaluasi
# ════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Perbandingan Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Metrik evaluasi pada data test (time-based split per provinsi)</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('**<span class="badge-lr">Regresi Linier</span>**', unsafe_allow_html=True)
        a, b = st.columns(2)
        a.metric("MAE",  f"Rp {eval_lr['MAE']:,.0f}")
        b.metric("RMSE", f"Rp {eval_lr['RMSE']:,.0f}")
        c, d = st.columns(2)
        c.metric("R²",   f"{eval_lr['R²']:.4f}")
        d.metric("MAPE", f"{eval_lr['MAPE (%)']:.2f}%")

    with col2:
        st.markdown('**<span class="badge-rf">Random Forest</span>**', unsafe_allow_html=True)
        a, b = st.columns(2)
        a.metric("MAE",  f"Rp {eval_rf['MAE']:,.0f}")
        b.metric("RMSE", f"Rp {eval_rf['RMSE']:,.0f}")
        c, d = st.columns(2)
        c.metric("R²",   f"{eval_rf['R²']:.4f}")
        d.metric("MAPE", f"{eval_rf['MAPE (%)']:.2f}%")

    st.markdown("---")
    st.markdown("#### Aktual vs Prediksi (Data Test — sample 300 titik)")
    sample_idx = np.linspace(0, len(y_test)-1, min(300, len(y_test)), dtype=int)
    y_act_s = y_test.values[sample_idx]

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Scatter(y=y_act_s, name='Aktual', mode='lines',
        line=dict(color='#1a1a1a', width=2)))
    fig_cmp.add_trace(go.Scatter(y=eval_rf['y_pred'][sample_idx], name='Random Forest',
        mode='lines', line=dict(color='#2D6A4F', width=1.8, dash='dash')))
    fig_cmp.add_trace(go.Scatter(y=eval_lr['y_pred'][sample_idx], name='Regresi Linier',
        mode='lines', line=dict(color='#3730A3', width=1.8, dash='dot')))
    fig_cmp.update_layout(
        plot_bgcolor='white', paper_bgcolor='white', height=380,
        yaxis=dict(title='Harga (Rp)', tickformat=',.0f', gridcolor='#F0EEE8'),
        xaxis=dict(title='Indeks Data Test'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=0,r=0,t=30,b=0)
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown("#### Feature Importance — Random Forest")
    fig_fi = px.bar(fi_df, x='Importance', y='Fitur', orientation='h',
        color='Importance', color_continuous_scale=[[0,'#D4EDDA'],[1,'#2D6A4F']])
    fig_fi.update_layout(
        plot_bgcolor='white', paper_bgcolor='white', height=360,
        yaxis=dict(autorange='reversed'), coloraxis_showscale=False,
        margin=dict(l=0,r=0,t=10,b=0)
    )
    st.plotly_chart(fig_fi, use_container_width=True)


# ════════════════════════════════
# TAB 4 — Data
# ════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Dataset Harga Beras per Provinsi</div>', unsafe_allow_html=True)

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        prov_filter = st.multiselect("Filter Provinsi", provinsi_list, default=provinsi_list[:5])
    with col_f2:
        min_d = df['tanggal'].min().date()
        max_d = df['tanggal'].max().date()
        date_range = st.date_input("Rentang Tanggal", value=(min_d, max_d), min_value=min_d, max_value=max_d)

    filtered = df[df['provinsi'].isin(prov_filter)]
    if len(date_range) == 2:
        filtered = filtered[
            (filtered['tanggal'].dt.date >= date_range[0]) &
            (filtered['tanggal'].dt.date <= date_range[1])
        ]

    st.markdown(f"**{len(filtered):,}** baris ditampilkan")
    disp = filtered[['tanggal','provinsi','harga']].sort_values('tanggal', ascending=False).head(500).copy()
    disp['tanggal'] = disp['tanggal'].dt.strftime('%d %b %Y')
    disp['harga']   = disp['harga'].apply(lambda x: f"Rp {x:,.0f}")
    disp.columns    = ['Tanggal','Provinsi','Harga']
    st.dataframe(disp, use_container_width=True, hide_index=True)

    st.markdown("#### Statistik Deskriptif per Provinsi")
    desc = df.groupby('provinsi')['harga'].describe().round(0)
    desc.columns = ['Count','Mean','Std','Min','25%','50%','75%','Max']
    st.dataframe(desc, use_container_width=True)
