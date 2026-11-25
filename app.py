import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- KONFIGURACJA STRONY ---
st.set_page_config(
    page_title="MEVO Executive BI – Analiza Sprzedaży, Klientów i Budżetu",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLIZACJA CSS (WYGLĄD PREMIUM BI) ---
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    div[data-testid="stMetric"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        padding: 16px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    div[data-testid="stMetric"] label {
        color: #8B949E !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #F0F6FC !important;
        font-size: 1.85rem !important;
        font-weight: 700 !important;
    }
    .insight-box {
        background-color: #161B22;
        border-left: 4px solid #00B4D8;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 16px;
        color: #C9D1D9;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# --- POBIERANIE I TRANSFORMACJA DANYCH (ETL) ---
@st.cache_data(ttl=60)
def load_and_model_data():
    gbfs_index = requests.get("https://gbfs.urbansharing.com/rowermevo.pl/gbfs.json", timeout=10).json()
    feeds = {f["name"]: f["url"] for f in gbfs_index["data"]["pl"]["feeds"]}

    stations_info = pd.DataFrame(requests.get(feeds["station_information"], timeout=10).json()["data"]["stations"])
    stations_status = pd.DataFrame(requests.get(feeds["station_status"], timeout=10).json()["data"]["stations"])
    free_bikes = pd.DataFrame(requests.get(feeds["free_bike_status"], timeout=10).json()["data"]["bikes"])

    # 1. Modelowanie tabeli stacji
    stations = pd.merge(stations_info, stations_status, on="station_id", how="inner")
    stations["oblozenie_pct"] = np.where(
        stations["capacity"] > 0,
        (stations["num_bikes_available"] / stations["capacity"] * 100).round(1),
        0
    )

    # Wyciągnięcie miasta z nazwy/adresu stacji (np. GDA, GDY, SOP)
    def extract_city(name):
        prefix = str(name).split("-")[0].strip().upper()
        mapping = {
            "GDA": "Gdańsk", "GDY": "Gdynia", "SOP": "Sopot",
            "RUM": "Rumia", "RED": "Reda", "WEJ": "Wejherowo",
            "PRU": "Pruszcz Gdański", "TCZ": "Tczew", "PUK": "Puck",
            "WLA": "Władysławowo", "ZUK": "Żukowo", "KOL": "Kolbudy",
            "STE": "Stężyca", "SOM": "Somonino", "KOS": "Kościerzyna", "SIE": "Sierakowice"
        }
        return mapping.get(prefix, "Inne / Metropolia")

    stations["Miasto"] = stations["name"].apply(extract_city)

    # Segmentacja klientów na podstawie charakterystyki lokalizacji stacji
    def classify_customer_segment(name):
        n = str(name).lower()
        if any(k in n for k in ["skm", "pkm", "dworzec", "główny", "wrzeszcz", "oliwa", "politechnika", "uniwersytet", "alchemia", "olivia"]):
            return "Commuter (Dojazdy Praca/Uczelnia)"
        elif any(k in n for k in ["plaża", "molo", "brzeźno", "jelitkowo", "orłowo", "bulwar", "park", "stogi", "sobieszewo", "nadmorski"]):
            return "Leisure (Rekreacja & Turystyka)"
        else:
            return "Residential (Ruch Lokalny / Dzielnice)"

    stations["Segment_Klienta"] = stations["name"].apply(classify_customer_segment)

    # Status operacyjny stacji (wykrywanie utraconej sprzedaży)
    def station_health(row):
        if row["num_bikes_available"] == 0:
            return "Brak towaru (Stockout)"
        elif row["oblozenie_pct"] >= 85:
            return "Nadpodaż (>85%)"
        else:
            return "Optymalny (1–84%)"

    stations["Status_Zapasu"] = stations.apply(station_health, axis=1)

    # 2. Czyszczenie i modelowanie floty rowerowej
    bikes = free_bikes[
        (free_bikes["lat"].between(53.8, 55.0)) &
        (free_bikes["lon"].between(17.5, 19.5))
    ].copy()
    bikes["zasieg_km"] = (bikes["current_range_meters"] / 1000).fillna(0).round(1)
    bikes["Typ_Produktu"] = bikes["vehicle_type_id"].map({"ebike": "E-Bike (Premium)", "bike": "Klasyczny (Standard)"}).fillna("Inny")

    return stations, bikes, datetime.now().strftime("%Y-%m-%d %H:%M:%S")

stations_df, bikes_df, updated_at = load_and_model_data()

# --- OBLICZENIA BIZNESOWE (FLEET & REVENUE MODEL) ---
total_fleet_declared = 4099  # Pełna skala floty MEVO w metropolii
bikes_parked_free = len(bikes_df[(~bikes_df["is_disabled"]) & (~bikes_df["is_reserved"])])
bikes_disabled = len(bikes_df[bikes_df["is_disabled"] == True])
bikes_reserved = len(bikes_df[bikes_df["is_reserved"] == True])
# Szacunek rowerów aktualnie wypożyczonych (w trasie)
bikes_in_use = max(total_fleet_declared - len(bikes_df), 0)

# --- PANEL BOCZNY (PARAMETRY MODELU FINANSOWEGO I ROWEROWEGO) ---
st.sidebar.title("🎛️ Centrum Sterowania BI")
st.sidebar.caption("Parametry symulacji budżetowej i operacyjnej:")

selected_cities = st.sidebar.multiselect(
    "Filtruj gminy (Obszar Metropolitalny):",
    options=sorted(stations_df["Miasto"].unique()),
    default=["Gdańsk", "Gdynia", "Sopot"]
)

st.sidebar.divider()
st.sidebar.subheader("💰 Założenia Budżetowe (What-If)")
avg_rev_per_hour = st.sidebar.slider("Średni przychód z 1 przejazdu / h (PLN):", 4.0, 15.0, 7.5, step=0.5)
target_utilization_pct = st.sidebar.slider("Budżetowy cel utylizacji floty (%):", 5.0, 40.0, 15.0, step=1.0)

st.sidebar.divider()
st.sidebar.subheader("🚲 Profil Rowerzysty (Autor)")
min_range = st.sidebar.slider("Minimalny zasięg e-bike'a na trasę (km):", 0, 100, 35, step=5)

# Filtrowanie stacji po wybranych miastach
if selected_cities:
    filtered_stations = stations_df[stations_df["Miasto"].isin(selected_cities)].copy()
else:
    filtered_stations = stations_df.copy()

# --- NAGŁÓWEK RAPORTU ---
st.title("📊 MEVO Executive BI – Sprzedaż, Klienci & Budżet")
st.markdown(
    f"Model analityczny przekładający dane telemetryczne IoT (GBFS) na wskaźniki biznesowe: **realizację budżetu, utylizację zasobów i segmentację bazy klientów**. "
    f"• *Live feed:* `{updated_at}`"
)

# --- GŁÓWNE KARTY KPI ---
actual_utilization_pct = round((bikes_in_use / total_fleet_declared) * 100, 1)
delta_utilization = round(actual_utilization_pct - target_utilization_pct, 1)

hourly_revenue_run_rate = round(bikes_in_use * avg_rev_per_hour, 0)
budget_hourly_target = round((total_fleet_declared * (target_utilization_pct / 100)) * avg_rev_per_hour, 0)
budget_realization_pct = round((hourly_revenue_run_rate / budget_hourly_target * 100), 1) if budget_hourly_target > 0 else 0

stockout_stations = len(filtered_stations[filtered_stations["num_bikes_available"] == 0])
stockout_rate = round((stockout_stations / len(filtered_stations) * 100), 1) if len(filtered_stations) > 0 else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Run-Rate Przychodu (PLN / h)",
    f"{hourly_revenue_run_rate:,.0f} zł".replace(",", " "),
    f"{budget_realization_pct}% planu budżetowego"
)
k2.metric(
    "Utylizacja Floty (W trasie)",
    f"{actual_utilization_pct}%",
    f"{delta_utilization:+.1f} p.p. vs Budżet ({target_utilization_pct}%)"
)
k3.metric(
    "Ryzyko Utraconej Sprzedaży",
    f"{stockout_stations} stacji",
    f"{stockout_rate}% pustych stacji (0 rowerów)",
    delta_color="inverse"
)
k4.metric(
    "Dostępne Zasoby (Wybrany obszar)",
    f"{int(filtered_stations['num_bikes_available'].sum()):,}".replace(",", " "),
    f"Pojemność: {int(filtered_stations['capacity'].sum()):,} miejsc".replace(",", " ")
)

st.write("")

# --- ZAKŁADKI ANALITYCZNE ---
tab1, tab2, tab3 = st.tabs([
    "💰 Wyniki Sprzedaży & Budżet",
    "👥 Baza Klientów & Segmentacja",
    "🚲 Operacje & Okiem Rowerzysty"
])

# ==============================================================================
# ZAKŁADKA 1: WYNIKI SPRZEDAŻY I REALIZACJA BUDŻETU
# ==============================================================================
with tab1:
    st.markdown(
        f"""<div class="insight-box">
        <b>Executive Summary:</b> Przy założonej cenie jednostkowej <b>{avg_rev_per_hour:.2f} PLN/h</b> oraz celu utylizacji <b>{target_utilization_pct}%</b>, 
        bieżący godzinowy potencjał przychodowy wynosi <b>{hourly_revenue_run_rate:,.0f} PLN</b> (realizacja planu: <b>{budget_realization_pct}%</b>). 
        Zidentyfikowano <b>{stockout_stations}</b> pustych stacji, które generują potencjalną utratę przychodu (brak dostępnego produktu dla klienta).
        </div>""".replace(",", " "),
        unsafe_allow_html=True
    )

    c1, c2 = st.columns([1.15, 1])

    with c1:
        st.subheader("📉 Dekompozycja Zasobów Floty (Waterfall Chart)")
        fig_waterfall = go.Figure(go.Waterfall(
            name="Flota",
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "total"],
            x=["Flota Całkowita", "Wolne (Na stacjach/ulicy)", "Serwis / Rozładowane", "Zarezerwowane", "Aktywnie Zarabiające (W trasie)"],
            textposition="outside",
            text=[
                f"{total_fleet_declared}",
                f"-{bikes_parked_free}",
                f"-{bikes_disabled}",
                f"-{bikes_reserved}",
                f"{bikes_in_use}"
            ],
            y=[total_fleet_declared, -bikes_parked_free, -bikes_disabled, -bikes_reserved, bikes_in_use],
            connector={"line": {"color": "#30363D"}},
            decreasing={"marker": {"color": "#E63946"}},
            increasing={"marker": {"color": "#2A9D8F"}},
            totals={"marker": {"color": "#00B4D8"}}
        ))
        fig_waterfall.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=430,
            margin=dict(t=30, b=20, l=10, r=10),
            yaxis_title="Liczba rowerów"
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)

    with c2:
        st.subheader("🏙️ Potencjał Podaży wg Gmin (Pojemność vs Dostępność)")
        city_agg = stations_df.groupby("Miasto", as_index=False).agg(
            Dostepne=("num_bikes_available", "sum"),
            Pojemnosc=("capacity", "sum"),
            Liczba_Stacji=("station_id", "count")
        ).sort_values("Pojemnosc", ascending=True).tail(8)

        fig_city = go.Figure()
        fig_city.add_trace(go.Bar(
            y=city_agg["Miasto"], x=city_agg["Pojemnosc"],
            name="Budżet / Pojemność stacji", orientation="h",
            marker_color="#30363D"
        ))
        fig_city.add_trace(go.Bar(
            y=city_agg["Miasto"], x=city_agg["Dostepne"],
            name="Aktualnie dostępne rowery", orientation="h",
            marker_color="#00B4D8",
            text=city_agg["Dostepne"], textposition="auto"
        ))
        fig_city.update_layout(
            barmode="overlay",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=430,
            margin=dict(t=30, b=20, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_city, use_container_width=True)

# ==============================================================================
# ZAKŁADKA 2: BAZA KLIENTÓW I SEGMENTACJA POPYTU
# ==============================================================================
with tab2:
    s1, s2 = st.columns([1, 1.3])

    with s1:
        st.subheader("🎯 Struktura Bazy Lokalizacji wg Segmentu Klienta")
        seg_df = filtered_stations.groupby("Segment_Klienta", as_index=False).agg(
            Liczba_Stacji=("station_id", "count"),
            Dostepne_Rowery=("num_bikes_available", "sum"),
            Srednie_Oblozenie=("oblozenie_pct", "mean")
        )
        fig_donut = px.pie(
            seg_df,
            values="Dostepne_Rowery",
            names="Segment_Klienta",
            hole=0.55,
            color_discrete_sequence=["#00B4D8", "#2A9D8F", "#F4A261"]
        )
        fig_donut.update_traces(textposition="inside", textinfo="percent+label")
        fig_donut.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            height=420,
            margin=dict(t=20, b=20, l=10, r=10)
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with s2:
        st.subheader("🔬 Macierz Efektywności Stacji (Pojemność vs Obłożenie %)")
        fig_scatter = px.scatter(
            filtered_stations,
            x="capacity",
            y="oblozenie_pct",
            color="Segment_Klienta",
            size="num_bikes_available",
            hover_name="name",
            hover_data={"Miasto": True, "num_bikes_available": True, "capacity": True, "oblozenie_pct": True},
            color_discrete_sequence=["#00B4D8", "#2A9D8F", "#F4A261"],
            labels={
                "capacity": "Pojemność stacji (liczba stojaków)",
                "oblozenie_pct": "Wypełnienie stacji (%)",
                "Segment_Klienta": "Segment"
            }
        )
        fig_scatter.add_hline(y=85, line_dash="dot", line_color="#F4A261", annotation_text="Próg nadpodaży (85%)")
        fig_scatter.add_hline(y=10, line_dash="dot", line_color="#E63946", annotation_text="Strefa ryzyka braku rowerów (<10%)")
        fig_scatter.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=420,
            margin=dict(t=20, b=20, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Tabela TOP stacji z wyborem kryterium biznesowego
    st.subheader("🏆 Ranking Stacji – Analiza Szczegółowa")
    ranking_mode = st.radio(
        "Wybierz perspektywę rankingu:",
        ["Największa dostępność (TOP 10)", "Najwyższe obłożenie % (TOP 10)", "Krytyczne braki – Puste stacje (0 rowerów)"],
        horizontal=True
    )

    if ranking_mode == "Największa dostępność (TOP 10)":
        table_df = filtered_stations.sort_values("num_bikes_available", ascending=False).head(10)
    elif ranking_mode == "Najwyższe obłożenie % (TOP 10)":
        table_df = filtered_stations.sort_values(["oblozenie_pct", "num_bikes_available"], ascending=False).head(10)
    else:
        table_df = filtered_stations[filtered_stations["num_bikes_available"] == 0].sort_values("capacity", ascending=False).head(15)

    st.dataframe(
        table_df[["name", "Miasto", "Segment_Klienta", "num_bikes_available", "capacity", "oblozenie_pct", "Status_Zapasu"]].rename(columns={
            "name": "Nazwa Stacji",
            "Segment_Klienta": "Segment Klienta",
            "num_bikes_available": "Dostępne Rowery",
            "capacity": "Pojemność",
            "oblozenie_pct": "Obłożenie (%)",
            "Status_Zapasu": "Status Operacyjny"
        }),
        use_container_width=True,
        hide_index=True
    )

# ==============================================================================
# ZAKŁADKA 3: OPERACJE & PERSPEKTYWA ROWERZYSTY
# ==============================================================================
with tab3:
    op1, op2 = st.columns([1.2, 1])

    with op1:
        st.subheader("📍 Rozkład Przestrzenny Stacji w Metropolii (Współrzędne GPS)")
        fig_geo = px.scatter(
            filtered_stations,
            x="lon",
            y="lat",
            size="num_bikes_available",
            color="Miasto",
            hover_name="name",
            hover_data={"num_bikes_available": True, "capacity": True, "oblozenie_pct": True, "lon": False, "lat": False},
            color_discrete_sequence=px.colors.qualitative.Bold,
            labels={"lon": "Długość geograficzna (E)", "lat": "Szerokość geograficzna (N)"}
        )
        fig_geo.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#161B22",
            height=450,
            margin=dict(t=20, b=20, l=10, r=10),
            xaxis=dict(showgrid=True, gridcolor="#30363D"),
            yaxis=dict(showgrid=True, gridcolor="#30363D", scaleanchor="x", scaleratio=1.6)
        )
        st.plotly_chart(fig_geo, use_container_width=True)

    with op2:
        st.subheader(f"🔋 Gotowość Floty E-Bike na Trasę ≥ {min_range} km")
        ebikes_avail = bikes_df[(bikes_df["vehicle_type_id"] == "ebike") & (~bikes_df["is_disabled"])].copy()
        ready_count = len(ebikes_avail[ebikes_avail["zasieg_km"] >= min_range])
        low_count = len(ebikes_avail) - ready_count

        fig_bat = px.histogram(
            ebikes_avail,
            x="zasieg_km",
            nbins=20,
            color_discrete_sequence=["#2A9D8F"],
            labels={"zasieg_km": "Szacowany zasięg baterii (km)", "count": "Liczba e-bike'ów"}
        )
        fig_bat.add_vline(
            x=min_range,
            line_dash="dash",
            line_color="#E63946",
            annotation_text=f"Twój próg: {min_range} km ({ready_count} szt. gotowych)"
        )
        fig_bat.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=450,
            bargap=0.1,
            yaxis_title="Liczba rowerów",
            margin=dict(t=20, b=20, l=10, r=10)
        )
        st.plotly_chart(fig_bat, use_container_width=True)

    st.info(
        f"🚴‍♂️ **Komentarz autora (perspektywa rowerzysty):** Jako osoba regularnie jeżdżąca na rowerze po Trójmieście wiem, że sama liczba dostępnych rowerów na stacji bywa myląca. "
        f"Obecnie z **{len(ebikes_avail)}** sprawnych rowerów elektrycznych w systemie, dokładnie **{ready_count}** ({round(ready_count/max(len(ebikes_avail),1)*100, 1)}%) "
        f"posiada baterię pozwalającą na przejechanie minimum **{min_range} km** bez ryzyka rozładowania w połowie trasy."
    )

# --- STOPKA ---
st.divider()
st.caption("Autor: Sylwester Zaborowski | Źródło danych: Otwarte Dane Systemu Roweru Metropolitalnego MEVO (GBFS) | Projekt portfolio BI & Data Analytics.")
