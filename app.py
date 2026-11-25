import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="MEVO Trójmiasto – Radar Rowerzysty", page_icon="🚲", layout="wide")

# Automatyczne wykrycie aktualnych adresów plików GBFS z katalogu głównego MEVO
@st.cache_data(ttl=60)
def load_mevo_data():
    gbfs_index = requests.get("https://gbfs.urbansharing.com/rowermevo.pl/gbfs.json", timeout=10).json()
    feeds = {f["name"]: f["url"] for f in gbfs_index["data"]["pl"]["feeds"]}

    stations_info = pd.DataFrame(requests.get(feeds["station_information"], timeout=10).json()["data"]["stations"])
    stations_status = pd.DataFrame(requests.get(feeds["station_status"], timeout=10).json()["data"]["stations"])
    free_bikes = pd.DataFrame(requests.get(feeds["free_bike_status"], timeout=10).json()["data"]["bikes"])

    # Połączenie słownika stacji ze statusem
    stations = pd.merge(stations_info, stations_status, on="station_id", how="inner")
    stations["oblozenie_pct"] = (stations["num_bikes_available"] / stations["capacity"] * 100).round(1)

    # Czyszczenie danych o rowerach (odfiltrowanie anomalii GPS spoza Pomorza)
    bikes = free_bikes[
        (free_bikes["lat"].between(53.8, 55.0)) &
        (free_bikes["lon"].between(17.5, 19.5))
    ].copy()
    bikes["zasieg_km"] = (bikes["current_range_meters"] / 1000).round(1)

    return stations, bikes, datetime.now().strftime("%Y-%m-%d %H:%M:%S")

stations_df, bikes_df, updated_at = load_mevo_data()

# --- NAGŁÓWEK I STORYTELLING ---
st.title("🚲 MEVO Trójmiasto – Radar Rowerzysty")
st.markdown(
    "Projekt analityczny stworzony z pasji do jazdy na rowerze po Trójmieście. "
    f"Dane pobierane na żywo z otwartego API MEVO (GBFS) • **Ostatnia aktualizacja:** `{updated_at}`"
)

# --- FILTRY W PANELU BOCZNYM ---
st.sidebar.header("🎛️ Panel Rowerzysty")
min_range = st.sidebar.slider("Minimalny zasięg e-bike'a na trasę (km):", 0, 100, 30, step=5)
only_available = st.sidebar.checkbox("Tylko rowery gotowe do jazdy (sprawne)", value=True)

filtered_bikes = bikes_df.copy()
if only_available:
    filtered_bikes = filtered_bikes[(~filtered_bikes["is_disabled"]) & (~filtered_bikes["is_reserved"])]

ebikes_ready = filtered_bikes[(filtered_bikes["vehicle_type_id"] == "ebike") & (filtered_bikes["zasieg_km"] >= min_range)]

# --- KARTY KPI ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Dostępne rowery na stacjach", int(stations_df["num_bikes_available"].sum()))
col2.metric(f"E-bike'i z zasięgiem ≥ {min_range} km", len(ebikes_ready))
col3.metric("Średni zasięg sprawnych e-bike'ów", f"{filtered_bikes[filtered_bikes['vehicle_type_id']=='ebike']['zasieg_km'].mean():.1f} km")
col4.metric("Aktywne stacje MEVO", len(stations_df[stations_df["is_renting"] == True]))

st.divider()

# --- SEKCJA 1: MAPA STACJI I TOP 10 ---
left, right = st.columns([1.3, 1])

with left:
    st.subheader("🗺️ Mapa dostępności na stacjach")
    fig_map = px.scatter_map(
        stations_df,
        lat="lat",
        lon="lon",
        size="num_bikes_available",
        color="oblozenie_pct",
        hover_name="name",
        hover_data={"num_bikes_available": True, "capacity": True, "oblozenie_pct": True, "lat": False, "lon": False},
        color_continuous_scale="Tealgrn",
        zoom=10,
        height=480,
        labels={"num_bikes_available": "Dostępne rowery", "capacity": "Pojemność", "oblozenie_pct": "Obłożenie (%)"}
    )
    fig_map.update_layout(map_style="carto-positron", margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig_map, use_container_width=True)

with right:
    st.subheader("🏆 TOP 10 stacji z największą liczbą rowerów")
    top10 = stations_df.sort_values("num_bikes_available", ascending=False).head(10)
    fig_bar = px.bar(
        top10.iloc[::-1],
        x="num_bikes_available",
        y="name",
        orientation="h",
        text="num_bikes_available",
        color="oblozenie_pct",
        color_continuous_scale="Tealgrn",
        height=480,
        labels={"num_bikes_available": "Liczba rowerów", "name": "Stacja", "oblozenie_pct": "Obłożenie (%)"}
    )
    fig_bar.update_layout(showlegend=False, coloraxis_showscale=False, margin={"r": 10, "t": 10, "l": 0, "b": 0})
    st.plotly_chart(fig_bar, use_container_width=True)

# --- SEKCJA 2: PERSPEKTYWA ROWERZYSTY (BATERIA) ---
st.subheader("🔋 Rozkład zasięgu baterii w dostępnych e-bike'ach")
ebikes_all_avail = filtered_bikes[filtered_bikes["vehicle_type_id"] == "ebike"].dropna(subset=["zasieg_km"])
fig_hist = px.histogram(
    ebikes_all_avail,
    x="zasieg_km",
    nbins=20,
    color_discrete_sequence=["#00A896"],
    labels={"zasieg_km": "Szacowany zasięg baterii (km)", "count": "Liczba rowerów"}
)
fig_hist.add_vline(x=min_range, line_dash="dash", line_color="#E63946", annotation_text=f"Twój próg: {min_range} km")
fig_hist.update_layout(bargap=0.1, height=300, yaxis_title="Liczba e-bike'ów")
st.plotly_chart(fig_hist, use_container_width=True)

# --- STOPKA ---
st.caption("Źródło danych: Otwarte Dane Systemu Roweru Metropolitalnego MEVO (standard GBFS). Niezależny projekt portfolio.")
