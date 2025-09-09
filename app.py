# app.py
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from datetime import datetime

# =========================
# KONFIGURASI
# =========================
st.set_page_config(page_title="Scholar Rank", layout="wide")
st.title("Daftar Nama Penerima")

DB_PATH = "data/data.db"
os.makedirs("data", exist_ok=True)
engine = create_engine(f"sqlite:///{DB_PATH}", future=True)

# =========================
# INISIALISASI TABEL (tambah kolom nama_siswa)
# =========================
with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS alternatives (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            alternatif TEXT NOT NULL,
            nama_siswa TEXT DEFAULT '',
            c1_tanggungan REAL NOT NULL,
            c2_penghasilan REAL NOT NULL,
            c3_status REAL NOT NULL,
            c4_jarak REAL NOT NULL
        )
    """))
    # Jika tabel lama belum punya kolom nama_siswa, tambahkan (aman idempotent)
    try:
        conn.execute(text("ALTER TABLE alternatives ADD COLUMN nama_siswa TEXT DEFAULT ''"))
    except Exception:
        pass  # sudah ada

# =========================
# DEFAULT DATA DARI DATA.csv
# =========================
def build_default_df():
    """
    Default data diambil dari DATA.csv (delimiter ';') dan diubah
    ke schema yang dipakai app:
    Alternatif, Nama_Siswa, C1_Tanggungan, C2_Penghasilan, C3_Status, C4_Jarak
    """
    CSV_PATH = "DATA.csv"
    if not os.path.exists(CSV_PATH):
        st.error(f"File default '{CSV_PATH}' tidak ditemukan. Letakkan file di folder kerja.")
        return pd.DataFrame(columns=[
            "Alternatif", "Nama_Siswa",
            "C1_Tanggungan","C2_Penghasilan","C3_Status","C4_Jarak"
        ])

    df = pd.read_csv(CSV_PATH, sep=";").reset_index(drop=True)

    # Samakan nama kolom penghasilan (beberapa file punya 1 spasi)
    if "PENGHASILAN  ORANG TUA" not in df.columns and "PENGHASILAN ORANG TUA" in df.columns:
        df = df.rename(columns={"PENGHASILAN ORANG TUA": "PENGHASILAN  ORANG TUA"})

    required = [
        "ALTERNATIF",
        "NAMA SISWA",
        "JUMLAH TANGGUNGAN",
        "PENGHASILAN  ORANG TUA",
        "STATUS ANAK",
        "JARAK KE SEKOLAH (Km)",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Kolom wajib tidak ditemukan di DATA.csv: {missing}")
        return pd.DataFrame(columns=[
            "Alternatif", "Nama_Siswa",
            "C1_Tanggungan","C2_Penghasilan","C3_Status","C4_Jarak"
        ])

    # Alternatif: pakai yang ada; kalau kosong → A1..An
    alts_raw = df["ALTERNATIF"].astype(str).str.strip()
    fallback_alts = [f"A{i+1}" for i in range(len(df))]
    alts = np.where(alts_raw.eq("") | alts_raw.eq("nan"), fallback_alts, alts_raw)

    # Nama Siswa
    nama = df["NAMA SISWA"].astype(str).str.strip()

    # C1_Tanggungan
    c1 = pd.to_numeric(df["JUMLAH TANGGUNGAN"], errors="coerce").fillna(0).astype(float)

    # C2_Penghasilan: bersihkan Rp, titik, koma, spasi
    peng = df["PENGHASILAN  ORANG TUA"].astype(str)
    peng = (
        peng.str.replace("Rp.", "", regex=False)
            .str.replace("Rp", "", regex=False)
            .str.replace(" ", "", regex=False)
            .str.replace(".", "", regex=False)
            .str.replace(",", "", regex=False)
    )
    c2 = pd.to_numeric(peng, errors="coerce").fillna(0).astype(float)

    # C3_Status (biner 0/1: yatim/piatu/yatim piatu -> 1)
    stat = df["STATUS ANAK"].astype(str).str.lower()
    c3 = stat.apply(lambda s: 1.0 if ("yatim" in s or "piatu" in s) else 0.0).astype(float)

    # C4_Jarak
    c4 = pd.to_numeric(df["JARAK KE SEKOLAH (Km)"], errors="coerce").fillna(0.0).astype(float)

    out = pd.DataFrame({
        "Alternatif": alts,
        "Nama_Siswa": nama,
        "C1_Tanggungan": c1,
        "C2_Penghasilan": c2,
        "C3_Status": c3,
        "C4_Jarak": c4,
    })

    return out

# =========================
# FUNGSI DB
# =========================
@st.cache_data
def load_alt_from_db() -> pd.DataFrame:
    with engine.begin() as conn:
        df = pd.read_sql(
            """
            SELECT id, created_at, alternatif, nama_siswa,
            c1_tanggungan, c2_penghasilan, c3_status, c4_jarak
            FROM alternatives ORDER BY id DESC
            """,
            conn
        )
    return df

def insert_alt(kode_alt: str, nama_siswa: str, c1: float, c2: float, c3: float, c4: float):
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO alternatives
                (created_at, alternatif, nama_siswa, c1_tanggungan, c2_penghasilan, c3_status, c4_jarak)
                VALUES (:ts, :alt, :nm, :c1, :c2, :c3, :c4)
            """),
            dict(
                ts=datetime.now().isoformat(timespec="seconds"),
                alt=kode_alt, nm=nama_siswa, c1=c1, c2=c2, c3=c3, c4=c4
            )
        )
    load_alt_from_db.clear()  # refresh cache

def delete_alt(row_id: int):
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM alternatives WHERE id=:i"), dict(i=row_id))
    load_alt_from_db.clear()

# =========================
# TOPSIS (4 kriteria)
# =========================
def topsis(X: np.ndarray, weights: np.ndarray, criteria: list[str]):
    denom = np.sqrt((X ** 2).sum(axis=0))
    denom = np.where(denom == 0, 1e-12, denom)
    R = X / denom
    V = R * weights

    n = V.shape[1]
    A_plus = np.zeros(n)
    A_minus = np.zeros(n)
    for j in range(n):
        if criteria[j].lower() == "benefit":
            A_plus[j] = np.max(V[:, j])
            A_minus[j] = np.min(V[:, j])
        else:
            A_plus[j] = np.min(V[:, j])
            A_minus[j] = np.max(V[:, j])

    D_plus = np.sqrt(((V - A_plus) ** 2).sum(axis=1))
    D_minus = np.sqrt(((V - A_minus) ** 2).sum(axis=1))
    Vi = D_minus / (D_plus + D_minus + 1e-12)

    return {
        "R": R,
        "V": V,
        "A_plus": A_plus,
        "A_minus": A_minus,
        "D_plus": D_plus,
        "D_minus": D_minus,
        "V_score": Vi,
    }

# =========================
# SIDEBAR (tambahkan input ALTERNATIF & NAMA SISWA, DLL)
# =========================
with st.sidebar:
    st.header("🗃️ Tambah Data Siswa / Siswi")
    with st.form("form_tambah_db", clear_on_submit=True):
        kode_alt = st.text_input("Kode Alternatif", placeholder="contoh: A88")
        nama_baru = st.text_input("Nama Siswa", placeholder="contoh: AFDOL FAHMUL ROZI")
        c1 = st.number_input("C1 Tanggungan", min_value=0.0, step=1.0)
        c2 = st.number_input("C2 Penghasilan (Rp/bln)", min_value=0.0, step=1000.0)
        c3 = st.number_input("C3 Status (0/1)", min_value=0.0, step=1.0)
        c4 = st.number_input("C4 Jarak (km)", min_value=0.0, step=0.1, format="%.3f")
        submitted = st.form_submit_button("➕ Tambah Data")

    if submitted:
        if not kode_alt.strip():
            st.error("Kode Alternatif tidak boleh kosong.")
        elif not nama_baru.strip():
            st.error("Nama Siswa tidak boleh kosong.")
        else:
            # === PENGECEKAN KODE ALTERNATIF SUDAH ADA? ===
            # Cek di default data
            df_default = build_default_df()
            kode_default = df_default["Alternatif"].astype(str).str.upper().tolist()
            # Cek di database
            df_db_check = load_alt_from_db()
            kode_db = df_db_check["alternatif"].astype(str).str.upper().tolist()
            
            if kode_alt.strip().upper() in kode_default or kode_alt.strip().upper() in kode_db:
                st.error(f"Kode Alternatif '{kode_alt.strip()}' sudah ada. Gunakan kode lain.")
            else:
                insert_alt(kode_alt.strip(), nama_baru.strip(), float(c1), float(c2), float(c3), float(c4))
                st.success("✅ Data Berhasil Disimpan.")

    st.markdown("---")
    st.caption("💡 Data yang Baru Ditambahkan:")
    db_df = load_alt_from_db()
    if not db_df.empty:
        st.dataframe(db_df, use_container_width=True, hide_index=True)
        del_id = st.selectbox("Hapus entri (pilih ID)", options=[None] + db_df["id"].tolist(), index=0)
        if del_id is not None and st.button("🗑️ Hapus Entri Terpilih"):
            delete_alt(int(del_id))
            st.success(f"ID {del_id} terhapus.")

    st.markdown("---")
    st.header("⚙️ Pengaturan Bobot")
    DEFAULT_WEIGHTS = np.array([0.30, 0.35, 0.20, 0.15])
    w1 = st.number_input("Bobot C1 (Tanggungan)", value=float(DEFAULT_WEIGHTS[0]), min_value=0.0, format="%.3f")
    w2 = st.number_input("Bobot C2 (Penghasilan)", value=float(DEFAULT_WEIGHTS[1]), min_value=0.0, format="%.3f")
    w3 = st.number_input("Bobot C3 (Status)", value=float(DEFAULT_WEIGHTS[2]), min_value=0.0, format="%.3f")
    w4 = st.number_input("Bobot C4 (Jarak)", value=float(DEFAULT_WEIGHTS[3]), min_value=0.0, format="%.3f")
    weights = np.array([w1, w2, w3, w4], dtype=float)
    if weights.sum() == 0:
        st.warning("Total bobot = 0, pakai default.")
        weights = DEFAULT_WEIGHTS.copy()
    else:
        weights = weights / weights.sum()
    
    st.header("🛠️ Pengaturan Attribut")
    c1_attr = st.selectbox("Atribut C1 (Tanggungan)", ["Benefit", "Cost"], index=0)
    c2_attr = st.selectbox("Atribut C2 (Penghasilan)", ["Benefit", "Cost"], index=1)
    c3_attr = st.selectbox("Atribut C3 (Status)", ["Benefit", "Cost"], index=0)
    c4_attr = st.selectbox("Atribut C4 (Jarak)", ["Benefit", "Cost"], index=0)
    criteria = [c1_attr, c2_attr, c3_attr, c4_attr]

# =========================
# GABUNG DATA (DEFAULT dari CSV + DB)
# =========================
df_base = build_default_df()  # berisi kolom: Alternatif, Nama_Siswa, C1..C4

if not db_df.empty:
    df_db = db_df.rename(columns={
        "alternatif":"Alternatif",
        "nama_siswa":"Nama_Siswa",
        "c1_tanggungan":"C1_Tanggungan",
        "c2_penghasilan":"C2_Penghasilan",
        "c3_status":"C3_Status",
        "c4_jarak":"C4_Jarak",
    })[["Alternatif","Nama_Siswa","C1_Tanggungan","C2_Penghasilan","C3_Status","C4_Jarak"]]
    df_all = pd.concat([df_base, df_db], ignore_index=True)
else:
    df_all = df_base.copy()

# =========================
# TAMPILKAN DEFAULT DATA ALTERNATIF
# =========================
st.subheader("📄 Data Lengkap Calon Penerima")
cols_order = ["Alternatif", "Nama_Siswa", "C1_Tanggungan","C2_Penghasilan","C3_Status","C4_Jarak"]
exist_cols = [c for c in cols_order if c in df_all.columns]
st.dataframe(df_all[exist_cols], use_container_width=True, hide_index=True)

if df_all.empty:
    st.error("Data kosong.")
    st.stop()

# =========================
# HITUNG TOPSIS
# =========================
try:
    X = df_all[["C1_Tanggungan","C2_Penghasilan","C3_Status","C4_Jarak"]].astype(float).values
except Exception:
    st.error("Pastikan kolom C1..C4 berisi angka.")
    st.stop()

calc = topsis(X, weights, criteria)
results = df_all[["Alternatif"]].copy()
results["D_plus"] = calc["D_plus"]
results["D_minus"] = calc["D_minus"]
results["V"] = calc["V_score"]
results["Kategori"] = np.where(results["V"] >= 0.5, "Layak", "Kurang Layak")
results_sorted = results.sort_values(by="V", ascending=False).reset_index(drop=True)
results_sorted.index = results_sorted.index + 1

st.subheader("🏆 Hasil Perankingan (TOPSIS)")
st.dataframe(results_sorted, use_container_width=True)

st.markdown("### 📊 Skor Preferensi (V)")
st.bar_chart(results_sorted.set_index("Alternatif")["V"])

with st.expander("🔎 Detail Per Alternatif"):
    pilihan = st.selectbox("Pilih alternatif", options=results_sorted["Alternatif"].tolist())
    ridx = df_all.index[df_all["Alternatif"] == pilihan][0]
    detail = pd.DataFrame({
        "Kriteria": ["C1_Tanggungan","C2_Penghasilan","C3_Status","C4_Jarak"],
        "Atribut": criteria,
        "Bobot": weights,
        "Nilai": X[ridx,:]
    })
    colA, colB = st.columns([1.2,1])
    with colA:
        st.dataframe(detail, hide_index=True, use_container_width=True)
    with colB:
        st.metric("D⁺ (ke Ideal +)", f"{calc['D_plus'][ridx]:.6f}")
        st.metric("D⁻ (ke Ideal −)", f"{calc['D_minus'][ridx]:.6f}")
        st.metric("V (Preferensi)", f"{calc['V_score'][ridx]:.6f}")
        st.metric("Kategori", "Layak" if calc["V_score"][ridx] >= 0.5 else "Kurang Layak")

# Unduh hasil
csv_rank = results_sorted.reset_index().rename(columns={"index":"Peringkat"}).to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Unduh Hasil (CSV)", data=csv_rank, file_name="hasil_topsis.csv", mime="text/csv")
