"""
Dashboard Analisis Sentimen Topik Kesehatan (TF-IDF + Neural Network MLP)
Data    : tweet_data_kesehatan.xlsx (sheet 'tweet_data') atau tweet_data.csv
Jalankan: streamlit run app_sentimen_nn.py
"""
import glob
import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

try:
    from wordcloud import WordCloud
    ADA_WC = True
except ImportError:  # app tetap jalan; tampilkan grafik kata teratas
    ADA_WC = False

st.set_page_config(page_title="Sentimen Kesehatan", page_icon="🩺", layout="wide")

# ----------------------------------------------------------------------------
# Konfigurasi (sama dengan notebook)
# ----------------------------------------------------------------------------
DATA_FILES = ["tweet_data_kesehatan.xlsx", "tweet_data.csv"]
LABELS = ["positif", "netral", "negatif"]
WARNA = {"positif": "#2e7d32", "netral": "#757575", "negatif": "#c62828"}
CMAP = {"positif": "Greens", "netral": "Blues", "negatif": "Reds"}
EMOJI = {"positif": "😊", "netral": "😐", "negatif": "😠"}

KATA_POSITIF = [
    "aman", "lega", "selamat", "bantuan", "membantu", "pulih", "sehat", "membaik",
    "sembuh", "berhasil", "sigap", "ramah", "cepat", "bagus", "bersih", "gratis",
    "terjangkau", "syukur", "alhamdulillah", "terima", "kasih", "apresiasi", "peduli",
    "semangat", "salut", "praktis", "memudahkan", "senang", "pahlawan", "tersedia",
    "sabar", "jelas",
]
KATA_NEGATIF = [
    "parah", "kecewa", "langka", "sengsara", "resah", "takut", "darurat", "buruk",
    "judes", "terlantar", "ditolak", "ribet", "mengecewakan", "kritis", "bahaya",
    "kotor", "tercemar", "mahal", "rugi", "sedih", "error", "gagal", "kesal", "tutup",
    "marah", "hoaks", "korban", "sakit", "penuh", "sesak", "menunda", "kekurangan",
    "meningkat", "naik",
]
STOPWORDS = {
    "yang", "di", "dan", "ini", "itu", "ada", "untuk", "dengan", "dari", "pada",
    "adalah", "ke", "oleh", "dapat", "mulai", "rt", "yg", "saya", "kita", "juga",
    "sudah", "lagi", "jadi", "karena", "atau", "akan", "bisa", "sih", "nya",
    # kata umum yang muncul di semua kelas dan tidak membedakan sentimen
    "kesehatan", "banget", "warga", "anak", "hari", "jam", "tentang", "lewat",
}


# ----------------------------------------------------------------------------
# Fungsi bantu
# ----------------------------------------------------------------------------
def bersihkan_teks(teks):
    """Langkah 3: cleansing."""
    teks = str(teks)
    teks = re.sub(r"https?://\S+|www\.\S+", " ", teks)   # URL
    teks = re.sub(r"@\w+", " ", teks)                     # mention
    teks = re.sub(r"#\w+", " ", teks)                     # hashtag
    teks = re.sub(r"\d+", " ", teks)                      # angka
    teks = re.sub(r"[^a-zA-Z\s]", " ", teks)              # simbol, tanda baca, emoji
    teks = teks.lower()                                   # case folding
    teks = re.sub(r"\s+", " ", teks).strip()              # spasi berlebih
    return teks


def hitung(teks, daftar):
    # kata utuh, agar "sehat" tidak ikut terhitung pada "kesehatan"
    return sum(len(re.findall(r"\b" + re.escape(k) + r"\b", teks)) for k in daftar)


def beri_label(teks_bersih):
    """Langkah 4: pelabelan rule-based."""
    pos = hitung(teks_bersih, KATA_POSITIF)
    neg = hitung(teks_bersih, KATA_NEGATIF)
    if pos > neg:
        return "positif"
    if neg > pos:
        return "negatif"
    return "netral"


def baca_data(src):
    nama = getattr(src, "name", str(src)).lower()
    if nama.endswith(".xlsx"):
        try:
            return pd.read_excel(src, sheet_name="tweet_data")
        except ValueError:
            if hasattr(src, "seek"):
                src.seek(0)
            return pd.read_excel(src)
    return pd.read_csv(src)


def cari_file():
    for f in DATA_FILES:
        if os.path.exists(f):
            return f
    for f in DATA_FILES:
        hasil = glob.glob("**/" + f, recursive=True)
        if hasil:
            return hasil[0]
    return None


@st.cache_data(show_spinner=False)
def siapkan_data(df):
    df = df.dropna(subset=["full_text"]).drop_duplicates(subset="full_text").copy()
    df["clean_text"] = df["full_text"].apply(bersihkan_teks)
    df = df[df["clean_text"] != ""].reset_index(drop=True)
    df["sentimen"] = df["clean_text"].apply(beri_label)
    return df


@st.cache_resource(show_spinner="Melatih model Neural Network...")
def latih_model(teks, label, hidden, max_iter):
    X = pd.Series(teks)
    y = pd.Series(label)
    strat = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat
    )
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    Xtr = tfidf.fit_transform(X_train)
    Xte = tfidf.transform(X_test)
    mlp = MLPClassifier(
        hidden_layer_sizes=(hidden,), activation="relu", solver="adam",
        max_iter=max_iter, random_state=42,
    )
    mlp.fit(Xtr, y_train)
    y_pred = mlp.predict(Xte)
    return tfidf, mlp, y_test.to_numpy(), y_pred, len(X_train), len(X_test)


def gambar_wordcloud(teks, lab):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    if ADA_WC:
        wc = WordCloud(width=600, height=400, background_color="white",
                       colormap=CMAP[lab], stopwords=STOPWORDS,
                       random_state=1).generate(teks)
        ax.imshow(wc, interpolation="bilinear")
    else:
        kata = [w for w in teks.split() if w not in STOPWORDS]
        top = Counter(kata).most_common(10)
        ax.axis("on")
        ax.barh([k for k, _ in top][::-1], [v for _, v in top][::-1], color=WARNA[lab])
        ax.set_title("10 kata teratas (wordcloud belum terpasang)")
    return fig


# ----------------------------------------------------------------------------
# Sidebar & data
# ----------------------------------------------------------------------------
st.title("🩺 Analisis Sentimen Topik Kesehatan")
st.caption("Data → Cleansing → Pelabelan Rule-Based → TF-IDF → MLP → Evaluasi → Prediksi")

st.sidebar.header("⚙️ Pengaturan")
upload = st.sidebar.file_uploader("Unggah data lain (CSV/XLSX, opsional)", type=["csv", "xlsx"])
hidden = st.sidebar.slider("Neuron hidden layer", 10, 300, 100, 10)
max_iter = st.sidebar.slider("Max iterasi", 100, 1000, 300, 50)
st.sidebar.write("Rasio train:test = **80:20**")

if upload is not None:
    raw = baca_data(upload)
    st.sidebar.success(f"Memakai: {upload.name}")
else:
    path = cari_file()
    if path is None:
        st.error("File data tidak ditemukan. Taruh `tweet_data_kesehatan.xlsx` di folder yang sama "
                 f"dengan app ini ({os.getcwd()}) atau unggah lewat sidebar.")
        st.stop()
    raw = baca_data(path)
    st.sidebar.info(f"Memakai: {path}")

wajib = ["created_at", "username", "full_text"]
if not set(wajib).issubset(raw.columns):
    st.error(f"Kolom wajib: {wajib}. Kolom yang ada: {list(raw.columns)}")
    st.stop()

df = siapkan_data(raw[wajib])

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📄 Data", "📊 Distribusi", "☁️ WordCloud", "🧠 Model & Evaluasi", "🔮 Prediksi"]
)

# ----------------------------------------------------------------------------
# Tab 1: Data
# ----------------------------------------------------------------------------
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah tweet", len(df))
    c2.metric("Pengguna unik", df["username"].nunique())
    c3.metric("Rata-rata kata (bersih)", f"{df['clean_text'].str.split().str.len().mean():.1f}")
    if len(df) < 300:
        st.warning("Data kurang dari 300 tweet (syarat minimal praktikum).")
    st.dataframe(df[["created_at", "username", "full_text", "clean_text", "sentimen"]], height=420)
    with st.expander("Kamus kata untuk pelabelan"):
        k1, k2 = st.columns(2)
        k1.write("**Positif**: " + ", ".join(KATA_POSITIF))
        k2.write("**Negatif**: " + ", ".join(KATA_NEGATIF))
    st.download_button("⬇️ Unduh dataset berlabel",
                       df.to_csv(index=False).encode("utf-8"), "tweet_berlabel.csv", "text/csv")

# ----------------------------------------------------------------------------
# Tab 2: Distribusi
# ----------------------------------------------------------------------------
with tab2:
    dist = df["sentimen"].value_counts().reindex(LABELS).fillna(0).astype(int)
    st.dataframe(dist.rename("jumlah").to_frame().T)
    colA, colB = st.columns(2)
    with colA:
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(dist.index, dist.values, color=[WARNA[l] for l in dist.index])
        ax.bar_label(bars)
        ax.set_title("Distribusi Sentimen (Bar Chart)")
        ax.set_ylabel("Jumlah tweet")
        st.pyplot(fig)
    with colB:
        fig, ax = plt.subplots(figsize=(5, 4))
        nz = dist[dist > 0]
        ax.pie(nz.values, labels=nz.index, autopct="%1.1f%%", startangle=90,
               colors=[WARNA[l] for l in nz.index])
        ax.set_title("Proporsi Sentimen (Pie Chart)")
        st.pyplot(fig)

# ----------------------------------------------------------------------------
# Tab 3: WordCloud
# ----------------------------------------------------------------------------
with tab3:
    if not ADA_WC:
        st.info("Library `wordcloud` belum terpasang (`pip install wordcloud`), ditampilkan grafik kata teratas.")
    cols = st.columns(3)
    for col, lab in zip(cols, LABELS):
        with col:
            st.subheader(f"{EMOJI[lab]} {lab.capitalize()}")
            teks_kelas = " ".join(df.loc[df["sentimen"] == lab, "clean_text"])
            if teks_kelas.strip():
                st.pyplot(gambar_wordcloud(teks_kelas, lab))
            else:
                st.info("Tidak ada data untuk kelas ini.")

# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
if df["sentimen"].nunique() < 2:
    st.error("Label hanya satu kelas, model tidak bisa dilatih. Perluas daftar kata atau tambah data.")
    st.stop()

tfidf, mlp, y_test, y_pred, n_train, n_test = latih_model(
    tuple(df["clean_text"]), tuple(df["sentimen"]), hidden, max_iter
)

# ----------------------------------------------------------------------------
# Tab 4: Evaluasi
# ----------------------------------------------------------------------------
with tab4:
    kelas = [l for l in LABELS if l in set(y_test)]
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, sup = precision_recall_fscore_support(y_test, y_pred, labels=kelas, zero_division=0)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)

    st.write(f"Data latih: **{n_train}** | Data uji: **{n_test}** | Iterasi: **{mlp.n_iter_}**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Akurasi", f"{acc:.4f}")
    m2.metric("Presisi (macro)", f"{p_m:.4f}")
    m3.metric("Recall (macro)", f"{r_m:.4f}")
    m4.metric("F1-Score (macro)", f"{f_m:.4f}")

    tabel = pd.DataFrame({"Presisi": prec, "Recall": rec, "F1-Score": f1, "Support": sup}, index=kelas)
    st.markdown("**Metrik per kelas**")
    st.dataframe(tabel.round(4))
    with st.expander("Classification report lengkap"):
        st.code(classification_report(y_test, y_pred, labels=kelas, zero_division=0))

    colA, colB = st.columns(2)
    with colA:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        x = np.arange(len(kelas)); w = 0.25
        ax.bar(x - w, prec, w, label="Presisi")
        ax.bar(x, rec, w, label="Recall")
        ax.bar(x + w, f1, w, label="F1-Score")
        ax.set_xticks(x)
        ax.set_xticklabels(kelas)
        ax.set_ylim(0, 1.1)
        ax.set_title("Metrik per Kelas")
        ax.legend()
        st.pyplot(fig)
    with colB:
        cm = confusion_matrix(y_test, y_pred, labels=kelas)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(kelas)))
        ax.set_yticks(range(len(kelas)))
        ax.set_xticklabels(kelas)
        ax.set_yticklabels(kelas)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        ax.set_title("Confusion Matrix")
        for i in range(len(kelas)):
            for j in range(len(kelas)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im, ax=ax)
        st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(mlp.loss_curve_)
    ax.set_title("Kurva Loss Pelatihan")
    ax.set_xlabel("Iterasi")
    ax.set_ylabel("Loss")
    st.pyplot(fig)

    st.caption("Catatan: label dibuat dari aturan kata (rule-based), jadi skor tinggi berarti model "
               "berhasil meniru aturan tersebut, belum tentu memahami sentimen sebenarnya.")

# ----------------------------------------------------------------------------
# Tab 5: Prediksi
# ----------------------------------------------------------------------------
with tab5:
    contoh = "Pelayanan puskesmas hari ini cepat dan ramah, pasien merasa lega"
    teks = st.text_area("Masukkan teks:", value=contoh, height=120)
    if st.button("Prediksi", type="primary"):
        bersih = bersihkan_teks(teks)
        if not bersih:
            st.warning("Teks kosong setelah dibersihkan.")
        else:
            vec = tfidf.transform([bersih])
            hasil = mlp.predict(vec)[0]
            proba = mlp.predict_proba(vec)[0]
            st.markdown(f"### {EMOJI[hasil]} Sentimen: **{hasil.upper()}**")
            st.write(f"Setelah cleansing: `{bersih}`")
            tp = pd.DataFrame({"Sentimen": mlp.classes_, "Probabilitas": proba})
            tp = tp.sort_values("Probabilitas", ascending=False).reset_index(drop=True)
            tp["Label"] = tp["Sentimen"].map(lambda s: f"{EMOJI[s]} {s}")
            tp["Persen"] = tp["Probabilitas"].map(lambda p: f"{p:.2%}")
            st.dataframe(tp[["Label", "Persen"]], hide_index=True)
            st.bar_chart(tp.set_index("Sentimen")["Probabilitas"])