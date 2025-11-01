
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go

import re
from html import unescape
from unidecode import unidecode
import emoji

from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer




URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#([A-Za-z0-9_]+)")

# stopwords de dominio que suelen quedar en Twitter
DOMAIN_STOP = {"amp", "rt", "via", "gt", "u", "im", "http", "https"}

def split_hashtag_token(tok: str) -> str:

    tok = tok.replace("_", " ")
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", tok)
    return " ".join(parts).lower()

def clean_text_for_wordcloud(text: str, keep_numbers=False):
    if not isinstance(text, str):
        return ""
    t = text
    t = unescape(t)                 
    t = t.lower()
    t = unidecode(t)                
    t = URL_RE.sub(" ", t)          
    t = MENTION_RE.sub(" ", t)     
  
    t = HASHTAG_RE.sub(lambda m: split_hashtag_token(m.group(1)), t)
  
    try:
        t = emoji.replace_emoji(t, replace="")
    except Exception:
    
        t = re.sub(r"[^\x00-\x7F]+", " ", t)

    
    t = re.sub(r"[^a-z\s]", " ", t)

    toks = [tok for tok in t.split() if tok not in DOMAIN_STOP]

   
    toks = [tok for tok in toks if tok.isalpha() and len(tok) > 2]

    return " ".join(toks)


try:
    from wordcloud import WordCloud
    WC_AVAILABLE = True
except Exception:
    WC_AVAILABLE = False


from wordcloud import WordCloud, STOPWORDS

def generate_wordcloud_from_series(series, outpath=None, max_words=200):
    text = " ".join(series.dropna().astype(str).values)
    stopw = set(STOPWORDS)
    stopw = stopw.union({"amp","rt","via","gt","u","im"})
    wc = WordCloud(width=1000, height=600, background_color="white",
                   stopwords=stopw, max_words=max_words, regexp=r"\w{3,}").generate(text)
    if outpath:
        wc.to_file(outpath)
    return wc

# -------- paths --------
BASE = Path(".")
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(exist_ok=True)
MODEL_DIR = BASE / "models"
DATA_PATH = BASE / "train.csv"
METRICS_PATH = MODEL_DIR / "metrics.csv"

st.set_page_config("Voces en crisis — Dashboard", layout="wide")

# -------- helpers --------
@st.cache_data
def load_data():
    import re, string
    df = pd.read_csv(DATA_PATH)
    # minimal cleaning assumptions from run_all.py
    df["text"] = df["text"].astype(str)

    if "text_clean" not in df.columns:
        # quitar URLs, menciones y hashtags
        df["text_clean"] = df["text"].str.lower().str.replace(r"https?://\S+|www\.\S+", " ", regex=True)
        df["text_clean"] = df["text_clean"].str.replace(r"@\w+", " ", regex=True)
        df["text_clean"] = df["text_clean"].str.replace("#", " ", regex=True)

        # construir patrón escapado a partir de string.punctuation
        punct_escaped = re.escape(string.punctuation)
        pattern = f"[{punct_escaped}]"   # por ejemplo: "[\\!\\\"\\#\\$...]" (ya escapado)
        df["text_clean"] = df["text_clean"].str.replace(pattern, " ", regex=True)

        df["text_clean"] = df["text_clean"].str.replace(r"\d+", " ", regex=True)
        df["text_clean"] = df["text_clean"].str.replace(r"\s+", " ", regex=True).str.strip()
        df["text_clean_wc"] = df["text"].astype(str).apply(clean_text_for_wordcloud)

    df["text_char_len"] = df["text"].astype(str).map(len)
    return df


@st.cache_data
def load_models():
    models = {}
    for f in MODEL_DIR.glob("*.joblib"):
        models[f.stem] = joblib.load(f)
    return models

@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        return pd.read_csv(METRICS_PATH).set_index("model")
    return pd.DataFrame()

def top_tokens_from_series(series, topn=25):
    c = Counter()
    for t in series.dropna().astype(str):
        c.update(t.split())
    return c.most_common(topn)

def top_ngrams_from_series(series, n=2, topn=20, min_df=2):
    # simple sliding-window bigrams/trigrams from cleaned text
    c = Counter()
    for s in series.dropna().astype(str):
        toks = s.split()
        if len(toks) < n: continue
        for i in range(len(toks)-n+1):
            gram = " ".join(toks[i:i+n])
            c[gram] += 1
    # filter min_df
    items = [(g,f) for g,f in c.items() if f>=min_df]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:topn]

@st.cache_data
def compute_distinctive_words(df_filtered, topn=25):
    """Calcula palabras distintivas usando TF-IDF"""
    if len(df_filtered) == 0:
        return None, None
    
    texts_class0 = df_filtered.loc[df_filtered["target"] == 0, "text_clean"].astype(str).tolist()
    texts_class1 = df_filtered.loc[df_filtered["target"] == 1, "text_clean"].astype(str).tolist()
    
    if len(texts_class0) == 0 or len(texts_class1) == 0:
        return None, None
    
    try:
        vectorizer = TfidfVectorizer(max_features=1000, min_df=2, ngram_range=(1, 1))
        tfidf_matrix = vectorizer.fit_transform(texts_class0 + texts_class1)
        feature_names = vectorizer.get_feature_names_out()
        
        # Promediar TF-IDF scores para cada clase
        tfidf_class0 = tfidf_matrix[:len(texts_class0)].mean(axis=0).A1
        tfidf_class1 = tfidf_matrix[len(texts_class0):].mean(axis=0).A1
        
        # Crear DataFrame con scores
        df_tfidf = pd.DataFrame({
            'token': feature_names,
            'tfidf_class0': tfidf_class0,
            'tfidf_class1': tfidf_class1
        })
        
        # Calcular diferencia (qué tan distintiva es cada palabra)
        df_tfidf['diff_class1'] = df_tfidf['tfidf_class1'] - df_tfidf['tfidf_class0']
        df_tfidf['diff_class0'] = df_tfidf['tfidf_class0'] - df_tfidf['tfidf_class1']
        
        # Top palabras más distintivas de cada clase
        top_distinct_class1 = df_tfidf.nlargest(topn, 'diff_class1')[['token', 'diff_class1']]
        top_distinct_class0 = df_tfidf.nlargest(topn, 'diff_class0')[['token', 'diff_class0']]
        
        return top_distinct_class1, top_distinct_class0
    except Exception as e:
        return None, None

# -------- load ----------
df = load_data()
models = load_models()
metrics_df = load_metrics()

# -------- sidebar (controls) --------
st.sidebar.header("Filtros")
sel_class = st.sidebar.selectbox("Clase", options=["All", "0 - No desastre", "1 - Desastre"])
min_len = int(df["text_char_len"].min())
max_len = int(df["text_char_len"].max())
sel_len = st.sidebar.slider("Rango longitud (caracteres)", min_value=min_len, max_value=max_len,
                            value=(min_len, min(280, max_len)))
topn = st.sidebar.slider("Top N tokens / n-grams", min_value=5, max_value=40, value=20)
sel_ngram = st.sidebar.selectbox("N-gram", options=["1 (unigram)", "2 (bigram)"], index=0)

# Nuevos filtros: keyword y location
if df["keyword"].notna().sum() > 0:
    keywords_list = ["All"] + sorted(df["keyword"].dropna().unique().tolist())
    sel_keyword = st.sidebar.selectbox("Keyword", options=keywords_list[:100] if len(keywords_list) > 100 else keywords_list)
else:
    sel_keyword = "All"

if df["location"].notna().sum() > 0:
    locations_list = ["All"] + sorted(df["location"].dropna().unique().tolist())
    sel_location = st.sidebar.selectbox("Location", options=locations_list[:100] if len(locations_list) > 100 else locations_list)
else:
    sel_location = "All"

sel_models = st.sidebar.multiselect("Modelos a mostrar (matrices/métricas)", options=list(models.keys()),
                                    default=list(models.keys()))

st.sidebar.markdown("---")
st.sidebar.markdown("Descargas")
# Esta función se ejecutará cuando el usuario haga clic en el botón
def get_filtered_csv():
    dff_export = df.copy()
    if sel_class != "All":
        dff_export = dff_export[dff_export["target"] == int(sel_class.split()[0])]
    dff_export = dff_export[(dff_export["text_char_len"] >= sel_len[0]) & (dff_export["text_char_len"] <= sel_len[1])]
    if sel_keyword != "All":
        dff_export = dff_export[dff_export["keyword"] == sel_keyword]
    if sel_location != "All":
        dff_export = dff_export[dff_export["location"] == sel_location]
    return dff_export.to_csv(index=False).encode("utf-8")

st.sidebar.download_button("Descargar CSV filtrado", get_filtered_csv(),
                           file_name="filtered_tweets.csv", mime="text/csv")

# -------- layout header --------
st.title("Voces en crisis — Dashboard")
st.markdown("Explora el dataset de tweets etiquetados. Usa los filtros en la barra lateral para enlazar las visualizaciones.")

# Apply filters for main views
dff = df.copy()
if sel_class != "All":
    dff = dff[dff["target"] == int(sel_class.split()[0])]
dff = dff[(dff["text_char_len"] >= sel_len[0]) & (dff["text_char_len"] <= sel_len[1])]
if sel_keyword != "All":
    dff = dff[dff["keyword"] == sel_keyword]
if sel_location != "All":
    dff = dff[dff["location"] == sel_location]

# Estadísticas resumen del dataset filtrado
st.markdown("### Estadísticas del Dataset Filtrado")
col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
with col_stat1:
    st.metric("Total tweets", f"{len(dff):,}")
with col_stat2:
    if len(dff) > 0:
        pct_cls1 = (dff["target"] == 1).sum() / len(dff) * 100
        st.metric("Desastre (1)", f"{(dff['target']==1).sum():,} ({pct_cls1:.1f}%)")
    else:
        st.metric("Desastre (1)", "0")
with col_stat3:
    if len(dff) > 0:
        avg_len = dff["text_char_len"].mean()
        st.metric("Longitud promedio", f"{avg_len:.0f} chars")
    else:
        st.metric("Longitud promedio", "0")
with col_stat4:
    if len(dff) > 0 and dff["keyword"].notna().sum() > 0:
        pct_kw = dff["keyword"].notna().sum() / len(dff) * 100
        st.metric("Con keyword", f"{dff['keyword'].notna().sum():,} ({pct_kw:.1f}%)")
    else:
        st.metric("Con keyword", "0")

# Row 1: Class distribution + interactive hist
col1, col2 = st.columns([1,2])
with col1:
    st.subheader("Distribución de clases")
    img_p = FIG_DIR / "01_class_distribution.png"
    if img_p.exists():
        st.image(str(img_p), use_container_width=True)
    else:
        vc = df["target"].value_counts().sort_index()
        fig = px.bar(x=vc.index.astype(str), y=vc.values, labels={"x":"target","y":"count"},
                     color=vc.index.astype(str), color_discrete_sequence=["#3182bd","#e6550d"][:len(vc)])
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Longitud de tweets")
    fig = px.histogram(df, x="text_char_len", color="target", nbins=35,
                       color_discrete_map={0:"#3182bd",1:"#e6550d"},
                       labels={"text_char_len":"Longitud (caracteres)", "count":"Frecuencia"})
    fig.update_layout(barmode="overlay", bargap=0.1)
    fig.update_traces(opacity=0.6)
    st.plotly_chart(fig, use_container_width=True)

# Row 2: Top tokens / ngrams (interactive by class)
st.subheader("Tokens y n-gramas más frecuentes")
col3, col4 = st.columns([1,1])
with col3:
    st.markdown("**Top tokens / n-gramas (clase seleccionada)**")
    if sel_ngram.startswith("1"):
        pairs = top_tokens_from_series(dff["text_clean"], topn=topn)
    else:
        n = int(sel_ngram.split()[0])
        pairs = top_ngrams_from_series(dff["text_clean"], n=n, topn=topn)
    if pairs:
        toks = [p for p,_ in pairs][::-1]
        freqs = [f for _,f in pairs][::-1]
        fig = go.Figure(go.Bar(x=freqs, y=toks, orientation="h",
                               marker_color="#e6550d" if sel_class!="All" and sel_class.startswith("1") else "#3182bd"))
        fig.update_layout(height= max(300, len(toks)*20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay tokens con la configuración actual.")

with col4:
    st.markdown("**Wordcloud**")
    if WC_AVAILABLE:
        
        wc_series = dff["text_clean_wc"].dropna().astype(str)
        text_wc = " ".join(wc_series.values).strip()

        if text_wc:
           
            wc = WordCloud(width=800, height=400, background_color="white",
                           stopwords=set(STOPWORDS).union({"amp","rt","via","gt","u","im"}),
                           max_words=200, regexp=r"\w{3,}").generate(text_wc)
            st.image(wc.to_array(), use_container_width=True)

            
            try:
                
                if sel_class.startswith("1"):
                    wc.to_file(str(FIG_DIR / "07_wordcloud_desastre.png"))
                elif sel_class.startswith("0"):
                    wc.to_file(str(FIG_DIR / "08_wordcloud_nodesastre.png"))
                else:
                    
                    wc.to_file(str(FIG_DIR / "wordcloud_all.png"))
            except Exception:
                
                pass
        else:
            st.info("No hay texto limpio suficiente para generar wordcloud. Revisar filtros.")
    else:
        st.info("WordCloud no instalado. Instálalo con `pip install wordcloud` para ver esta visualización.")


# Row 3: Palabras Distintivas (TF-IDF)
st.subheader("Palabras Distintivas (Análisis TF-IDF)")
st.markdown("**Palabras más características de cada clase usando TF-IDF**")

distinct_1, distinct_0 = compute_distinctive_words(dff, topn=topn)

if distinct_1 is not None and distinct_0 is not None:
    tfidf_col1, tfidf_col2 = st.columns(2)
    
    with tfidf_col1:
        st.markdown("**Top palabras distintivas — Desastre (1)**")
        distinct_1_reversed = distinct_1.sort_values('diff_class1', ascending=True)
        fig_tfidf1 = go.Figure(go.Bar(
            x=distinct_1_reversed['diff_class1'].values,
            y=distinct_1_reversed['token'].values,
            orientation='h',
            marker_color='#e6550d'
        ))
        fig_tfidf1.update_layout(
            height=max(400, len(distinct_1)*20),
            xaxis_title="Score TF-IDF Distintivo",
            yaxis_title="",
            showlegend=False
        )
        st.plotly_chart(fig_tfidf1, use_container_width=True)
    
    with tfidf_col2:
        st.markdown("**Top palabras distintivas — No Desastre (0)**")
        distinct_0_reversed = distinct_0.sort_values('diff_class0', ascending=True)
        fig_tfidf0 = go.Figure(go.Bar(
            x=distinct_0_reversed['diff_class0'].values,
            y=distinct_0_reversed['token'].values,
            orientation='h',
            marker_color='#3182bd'
        ))
        fig_tfidf0.update_layout(
            height=max(400, len(distinct_0)*20),
            xaxis_title="Score TF-IDF Distintivo",
            yaxis_title="",
            showlegend=False
        )
        st.plotly_chart(fig_tfidf0, use_container_width=True)
else:
    st.info("No hay suficientes datos con los filtros actuales para calcular palabras distintivas. Necesitas datos de ambas clases.")

# Row 4: Bigrams / Trigrams area (always computed from filtered dff)
st.subheader("N-gramas")
bi = top_ngrams_from_series(dff["text_clean"], n=2, topn=20)
tri = top_ngrams_from_series(dff["text_clean"], n=3, topn=15)
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Top bigramas**")
    if bi:
        df_bi = pd.DataFrame(bi, columns=["bigram","freq"])
        fig = px.bar(df_bi, x="freq", y="bigram", orientation="h", labels={"freq":"Frecuencia"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay bigramas suficientes.")
with c2:
    st.markdown("**Top trigramas**")
    if tri:
        df_tri = pd.DataFrame(tri, columns=["trigram","freq"])
        fig = px.bar(df_tri, x="freq", y="trigram", orientation="h", labels={"freq":"Frecuencia"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay trigramas suficientes.")
        

# Row 5: Análisis de Keywords y Locations
if df["keyword"].notna().sum() > 0 or df["location"].notna().sum() > 0:
    st.subheader("Análisis de Keywords y Locations")
    kw_col1, kw_col2 = st.columns(2)
    
    with kw_col1:
        st.markdown("**Top Keywords por Clase (filtrado)**")
        if len(dff) > 0:
            # Top keywords para clase 1
            kw1 = dff.loc[dff["target"]==1, "keyword"].value_counts().head(10)
            # Top keywords para clase 0
            kw0 = dff.loc[dff["target"]==0, "keyword"].value_counts().head(10)
            
            if len(kw1) > 0 or len(kw0) > 0:
                kw_fig_data = []
                if len(kw1) > 0:
                    for kw, freq in kw1.items():
                        kw_fig_data.append({"keyword": kw, "freq": freq, "clase": "1 - Desastre"})
                if len(kw0) > 0:
                    for kw, freq in kw0.items():
                        kw_fig_data.append({"keyword": kw, "freq": freq, "clase": "0 - No desastre"})
                
                if kw_fig_data:
                    kw_df = pd.DataFrame(kw_fig_data)
                    fig_kw = px.bar(kw_df, x="freq", y="keyword", color="clase", orientation="h",
                                   color_discrete_map={"1 - Desastre": "#e6550d", "0 - No desastre": "#3182bd"})
                    fig_kw.update_layout(height=max(400, len(kw_df)*25))
                    st.plotly_chart(fig_kw, use_container_width=True)
            else:
                st.info("No hay keywords disponibles con los filtros actuales.")
        else:
            st.info("No hay datos con los filtros actuales.")
    
    with kw_col2:
        st.markdown("**Top Locations por Clase (filtrado)**")
        if len(dff) > 0:
            # Top locations para clase 1
            loc1 = dff.loc[dff["target"]==1, "location"].value_counts().head(10)
            # Top locations para clase 0
            loc0 = dff.loc[dff["target"]==0, "location"].value_counts().head(10)
            
            if len(loc1) > 0 or len(loc0) > 0:
                loc_fig_data = []
                if len(loc1) > 0:
                    for loc, freq in loc1.items():
                        loc_fig_data.append({"location": loc, "freq": freq, "clase": "1 - Desastre"})
                if len(loc0) > 0:
                    for loc, freq in loc0.items():
                        loc_fig_data.append({"location": loc, "freq": freq, "clase": "0 - No desastre"})
                
                if loc_fig_data:
                    loc_df = pd.DataFrame(loc_fig_data)
                    fig_loc = px.bar(loc_df, x="freq", y="location", color="clase", orientation="h",
                                      color_discrete_map={"1 - Desastre": "#e6550d", "0 - No desastre": "#3182bd"})
                    fig_loc.update_layout(height=max(400, len(loc_df)*25))
                    st.plotly_chart(fig_loc, use_container_width=True)
            else:
                st.info("No hay locations disponibles con los filtros actuales.")
        else:
            st.info("No hay datos con los filtros actuales.")

# Row 6: Ejemplos (tabla) - enlazada con token selection (multi)
st.subheader("Ejemplos de tweets")
tokens_sel = st.multiselect("Filtrar por tokens", options=list(pd.Series(" ".join(dff["text_clean"].astype(str)).split()).value_counts().index[:200]))
# aplicar token filter
display_df = dff.copy()
if tokens_sel:
    display_df = display_df[display_df["text_clean"].apply(lambda s: all(t in s.split() for t in tokens_sel))]
if len(display_df) > 0:
    st.dataframe(display_df[["id","keyword","location","text","target"]].sample(min(200, len(display_df))), use_container_width=True)
else:
    st.info("No hay tweets que coincidan con los filtros actuales.")

# Row 7: Model comparison (metrics + confusion matrices)
st.subheader("Comparación de modelos y matrices de confusión")
if not metrics_df.empty:
    st.table(metrics_df.loc[sorted(metrics_df.index)].round(3))
else:
    st.info("No se encontró metrics.csv en models/ (ejecuta run_all.py primero).")



# show confusion matrix images for selected models
cols = st.columns(len(sel_models) if sel_models else 1)
for i, m in enumerate(sel_models):
    with cols[i]:
        st.markdown(f"**{m}**")
        cm_path = FIG_DIR / f"cm_{m}.png"
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)
        else:
            st.info("Matriz no disponible. Ejecuta run_all.py para generarla.")

# Row 8: prediccion rapida con modelo seleccionado
st.subheader("Probar modelo con texto libre")
model_choice = st.selectbox("Elegir modelo para predecir", options=list(models.keys()) if models else [])
text_in = st.text_area("Escribe un tweet de ejemplo aquí", value="Massive fire reported near the highway, people evacuating")
if st.button("Predecir"):
    if model_choice and model_choice in models:
        model = models[model_choice]
        # aplicar misma limpieza rápida que run_all
        txt = text_in.lower()
        txt = txt.replace("#"," ")
        pred = model.predict([txt])[0]
        proba = model.predict_proba([txt])[0,1] if hasattr(model, "predict_proba") else None
        st.write("Predicción (1 = desastre):", int(pred))
        if proba is not None:
            st.write(f"Probabilidad clase 1: {proba:.3f}")
    else:
        st.error("No hay modelos cargados.")



st.markdown("---")

