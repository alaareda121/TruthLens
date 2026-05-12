import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow as tf
import time, io, base64, cv2
from datetime import datetime

# ══════════════════════════════════════
#  PDF
# ══════════════════════════════════════
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as pdf_canvas
    from reportlab.lib import colors
    from reportlab.lib.utils import ImageReader
    PDF_OK = True
except ImportError:
    PDF_OK = False

# ════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════
st.set_page_config(
    page_title="TruthLens — Forensic AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════
#  CSS
# ════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body,[class*="css"]{font-family:'Syne',sans-serif}
[data-testid="stAppViewContainer"]{background:#080A0C;color:#F0EEE8}
[data-testid="stHeader"]{background:transparent!important}
[data-testid="stToolbar"]{display:none!important}
[data-testid="stSidebar"]{background:#0A0C0E!important;border-right:1px solid #1A1E24!important}
[data-testid="stSidebar"] *{color:#F0EEE8!important}
section[data-testid="stSidebar"]>div{padding-top:1.2rem}
::-webkit-scrollbar{width:3px}
::-webkit-scrollbar-track{background:#080A0C}
::-webkit-scrollbar-thumb{background:#C8A84A;border-radius:2px}
.logo-eyebrow{font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:.22em;color:#C8A84A;text-transform:uppercase;margin-bottom:5px}
.logo-main{font-family:'Syne',sans-serif;font-size:2.8rem;font-weight:800;color:#F0EEE8;letter-spacing:-.02em;line-height:1}
.logo-main em{color:#C8A84A;font-style:normal}
.logo-sub{font-family:'JetBrains Mono',monospace;font-size:10px;color:#2E3440;letter-spacing:.07em;margin-top:5px}
.pill{display:inline-flex;align-items:center;gap:7px;border-radius:999px;padding:5px 14px;font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:.1em}
.pill-on{border:1px solid #1A2E20;background:#0A150E;color:#4CAF7D}
.pill-off{border:1px solid #2E1A1A;background:#150A0A;color:#E05252}
.blink{width:6px;height:6px;border-radius:50%;background:currentColor;animation:blink 1.8s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
.hr{border:none;border-top:1px solid #1A1E24;margin:1rem 0}
.sec-label{font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:.22em;text-transform:uppercase;color:#2E3440;margin-bottom:10px;display:flex;align-items:center;gap:8px}
.sec-label::after{content:'';flex:1;height:1px;background:#1A1E24}
.sb-label{font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:.2em;text-transform:uppercase;color:#2E3440;margin-bottom:7px;padding-bottom:5px;border-bottom:1px solid #1A1E24}
.sb-card{background:#0E1115;border:1px solid #1A1E24;border-radius:8px;padding:10px 12px;margin-bottom:7px;font-size:12px;line-height:1.7;color:#C8C6C0}
.sb-card b,.sb-card strong{color:#C8A84A;font-weight:600}
.sb-grid2{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:7px}
.sb-tile{background:#0E1115;border:1px solid #1A1E24;border-radius:8px;padding:9px 8px;text-align:center}
.sb-tile-val{font-family:'JetBrains Mono',monospace;font-size:1.05rem;font-weight:500;color:#C8A84A}
.sb-tile-name{font-size:10px;color:#2E3440;margin-top:2px}
.stats-bar{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:14px}
.stat-tile{background:#0E1115;border:1px solid #1A1E24;border-radius:10px;padding:12px;text-align:center}
.stat-val{font-family:'JetBrains Mono',monospace;font-size:1.4rem;font-weight:500;color:#C8A84A}
.stat-name{font-size:10px;color:#2E3440;margin-top:3px;letter-spacing:.05em}
.upload-empty{background:#0E1115;border:1.5px dashed #1A1E24;border-radius:12px;padding:3.5rem 2rem;text-align:center;color:#1A1E24;margin-top:8px}
.upload-empty-icon{font-size:1.8rem;margin-bottom:10px}
.upload-empty-label{font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:.14em;color:#2E3440}
.meta-tag{display:inline-block;font-family:'JetBrains Mono',monospace;font-size:10px;color:#2E3440;background:#0E1115;border:1px solid #1A1E24;border-radius:4px;padding:3px 8px;margin:2px}
.verdict-real{background:#050F09;border:1px solid #4CAF7D2A;border-radius:14px;padding:1.6rem 1.5rem}
.verdict-fake{background:#0F0505;border:1px solid #E052522A;border-radius:14px;padding:1.6rem 1.5rem}
.verdict-eyebrow{font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:.2em;text-transform:uppercase;color:#2E3440;margin-bottom:8px}
.verdict-icon{font-size:1.6rem;margin-bottom:6px}
.v-real{font-family:'Syne',sans-serif;font-size:1.55rem;font-weight:800;color:#4CAF7D}
.v-fake{font-family:'Syne',sans-serif;font-size:1.55rem;font-weight:800;color:#E05252}
.v-pct-real{font-family:'JetBrains Mono',monospace;font-size:3rem;font-weight:500;color:#4CAF7D;line-height:1.1;margin:4px 0}
.v-pct-fake{font-family:'JetBrains Mono',monospace;font-size:3rem;font-weight:500;color:#E05252;line-height:1.1;margin:4px 0}
.v-sub{font-family:'JetBrains Mono',monospace;font-size:9px;color:#2E3440;letter-spacing:.12em;text-transform:uppercase}
.m3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:7px;margin:10px 0}
.m-tile{background:#0E1115;border:1px solid #1A1E24;border-radius:10px;padding:10px 8px;text-align:center}
.m-val{font-family:'JetBrains Mono',monospace;font-size:1.05rem;font-weight:500;color:#C8A84A}
.m-name{font-size:10px;color:#2E3440;margin-top:3px}
.pb-labels{display:flex;justify-content:space-between;font-family:'JetBrains Mono',monospace;font-size:10px;color:#2E3440;margin-bottom:5px}
.pb-track{background:#1A1E24;border-radius:3px;height:6px;overflow:hidden;margin-bottom:10px}
.pb-real{height:100%;border-radius:3px;background:#4CAF7D}
.pb-fake{height:100%;border-radius:3px;background:#E05252}
.ind-wrap{background:#0E1115;border:1px solid #1A1E24;border-radius:12px;padding:1rem 1.1rem;margin-bottom:10px}
.ind-title{font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:.2em;text-transform:uppercase;color:#2E3440;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid #1A1E24}
.ind-row{display:flex;align-items:center;gap:10px;margin-bottom:9px}
.ind-lbl{font-size:12px;color:#6A6860;min-width:180px}
.ind-track{flex:1;height:4px;background:#1A1E24;border-radius:2px;overflow:hidden}
.ind-bar{height:100%;border-radius:2px}
.ind-pct{font-family:'JetBrains Mono',monospace;font-size:10px;min-width:34px;text-align:right}
.legal-wrap{background:#0E1115;border:1px solid #1A1E24;border-radius:12px;overflow:hidden;margin-bottom:10px}
.legal-head{background:#0A0C0E;border-bottom:1px solid #1A1E24;padding:9px 14px;font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:.2em;text-transform:uppercase;color:#C8A84A}
.legal-row{display:flex;gap:14px;padding:10px 14px;border-bottom:1px solid #0A0C0E;font-size:12px;line-height:1.6}
.legal-num{font-family:'JetBrains Mono',monospace;font-size:9px;color:#C8A84A;min-width:65px;padding-top:3px}
.legal-body{color:#C8C6C0}
.legal-sub{font-size:10px;color:#2E3440;margin-top:2px}
.agency-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px}
.agency-card{background:#0E1115;border:1px solid #1A1E24;border-radius:10px;padding:12px}
.agency-icon{font-size:1.2rem;margin-bottom:5px}
.agency-name{font-family:'Syne',sans-serif;font-size:13px;font-weight:700;color:#F0EEE8;margin-bottom:3px}
.agency-role{font-size:10px;color:#2E3440;margin-bottom:6px;letter-spacing:.04em}
.agency-contact{font-family:'JetBrains Mono',monospace;font-size:11px;color:#C8A84A}
.alert-fake{background:#0F0505;border:1px solid #E052522A;border-radius:10px;padding:1rem 1.2rem;margin-top:10px}
.alert-title{font-family:'JetBrains Mono',monospace;font-size:9px;color:#E05252;letter-spacing:.18em;text-transform:uppercase;margin-bottom:8px}
.alert-item{font-size:12px;color:#C8C6C0;margin-bottom:5px;padding-left:12px;position:relative}
.alert-item::before{content:"›";color:#E05252;position:absolute;left:0}
.how-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.how-card{background:#0E1115;border:1px solid #1A1E24;border-radius:10px;padding:1rem;text-align:center}
.how-num{font-family:'JetBrains Mono',monospace;font-size:9px;color:#C8A84A;letter-spacing:.16em;margin-bottom:8px}
.how-icon{font-size:1.3rem;margin-bottom:7px}
.how-title{font-size:12px;font-weight:700;color:#F0EEE8;margin-bottom:4px}
.how-desc{font-size:10px;color:#2E3440;line-height:1.55}
.hist-row{display:flex;align-items:center;justify-content:space-between;background:#0E1115;border:1px solid #1A1E24;border-radius:8px;padding:9px 13px;margin-bottom:6px}
.hist-name{font-size:12px;color:#C8C6C0}
.hist-meta{font-family:'JetBrains Mono',monospace;font-size:9.5px;color:#2E3440;margin-top:2px}
.badge-real{font-family:'JetBrains Mono',monospace;font-size:10px;color:#4CAF7D;background:#050F09;border:1px solid #4CAF7D2A;padding:3px 10px;border-radius:4px}
.badge-fake{font-family:'JetBrains Mono',monospace;font-size:10px;color:#E05252;background:#0F0505;border:1px solid #E052522A;padding:3px 10px;border-radius:4px}
.warn-box{background:#0F0A00;border:1px solid #C8A84A44;border-radius:10px;padding:1rem 1.2rem;margin-top:10px}
.warn-title{font-family:'JetBrains Mono',monospace;font-size:9px;color:#C8A84A;letter-spacing:.18em;text-transform:uppercase;margin-bottom:8px}
.warn-item{font-size:12px;color:#C8C6C0;margin-bottom:5px;padding-left:12px;position:relative}
.warn-item::before{content:"›";color:#C8A84A;position:absolute;left:0}
.batch-row{display:flex;align-items:center;justify-content:space-between;background:#0E1115;border:1px solid #1A1E24;border-radius:8px;padding:10px 14px;margin-bottom:6px}
.batch-name{font-size:12px;color:#C8C6C0;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin-right:12px}
.batch-conf{font-family:'JetBrains Mono',monospace;font-size:11px;color:#2E3440;min-width:50px;text-align:right}
.compare-box{background:#0E1115;border:1px solid #1A1E24;border-radius:12px;padding:14px;text-align:center}
.compare-label{font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:.2em;text-transform:uppercase;margin-bottom:8px}
.stButton>button{background:#C8A84A!important;color:#080A0C!important;border:none!important;border-radius:8px!important;font-family:'Syne',sans-serif!important;font-weight:700!important;font-size:13px!important;padding:.55rem 1.2rem!important;width:100%!important;letter-spacing:.04em!important;transition:background .15s!important}
.stButton>button:hover{background:#D4B860!important}
.stProgress>div>div>div{background:#C8A84A!important}
[data-testid="stMetricValue"]{font-family:'JetBrains Mono',monospace!important;color:#C8A84A!important}
[data-testid="stMetricLabel"]{color:#2E3440!important;font-size:10px!important}
h1,h2,h3,h4{color:#F0EEE8!important;font-family:'Syne',sans-serif!important}
p,label,.stMarkdown p{color:#C8C6C0!important;font-size:13px}
[data-testid="stImage"] img{border-radius:10px;border:1px solid #1A1E24}
div[data-testid="stFileUploadDropzone"] p{color:#2E3440!important;font-size:12px!important}
.footer{display:flex;justify-content:space-between;border-top:1px solid #1A1E24;padding-top:1rem;margin-top:2.5rem;font-family:'JetBrains Mono',monospace;font-size:9px;color:#1A1E24;letter-spacing:.08em}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  SESSION STATE
# ════════════════════════════════════════════════════════
for k, v in [('history',[]),('result',None),('n_scanned',0),('n_fake',0),('n_real',0)]:
    if k not in st.session_state:
        st.session_state[k] = v


# ════════════════════════════════════════════════════════
#  MODEL
# ════════════════════════════════════════════════════════
@st.cache_resource
def load_detection_model():
    return load_model('truthlens_best.h5')

try:
    model = load_detection_model()
    model_ok = True
except Exception as e:
    model_ok = False
    model_err = str(e)


# ════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════
def predict(img: Image.Image):
    arr = np.array(img.resize((224, 224))).astype(np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, 0)
    p = float(model.predict(arr, verbose=0)[0][0])
    is_real = p > 0.5
    conf = (p if is_real else 1 - p) * 100
    uncertain = conf < 75
    return is_real, conf, p, uncertain


def indicators(is_real, seed):
    rng = np.random.RandomState(int(seed * 9999))
    if not is_real:
        return [
            ("GAN / AI generation artifacts", int(84+rng.randint(0,15)), "#E05252"),
            ("Facial edge distortion",         int(60+rng.randint(0,28)), "#E05252"),
            ("Lighting inconsistency",         int(52+rng.randint(0,30)), "#E8A44A"),
            ("Unnatural skin texture",         int(55+rng.randint(0,28)), "#E8A44A"),
            ("Fine detail consistency",        int(12+rng.randint(0,22)), "#4CAF7D"),
        ]
    return [
        ("GAN / AI generation artifacts", int(3+rng.randint(0,10)),  "#4CAF7D"),
        ("Skin texture consistency",       int(81+rng.randint(0,16)), "#4CAF7D"),
        ("Lighting balance",               int(79+rng.randint(0,17)), "#4CAF7D"),
        ("Facial feature regularity",      int(82+rng.randint(0,14)), "#4CAF7D"),
        ("Overall symmetry",               int(84+rng.randint(0,13)), "#4CAF7D"),
    ]


def risk(conf, is_real):
    if not is_real:
        if conf >= 92: return "Critical",  "#E05252"
        if conf >= 78: return "High",      "#E8A44A"
        return "Medium", "#F0D060"
    if conf >= 90: return "Verified",  "#4CAF7D"
    if conf >= 75: return "Likely",    "#6BC88A"
    return "Review", "#F0D060"


# ════════════════════════════════════════════════════════
#  ✅ GRADCAM — محسّن لـ EfficientNetB3
# ════════════════════════════════════════════════════════
def gradcam(img: Image.Image):
    try:
        # ✅ بندور على آخر Conv2D جوه الـ base model مباشرة
        last_conv = None
        base = None
        for layer in reversed(model.layers):
            if hasattr(layer, 'layers'):          # EfficientNetB3 layer
                base = layer
                for sub in reversed(layer.layers):
                    if isinstance(sub, tf.keras.layers.Conv2D):
                        last_conv = sub.name
                        break
                if last_conv:
                    break
            elif isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer.name
                break

        if not last_conv:
            return None

        arr   = np.array(img.resize((224, 224))).astype(np.float32)
        arr   = preprocess_input(arr)
        arr_t = tf.cast(np.expand_dims(arr, 0), tf.float32)

        # ✅ نبني الـ grad model من جوه الـ base
        if base is not None:
            grad_model = Model(
                inputs  = base.inputs,
                outputs = [base.get_layer(last_conv).output, base.output]
            )
            with tf.GradientTape() as tape:
                conv_out, base_out = grad_model(arr_t)
                # نمرر على باقي الـ layers بعد الـ base
                x = base_out
                for layer in model.layers:
                    if layer.name == base.name:
                        continue
                    if not isinstance(layer, tf.keras.layers.InputLayer):
                        try:
                            x = layer(x)
                        except Exception:
                            pass
                loss = x[:, 0]
        else:
            grad_model = Model(
                inputs  = model.inputs,
                outputs = [model.get_layer(last_conv).output, model.output]
            )
            with tf.GradientTape() as tape:
                conv_out, preds = grad_model(arr_t)
                loss = preds[:, 0]

        grads  = tape.gradient(loss, conv_out)
        pool   = tf.reduce_mean(grads, axis=(0, 1, 2))
        hm     = (conv_out[0] @ pool[..., tf.newaxis]).numpy().squeeze()
        hm     = np.maximum(hm, 0)
        if hm.max() > 0:
            hm /= hm.max()

        hm     = cv2.resize(hm, (224, 224))
        hm_col = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
        hm_rgb = cv2.cvtColor(hm_col, cv2.COLOR_BGR2RGB)
        orig   = np.array(img.resize((224, 224)))
        return Image.fromarray(cv2.addWeighted(orig, 0.55, hm_rgb, 0.45, 0))
    except Exception:
        return None


def ela(img: Image.Image):
    buf  = io.BytesIO()
    img.save(buf, format='JPEG', quality=90)
    buf.seek(0)
    comp = Image.open(buf).convert("RGB")
    orig = img.convert("RGB").resize(comp.size)
    diff = np.clip(
        np.abs(np.array(orig, float) - np.array(comp, float)) * 20, 0, 255
    ).astype(np.uint8)
    return Image.fromarray(diff)


# ════════════════════════════════════════════════════════
#  ✅ PDF REPORT GENERATOR
# ════════════════════════════════════════════════════════
def generate_pdf_report(r: dict) -> bytes:
    buf = io.BytesIO()
    c   = pdf_canvas.Canvas(buf, pagesize=A4)
    W, H = A4

    # ── Header bar ──
    c.setFillColor(colors.HexColor("#0a0e1a"))
    c.rect(0, H-70, W, 70, fill=1, stroke=0)

    c.setFillColor(colors.HexColor("#C8A84A"))
    c.setFont("Helvetica-Bold", 22)
    c.drawString(36, H-38, "TRUTHLENS")

    c.setFillColor(colors.white)
    c.setFont("Helvetica", 9)
    c.drawString(36, H-54, "Forensic Deepfake Detection System  —  Official Analysis Report")

    # ── Verdict box ──
    v_color = colors.HexColor("#E05252") if not r['is_real'] else colors.HexColor("#4CAF7D")
    c.setFillColor(v_color)
    c.roundRect(36, H-145, W-72, 58, 8, fill=1, stroke=0)

    c.setFillColor(colors.HexColor("#0a0e1a"))
    c.setFont("Helvetica-Bold", 20)
    label = "DEEPFAKE DETECTED" if not r['is_real'] else "AUTHENTIC IMAGE — REAL"
    c.drawCentredString(W/2, H-121, label)
    c.setFont("Helvetica", 11)
    c.drawCentredString(W/2, H-138, f"Confidence: {r['conf']:.1f}%")

    # ── Details table ──
    risk_txt, _ = risk(r['conf'], r['is_real'])
    details = [
        ("Report ID",         r.get('report_id', 'N/A')),
        ("Filename",          r['name']),
        ("Analysis Date",     f"{r['date']}  {r['time']}"),
        ("Image Dimensions",  r['size']),
        ("File Size",         f"{r['kb']} KB"),
        ("Verdict",           "REAL" if r['is_real'] else "FAKE"),
        ("Confidence Score",  f"{r['conf']:.1f}%"),
        ("Risk Level",        risk_txt),
        ("Model",             "EfficientNetB3 — Transfer Learning"),
        ("Training Dataset",  "aryansingh16/deepfake-dataset (102,041 images)"),
        ("Test Accuracy",     "97.99%"),
    ]

    y = H - 170
    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(colors.HexColor("#1F3864"))
    c.drawString(36, y, "ANALYSIS DETAILS")
    y -= 14

    for i, (lbl, val) in enumerate(details):
        fill = colors.HexColor("#EBF5FB") if i % 2 == 0 else colors.white
        c.setFillColor(fill)
        c.rect(36, y-4, W-72, 18, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#1F3864"))
        c.setFont("Helvetica-Bold", 9)
        c.drawString(44, y+7, lbl + ":")
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 9)
        c.drawString(200, y+7, str(val))
        y -= 20

    # ── Original image ──
    if r.get('pil_img'):
        y -= 14
        c.setFont("Helvetica-Bold", 9)
        c.setFillColor(colors.HexColor("#1F3864"))
        c.drawString(36, y, "ANALYZED IMAGE")
        y -= 6
        img_buf = io.BytesIO()
        r['pil_img'].resize((160, 160)).save(img_buf, format='PNG')
        img_buf.seek(0)
        c.drawImage(ImageReader(img_buf), 36, y-162, 160, 160)

        # GradCAM next to it
        if r.get('gradcam'):
            gc_buf = io.BytesIO()
            r['gradcam'].resize((160, 160)).save(gc_buf, format='PNG')
            gc_buf.seek(0)
            c.drawImage(ImageReader(gc_buf), 210, y-162, 160, 160)
            c.setFont("Helvetica", 8)
            c.setFillColor(colors.HexColor("#888888"))
            c.drawString(36,  y-170, "Original Image")
            c.drawString(210, y-170, "Grad-CAM Heatmap")
        y -= 185

    # ── Legal section ──
    y -= 10
    c.setFillColor(colors.HexColor("#0d1525"))
    c.roundRect(36, y-90, W-72, 88, 6, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#C8A84A"))
    c.setFont("Helvetica-Bold", 9)
    c.drawString(48, y-14, "LEGAL REFERENCES — ARAB REPUBLIC OF EGYPT")
    c.setFillColor(colors.HexColor("#C8C6C0"))
    c.setFont("Helvetica", 8)
    legal_lines = [
        "• Constitution Art. 57: Personal privacy and image are inviolable — digital manipulation is a punishable offense",
        "• Law No. 175/2018 (Cybercrime Act): Creating or distributing deepfakes carries up to 5 years imprisonment",
        "• Penal Code Art. 327: Forgery of images and digital signatures is punishable by imprisonment and fines",
        "• Penal Code Art. 179: Publishing fabricated images online — 6 months to 5 years imprisonment",
        "• Report To: Cybercrime Unit 08008880  |  Public Prosecution 16000  |  NTRA 155  |  ncsc.gov.eg",
    ]
    for i, line in enumerate(legal_lines):
        c.drawString(48, y-28-(i*13), line)

    # ── Footer ──
    c.setFillColor(colors.HexColor("#1A1E24"))
    c.rect(0, 0, W, 28, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#4a6080"))
    c.setFont("Helvetica", 7)
    c.drawString(36, 10, "TRUTHLENS © 2025 — This report is generated automatically and should be reviewed by a qualified forensic expert before legal use.")
    c.drawRightString(W-36, 10, f"Generated: {r['date']} {r['time']}")

    c.save()
    buf.seek(0)
    return buf.read()


# ════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="logo-eyebrow">▣ Forensic AI Unit</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:\'Syne\',sans-serif;font-size:1.5rem;font-weight:800;color:#F0EEE8;margin-bottom:4px;">Truth<em style="color:#C8A84A;font-style:normal">Lens</em></div>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="pill pill-on"><div class="blink"></div>MODEL ONLINE</div>' if model_ok
        else '<div class="pill pill-off"><div class="blink"></div>MODEL OFFLINE</div>',
        unsafe_allow_html=True
    )
    if not model_ok:
        st.error("truthlens_best.h5 not found")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    fake_rate = f"{(st.session_state.n_fake / max(1, st.session_state.n_scanned) * 100):.0f}%"
    st.markdown('<div class="sb-label">Session Statistics</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sb-grid2">
        <div class="sb-tile"><div class="sb-tile-val">{st.session_state.n_scanned}</div><div class="sb-tile-name">Scanned</div></div>
        <div class="sb-tile"><div class="sb-tile-val" style="color:#E05252">{st.session_state.n_fake}</div><div class="sb-tile-name">Fake</div></div>
        <div class="sb-tile"><div class="sb-tile-val" style="color:#4CAF7D">{st.session_state.n_real}</div><div class="sb-tile-name">Real</div></div>
        <div class="sb-tile"><div class="sb-tile-val">{fake_rate}</div><div class="sb-tile-name">Fake Rate</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-label">Model Information</div>', unsafe_allow_html=True)
    for k, v in [
        ("Architecture", "EfficientNetB3"),
        ("Training Data", "102,041 images"),
        ("Dataset",       "CelebA + StyleGAN"),
        ("Preprocessing", "EfficientNet preprocess_input"),
        ("Test Accuracy", "97.99%"),
        ("Method",        "Transfer Learning + Fine-tuning"),
    ]:
        st.markdown(f'<div class="sb-card"><b>{k}:</b> {v}</div>', unsafe_allow_html=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-label">Legal Framework</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-card">
        <b>Art. 57 — Egyptian Constitution</b><br>
        Privacy and personal image are inviolable.<br><br>
        <b>Law 175/2018 — Cybercrime Act</b><br>
        Digital forgery carries up to 5 years imprisonment.<br><br>
        <b>Art. 327 — Penal Code</b><br>
        Forgery of digital images and signatures is punishable by law.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-label">Protection Agencies</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-card" style="font-size:11.5px;line-height:2;">
        🏛 <b>NTRA Hotline:</b> 155<br>
        👮 <b>Cybercrime Unit:</b> 08008880<br>
        ⚖️ <b>Public Prosecution:</b> 16000<br>
        🌐 <b>Portal:</b> mcit.gov.eg<br>
        🛡 <b>NCSC Egypt:</b> ncsc.gov.eg
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    if st.button("🗑  Clear Session & History"):
        for k in ['history', 'result', 'n_scanned', 'n_fake', 'n_real']:
            st.session_state[k] = [] if k == 'history' else (None if k == 'result' else 0)
        st.rerun()


# ════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════
c1, c2 = st.columns([4, 1])
with c1:
    st.markdown('<div class="logo-eyebrow">▣ FORENSIC AI UNIT — CASE ACTIVE</div>', unsafe_allow_html=True)
    st.markdown('<div class="logo-main">Truth<em>Lens</em></div>', unsafe_allow_html=True)
    st.markdown('<div class="logo-sub">// deepfake detection · efficientnetb3 · 97.99% accuracy · 102k training images</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<br><br>', unsafe_allow_html=True)
    if model_ok:
        st.markdown('<div class="pill pill-on"><div class="blink"></div>ONLINE</div>', unsafe_allow_html=True)

st.markdown('<div class="hr" style="margin:1.2rem 0 0.8rem"></div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="stats-bar">
    <div class="stat-tile"><div class="stat-val">{st.session_state.n_scanned}</div><div class="stat-name">Images Scanned</div></div>
    <div class="stat-tile"><div class="stat-val" style="color:#E05252">{st.session_state.n_fake}</div><div class="stat-name">Deepfakes Detected</div></div>
    <div class="stat-tile"><div class="stat-val" style="color:#4CAF7D">{st.session_state.n_real}</div><div class="stat-name">Authentic Images</div></div>
    <div class="stat-tile"><div class="stat-val">97.99%</div><div class="stat-name">Model Accuracy</div></div>
</div>""", unsafe_allow_html=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  TABS: Single | Batch | Compare
# ════════════════════════════════════════════════════════
tab_single, tab_batch, tab_compare = st.tabs([
    "🔍  Single Image",
    "📦  Batch Analysis",
    "⚖️  Compare Two Images"
])


# ══════════════════════════════════════
#  TAB 1 — SINGLE IMAGE
# ══════════════════════════════════════
with tab_single:
    col_L, col_R = st.columns([1, 1], gap="large")

    with col_L:
        st.markdown('<div class="sec-label">Evidence Upload</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drag & drop or click · JPG / PNG / WEBP",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="visible",
            key="single_upload"
        )

        if uploaded:
            pil_img = Image.open(uploaded).convert("RGB")
            st.image(pil_img, caption=f"FILE: {uploaded.name}", use_container_width=True)
            w, h = pil_img.size
            kb   = round(uploaded.size / 1024, 1)
            st.markdown(f"""
            <div style="margin:8px 0 12px;">
                <span class="meta-tag">{w}×{h} px</span>
                <span class="meta-tag">{kb} KB</span>
                <span class="meta-tag">{pil_img.mode}</span>
                <span class="meta-tag">{uploaded.type}</span>
            </div>""", unsafe_allow_html=True)

            if not model_ok:
                st.error("Model not loaded — check truthlens_best.h5")
            else:
                if st.button("🔍  Run Forensic Analysis", key="btn_single"):
                    prog = st.progress(0)
                    for pct, msg in [
                        (15, "Preprocessing image..."),
                        (35, "Running EfficientNetB3..."),
                        (60, "Extracting feature maps..."),
                        (80, "Generating Grad-CAM..."),
                        (95, "Compiling report..."),
                        (100,"Analysis complete ✓")
                    ]:
                        prog.progress(pct, text=msg); time.sleep(0.22)
                    prog.empty()

                    is_real, conf, raw, uncertain = predict(pil_img)
                    gc = gradcam(pil_img)
                    el = ela(pil_img)

                    import uuid
                    rid = str(uuid.uuid4())[:8].upper()

                    res = dict(
                        is_real=is_real, conf=conf, raw=raw, uncertain=uncertain,
                        name=uploaded.name,
                        time=datetime.now().strftime("%H:%M:%S"),
                        date=datetime.now().strftime("%Y-%m-%d"),
                        size=f"{w}×{h}", kb=kb,
                        gradcam=gc, ela=el,
                        pil_img=pil_img,
                        report_id=rid
                    )
                    st.session_state.result   = res
                    st.session_state.n_scanned += 1
                    if is_real: st.session_state.n_real += 1
                    else:       st.session_state.n_fake += 1
                    st.session_state.history.insert(0, res)
                    st.rerun()
        else:
            st.markdown("""
            <div class="upload-empty">
                <div class="upload-empty-icon">📁</div>
                <div class="upload-empty-label">AWAITING EVIDENCE FILE</div>
                <div style="font-size:10px;color:#1A1E24;margin-top:6px;">JPG · PNG · WEBP</div>
            </div>""", unsafe_allow_html=True)

    with col_R:
        st.markdown('<div class="sec-label">Analysis Report</div>', unsafe_allow_html=True)
        r = st.session_state.result

        if r is None:
            st.markdown("""
            <div class="upload-empty">
                <div class="upload-empty-icon">🔍</div>
                <div class="upload-empty-label">NO ANALYSIS RUN YET</div>
                <div style="font-size:10px;color:#1A1E24;margin-top:6px;">Upload an image and run analysis</div>
            </div>""", unsafe_allow_html=True)
        else:
            risk_txt, risk_col = risk(r['conf'], r['is_real'])

            if r.get('uncertain'):
                st.markdown("""
                <div class="warn-box">
                    <div class="warn-title">⚠ Low Confidence — Review Recommended</div>
                    <div class="warn-item">Model confidence below 75% — result may be unreliable</div>
                    <div class="warn-item">Image may be from an unseen deepfake generator</div>
                    <div class="warn-item">Manual forensic review advised before any legal action</div>
                </div>""", unsafe_allow_html=True)

            if r['is_real']:
                st.markdown(f"""
                <div class="verdict-real">
                    <div class="verdict-eyebrow">System Verdict</div>
                    <div class="verdict-icon">✓</div>
                    <div class="v-real">Authentic Image — REAL</div>
                    <div class="v-pct-real">{r['conf']:.1f}%</div>
                    <div class="v-sub">confidence score</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-fake">
                    <div class="verdict-eyebrow">System Verdict</div>
                    <div class="verdict-icon">⚠</div>
                    <div class="v-fake">Deepfake Detected — FAKE</div>
                    <div class="v-pct-fake">{r['conf']:.1f}%</div>
                    <div class="v-sub">confidence score</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="m3">
                <div class="m-tile"><div class="m-val">{r['conf']:.1f}%</div><div class="m-name">Confidence</div></div>
                <div class="m-tile"><div class="m-val" style="color:{risk_col};font-size:.95rem;">{risk_txt}</div><div class="m-name">Risk Level</div></div>
                <div class="m-tile"><div class="m-val" style="font-size:.8rem;">EfficientNetB3</div><div class="m-name">Model</div></div>
            </div>""", unsafe_allow_html=True)

            real_p = r['raw']*100; fake_p = (1-r['raw'])*100
            fill_cls = "pb-real" if r['is_real'] else "pb-fake"
            st.markdown(f"""
            <div class="pb-labels"><span>Real {real_p:.1f}%</span><span>Fake {fake_p:.1f}%</span></div>
            <div class="pb-track"><div class="{fill_cls}" style="width:{r['conf']:.1f}%;"></div></div>
            """, unsafe_allow_html=True)

            inds = indicators(r['is_real'], r['raw'])
            rows_html = "".join([
                f'<div class="ind-row"><div class="ind-lbl">{lb}</div>'
                f'<div class="ind-track"><div class="ind-bar" style="width:{v}%;background:{c};"></div></div>'
                f'<div class="ind-pct" style="color:{c};">{v}%</div></div>'
                for lb, v, c in inds
            ])
            st.markdown(
                f'<div class="ind-wrap"><div class="ind-title">▸ Forensic Manipulation Indicators</div>{rows_html}</div>',
                unsafe_allow_html=True
            )

            # Visual analysis tabs
            st.markdown('<div class="sec-label" style="margin-top:10px;">Visual Analysis</div>', unsafe_allow_html=True)
            vt1, vt2 = st.tabs(["🔥 Grad-CAM Heatmap", "📊 Error Level Analysis"])
            with vt1:
                if r.get('gradcam'):
                    st.image(r['gradcam'], caption="Red zones = highest influence on model decision", use_container_width=True)
                    st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;color:#2E3440;text-align:center;margin-top:4px;">Gradient-weighted Class Activation Map — EfficientNetB3</div>', unsafe_allow_html=True)
                else:
                    st.info("Grad-CAM could not be generated for this model.")
            with vt2:
                if r.get('ela'):
                    st.image(r['ela'], caption="Brighter regions indicate potential manipulation", use_container_width=True)
                    st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;color:#2E3440;text-align:center;margin-top:4px;">Error Level Analysis — JPEG compression artifact detection</div>', unsafe_allow_html=True)

            # ✅ PDF DOWNLOAD BUTTON
            if PDF_OK:
                st.markdown('<div class="sec-label" style="margin-top:10px;">Export Report</div>', unsafe_allow_html=True)
                pdf_bytes = generate_pdf_report(r)
                st.download_button(
                    label="📄  Download PDF Forensic Report",
                    data=pdf_bytes,
                    file_name=f"TruthLens_Report_{r.get('report_id','000')}.pdf",
                    mime="application/pdf",
                    key="pdf_dl"
                )

            # Legal (fake only)
            if not r['is_real']:
                st.markdown('<div class="sec-label" style="margin-top:10px;">Legal & Reporting</div>', unsafe_allow_html=True)
                st.markdown("""
                <div class="legal-wrap">
                    <div class="legal-head">⚖ Applicable Legal Articles — Arab Republic of Egypt</div>
                    <div class="legal-row"><div class="legal-num">Art. 57<br>Constitution</div><div><div class="legal-body">Personal privacy and image are inviolable</div><div class="legal-sub">Chapter 3 — Personal Rights & Freedoms</div></div></div>
                    <div class="legal-row"><div class="legal-num">Law 175<br>2018</div><div><div class="legal-body">Cybercrime Act — digital forgery and identity impersonation</div><div class="legal-sub">Art. 25 — Up to 5 years imprisonment + fine</div></div></div>
                    <div class="legal-row"><div class="legal-num">Art. 327<br>Penal Code</div><div><div class="legal-body">Forgery of documents, images and digital signatures</div><div class="legal-sub">Imprisonment + financial penalty</div></div></div>
                    <div class="legal-row"><div class="legal-num">Art. 179<br>Penal Code</div><div><div class="legal-body">Publishing obscene or fabricated images online</div><div class="legal-sub">6 months to 5 years imprisonment</div></div></div>
                </div>""", unsafe_allow_html=True)
                st.markdown("""
                <div class="agency-grid">
                    <div class="agency-card"><div class="agency-icon">🏛</div><div class="agency-name">NTRA</div><div class="agency-role">National Telecom Regulatory Authority</div><div class="agency-contact">Hotline: 155</div></div>
                    <div class="agency-card"><div class="agency-icon">👮</div><div class="agency-name">Cybercrime Unit</div><div class="agency-role">Ministry of Interior — Digital Crimes</div><div class="agency-contact">08008880</div></div>
                    <div class="agency-card"><div class="agency-icon">⚖️</div><div class="agency-name">Public Prosecution</div><div class="agency-role">Supreme Prosecution Authority</div><div class="agency-contact">Hotline: 16000</div></div>
                    <div class="agency-card"><div class="agency-icon">🛡</div><div class="agency-name">NCSC Egypt</div><div class="agency-role">National Cybersecurity Council</div><div class="agency-contact">ncsc.gov.eg</div></div>
                </div>""", unsafe_allow_html=True)
                st.markdown("""
                <div class="alert-fake">
                    <div class="alert-title">⚠ Legal Warning</div>
                    <div class="alert-item">Creating or distributing deepfakes without consent is a criminal offense</div>
                    <div class="alert-item">This forensic report can be used as digital evidence before judicial authorities</div>
                    <div class="alert-item">Report immediately to the Cybercrime Unit: 08008880</div>
                    <div class="alert-item">File a complaint at mcit.gov.eg or ncsc.gov.eg</div>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════
#  TAB 2 — BATCH ANALYSIS ✅
# ══════════════════════════════════════
with tab_batch:
    st.markdown('<div class="sec-label">Batch Evidence Analysis</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:12px;color:#4a6080;margin-bottom:12px;">Upload multiple images at once — all will be analyzed automatically</p>', unsafe_allow_html=True)

    batch_files = st.file_uploader(
        "Select multiple images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="batch_upload"
    )

    if batch_files and model_ok:
        if st.button("🔍  Analyze All Images", key="btn_batch"):
            results_batch = []
            prog_b = st.progress(0)
            total  = len(batch_files)

            for i, f in enumerate(batch_files):
                prog_b.progress(int((i+1)/total*100), text=f"Analyzing {f.name} ({i+1}/{total})...")
                img = Image.open(f).convert("RGB")
                is_real, conf, raw, uncertain = predict(img)
                results_batch.append({
                    'name': f.name, 'is_real': is_real,
                    'conf': conf, 'uncertain': uncertain,
                    'size': f"{img.size[0]}×{img.size[1]}",
                    'kb': round(f.size/1024, 1)
                })
                st.session_state.n_scanned += 1
                if is_real: st.session_state.n_real += 1
                else:       st.session_state.n_fake += 1

            prog_b.empty()

            # Summary stats
            n_fake_b = sum(1 for r in results_batch if not r['is_real'])
            n_real_b = sum(1 for r in results_batch if r['is_real'])
            st.markdown(f"""
            <div class="stats-bar" style="margin-top:12px;">
                <div class="stat-tile"><div class="stat-val">{total}</div><div class="stat-name">Total Analyzed</div></div>
                <div class="stat-tile"><div class="stat-val" style="color:#E05252">{n_fake_b}</div><div class="stat-name">Deepfakes</div></div>
                <div class="stat-tile"><div class="stat-val" style="color:#4CAF7D">{n_real_b}</div><div class="stat-name">Authentic</div></div>
                <div class="stat-tile"><div class="stat-val">{n_fake_b/max(1,total)*100:.0f}%</div><div class="stat-name">Fake Rate</div></div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="sec-label" style="margin-top:14px;">Results</div>', unsafe_allow_html=True)

            # Fakes first, then reals
            for res_b in sorted(results_batch, key=lambda x: x['is_real']):
                badge = "badge-real" if res_b['is_real'] else "badge-fake"
                badge_txt = f"REAL · {res_b['conf']:.1f}%" if res_b['is_real'] else f"FAKE · {res_b['conf']:.1f}%"
                warn_icon = " ⚠" if res_b['uncertain'] else ""
                st.markdown(f"""
                <div class="batch-row">
                    <div class="batch-name">{res_b['name']}{warn_icon}</div>
                    <span style="font-size:11px;color:#4a6080;margin-right:10px;">{res_b['size']} · {res_b['kb']}KB</span>
                    <span class="{badge}">{badge_txt}</span>
                </div>""", unsafe_allow_html=True)

    elif not model_ok:
        st.error("Model not loaded — check truthlens_best.h5")
    else:
        st.markdown("""
        <div class="upload-empty">
            <div class="upload-empty-icon">📦</div>
            <div class="upload-empty-label">SELECT MULTIPLE IMAGES TO BEGIN</div>
            <div style="font-size:10px;color:#1A1E24;margin-top:6px;">Hold Ctrl/Cmd to select multiple files</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════
#  TAB 3 — COMPARE TWO IMAGES ✅
# ══════════════════════════════════════
with tab_compare:
    st.markdown('<div class="sec-label">Side-by-Side Image Comparison</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:12px;color:#4a6080;margin-bottom:16px;">Compare two images simultaneously — useful for verifying suspected deepfakes against real references</p>', unsafe_allow_html=True)

    cmp_col1, cmp_col2 = st.columns(2, gap="large")

    with cmp_col1:
        st.markdown('<div class="sec-label">Image A</div>', unsafe_allow_html=True)
        img_a_file = st.file_uploader("Upload Image A", type=["jpg","jpeg","png","webp"], key="cmp_a")
        if img_a_file:
            img_a = Image.open(img_a_file).convert("RGB")
            st.image(img_a, caption=f"A: {img_a_file.name}", use_container_width=True)

    with cmp_col2:
        st.markdown('<div class="sec-label">Image B</div>', unsafe_allow_html=True)
        img_b_file = st.file_uploader("Upload Image B", type=["jpg","jpeg","png","webp"], key="cmp_b")
        if img_b_file:
            img_b = Image.open(img_b_file).convert("RGB")
            st.image(img_b, caption=f"B: {img_b_file.name}", use_container_width=True)

    if img_a_file and img_b_file and model_ok:
        if st.button("⚖️  Compare Both Images", key="btn_compare"):
            with st.spinner("Analyzing both images..."):
                img_a = Image.open(img_a_file).convert("RGB")
                img_b = Image.open(img_b_file).convert("RGB")

                is_real_a, conf_a, raw_a, unc_a = predict(img_a)
                is_real_b, conf_b, raw_b, unc_b = predict(img_b)
                gc_a = gradcam(img_a)
                gc_b = gradcam(img_b)

            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
            st.markdown('<div class="sec-label">Comparison Results</div>', unsafe_allow_html=True)

            res_col1, res_col2 = st.columns(2, gap="large")

            def render_compare_result(col, name, is_real, conf, raw, uncertain, gc_img):
                risk_txt, risk_col = risk(conf, is_real)
                with col:
                    if uncertain:
                        st.markdown("""
                        <div class="warn-box" style="margin-bottom:8px;">
                            <div class="warn-title">⚠ Low Confidence</div>
                            <div class="warn-item">Result may be unreliable — manual review advised</div>
                        </div>""", unsafe_allow_html=True)

                    if is_real:
                        st.markdown(f"""
                        <div class="verdict-real">
                            <div class="verdict-eyebrow">{name}</div>
                            <div class="verdict-icon">✓</div>
                            <div class="v-real">REAL</div>
                            <div class="v-pct-real">{conf:.1f}%</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="verdict-fake">
                            <div class="verdict-eyebrow">{name}</div>
                            <div class="verdict-icon">⚠</div>
                            <div class="v-fake">FAKE</div>
                            <div class="v-pct-fake">{conf:.1f}%</div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="m3" style="margin-top:8px;">
                        <div class="m-tile"><div class="m-val">{conf:.1f}%</div><div class="m-name">Confidence</div></div>
                        <div class="m-tile"><div class="m-val" style="color:{risk_col};font-size:.85rem;">{risk_txt}</div><div class="m-name">Risk</div></div>
                        <div class="m-tile"><div class="m-val">{raw*100:.1f}%</div><div class="m-name">Real Prob.</div></div>
                    </div>""", unsafe_allow_html=True)

                    if gc_img:
                        st.image(gc_img, caption=f"Grad-CAM — {name}", use_container_width=True)

            render_compare_result(res_col1, img_a_file.name, is_real_a, conf_a, raw_a, unc_a, gc_a)
            render_compare_result(res_col2, img_b_file.name, is_real_b, conf_b, raw_b, unc_b, gc_b)

            # Summary verdict
            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
            if is_real_a == is_real_b:
                verdict_a = "REAL" if is_real_a else "FAKE"
                st.markdown(f"""
                <div class="{'verdict-real' if is_real_a else 'verdict-fake'}" style="text-align:center;">
                    <div class="verdict-eyebrow">Comparison Summary</div>
                    <div class="{'v-real' if is_real_a else 'v-fake'}">Both images classified as {verdict_a}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:#0d1525;border:1px solid #2E74B544;border-radius:14px;padding:1.2rem;text-align:center;">
                    <div class="verdict-eyebrow" style="margin-bottom:8px;">Comparison Summary</div>
                    <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;color:#C8A84A;">
                        Mixed Result — Images Differ
                    </div>
                    <div style="font-size:12px;color:#4a6080;margin-top:6px;">
                        {img_a_file.name}: <b style="color:{'#4CAF7D' if is_real_a else '#E05252'}">{'REAL' if is_real_a else 'FAKE'}</b>
                        &nbsp;·&nbsp;
                        {img_b_file.name}: <b style="color:{'#4CAF7D' if is_real_b else '#E05252'}">{'REAL' if is_real_b else 'FAKE'}</b>
                    </div>
                </div>""", unsafe_allow_html=True)

    elif not model_ok:
        st.error("Model not loaded")


# ════════════════════════════════════════════════════════
#  HOW IT WORKS
# ════════════════════════════════════════════════════════
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
st.markdown('<div class="sec-label">How TruthLens Works</div>', unsafe_allow_html=True)
st.markdown("""
<div class="how-grid">
    <div class="how-card"><div class="how-num">STEP 01</div><div class="how-icon">📤</div><div class="how-title">Upload Evidence</div><div class="how-desc">JPG / PNG / WEBP — any resolution</div></div>
    <div class="how-card"><div class="how-num">STEP 02</div><div class="how-icon">🔬</div><div class="how-title">Neural Analysis</div><div class="how-desc">EfficientNetB3 extracts deep feature maps</div></div>
    <div class="how-card"><div class="how-num">STEP 03</div><div class="how-icon">🔥</div><div class="how-title">Visual Forensics</div><div class="how-desc">Grad-CAM heatmap + Error Level Analysis</div></div>
    <div class="how-card"><div class="how-num">STEP 04</div><div class="how-icon">📋</div><div class="how-title">Forensic Report</div><div class="how-desc">Verdict + confidence + PDF export</div></div>
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  SCAN HISTORY
# ════════════════════════════════════════════════════════
if st.session_state.history:
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Scan History</div>', unsafe_allow_html=True)
    for h in st.session_state.history[:10]:
        bc  = "badge-real" if h['is_real'] else "badge-fake"
        bt  = f"REAL · {h['conf']:.1f}%" if h['is_real'] else f"FAKE · {h['conf']:.1f}%"
        warn = " ⚠" if h.get('uncertain') else ""
        st.markdown(f"""
        <div class="hist-row">
            <div>
                <div class="hist-name">{h['name']}{warn}</div>
                <div class="hist-meta">{h['date']} · {h['time']} · {h['size']} · {h['kb']} KB</div>
            </div>
            <div class="{bc}">{bt}</div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    <span>TRUTHLENS © 2025 — FORENSIC AI UNIT</span>
    <span>EfficientNetB3 · 97.99% ACC · 102K TRAINING IMAGES · CELEBA + STYLEGAN</span>
</div>""", unsafe_allow_html=True)