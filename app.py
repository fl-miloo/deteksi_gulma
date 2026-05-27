"""
==========================================================
  GULMAIFY — app.py
  Flask + MySQL + YOLOv8 + Google Login
==========================================================
"""

from flask import (
    Flask, render_template, request, redirect,
    url_for, jsonify, session, flash
)
from ultralytics import YOLO
import mysql.connector
import os, uuid, cv2
import requests 
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash
from functools import wraps
from datetime import datetime
from authlib.integrations.flask_client import OAuth
from collections import OrderedDict
from math import ceil
from dotenv import load_dotenv

# ============================================================
#  KONFIGURASI
# ============================================================
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
MODEL_FOLDER  = os.path.join(BASE_DIR, "models")
ALLOWED_EXT   = {"png", "jpg", "jpeg", "webp"}

app.config['UPLOAD_FOLDER']       = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH']  = 16 * 1024 * 1024  # 16MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER,  exist_ok=True)

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
}

# ============================================================
#  GOOGLE OAUTH
# ============================================================
GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

oauth  = OAuth(app)
google = oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    access_token_url='https://accounts.google.com/o/oauth2/token',
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    api_base_url='https://www.googleapis.com/oauth2/v1/',
    client_kwargs={'scope': 'email profile'},
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration'
)

# ============================================================
#  LOAD MODEL YOLO
# ============================================================
MODEL_PATH = os.path.join(MODEL_FOLDER, "best.pt")

def load_model():
    if os.path.exists(MODEL_PATH):
        print(f"[MODEL] Menggunakan: {MODEL_PATH}")
        return YOLO(MODEL_PATH)
    print("[MODEL] best.pt tidak ditemukan, fallback ke yolov8n.pt")
    return YOLO("yolov8n.pt")

model = load_model()

# ============================================================
#  UTILITAS
# ============================================================
def get_db():
    return mysql.connector.connect(**DB_CONFIG)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def get_or_create_session_id():
    if session.get("user_logged_in") and session.get("user_id"):
        sid = f"user_{session['user_id']}"
        session["session_id"] = sid
        return sid
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]

def get_forum_posts(cur, sort='terbaru', limit=5):
    """Helper: ambil forum posts untuk ditampilkan di landing/beranda."""
    if sort == 'teramai':
        cur.execute("""
            SELECT f.id_forum, f.judul, f.deskripsi, f.tgl_post,
                   u.nama as user_nama,
                   (SELECT COUNT(*) FROM tb_komentar k
                    WHERE k.id_forum = f.id_forum AND k.is_deleted = 0) as jumlah_komentar
            FROM tb_forum f
            JOIN tb_user u ON f.user_id = u.id_user
            WHERE f.is_active = 1
            ORDER BY jumlah_komentar DESC, f.tgl_post DESC
            LIMIT %s
        """, (limit,))
    else:
        cur.execute("""
            SELECT f.id_forum, f.judul, f.deskripsi, f.tgl_post,
                   u.nama as user_nama,
                   (SELECT COUNT(*) FROM tb_komentar k
                    WHERE k.id_forum = f.id_forum AND k.is_deleted = 0) as jumlah_komentar
            FROM tb_forum f
            JOIN tb_user u ON f.user_id = u.id_user
            WHERE f.is_active = 1
            ORDER BY f.tgl_post DESC
            LIMIT %s
        """, (limit,))
    return cur.fetchall()

# ============================================================
#  DECORATORS
# ============================================================
def user_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user_logged_in"):
            flash("Silakan masuk terlebih dahulu.", "info")
            return redirect(url_for("user_login"))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return decorated

# ============================================================
#  LANDING PAGE (PUBLIK)
# ============================================================
@app.route('/')
def landing():
    db   = get_db()
    cur  = db.cursor(dictionary=True)
    sort = request.args.get('sort', 'terbaru')

    cur.execute("SELECT id_gulma, nama_umum, nama_latin, deskripsi_umum, foto_referensi FROM tb_gulma ORDER BY id_gulma LIMIT 5")
    list_gulma = cur.fetchall()

    forum_posts = get_forum_posts(cur, sort=sort, limit=3)

    cur.close()
    db.close()
    return render_template('landing.html', list_gulma=list_gulma, forum_posts=forum_posts, sort=sort)

# ============================================================
#  AUTH USER — GOOGLE
# ============================================================
@app.route("/login")
def user_login():
    if session.get("user_logged_in"):
        return redirect(url_for("beranda"))
    return render_template("login.html")

@app.route("/user/google/login")
def user_google_login():
    redirect_uri = url_for("user_google_callback", _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route("/user/google/callback")
def user_google_callback():
    try:
        google.authorize_access_token()
        resp = google.get("userinfo")
        user_info = resp.json()
        if not user_info:
            flash("Gagal mendapatkan data dari Google!", "danger")
            return redirect(url_for("user_login"))

        email = user_info.get("email")
        google_id = user_info.get("id")
        fullname = user_info.get("name", email.split("@")[0])
        foto = user_info.get("picture", "")

        if not email:
            flash("Email tidak ditemukan!", "danger")
            return redirect(url_for("user_login"))

        db = get_db()
        cur = db.cursor(dictionary=True)
        cur.execute("SELECT * FROM tb_user WHERE google_id = %s OR email = %s", (google_id, email))
        existing = cur.fetchone()

        if existing:
            # Update last_login dan foto_profile (paksa update dengan foto terbaru dari Google)
            cur.execute("""
                UPDATE tb_user
                SET last_login = %s, foto_profile = %s
                WHERE id_user = %s
            """, (datetime.now(), foto, existing["id_user"]))
            db.commit()

            # Ambil ulang data user yang sudah diupdate
            cur.execute("SELECT * FROM tb_user WHERE id_user = %s", (existing["id_user"],))
            user_data = cur.fetchone()

            session["user_logged_in"] = True
            session["user_id"] = user_data["id_user"]
            session["user_nama"] = user_data["nama"]
            session["user_email"] = user_data["email"]
            session["user_foto"] = user_data.get("foto_profile") or foto
            session["session_id"] = f"user_{user_data['id_user']}"
            flash(f"Selamat datang kembali, {user_data['nama']}!", "success")
        else:
            # User baru
            cur.execute("""
                INSERT INTO tb_user (nama, email, google_id, foto_profile, last_login)
                VALUES (%s, %s, %s, %s, %s)
            """, (fullname, email, google_id, foto, datetime.now()))
            db.commit()
            id_user = cur.lastrowid
            cur.execute("SELECT * FROM tb_user WHERE id_user = %s", (id_user,))
            user_data = cur.fetchone()

            session["user_logged_in"] = True
            session["user_id"] = user_data["id_user"]
            session["user_nama"] = user_data["nama"]
            session["user_email"] = user_data["email"]
            session["user_foto"] = user_data.get("foto_profile") or foto
            session["session_id"] = f"user_{user_data['id_user']}"
            flash(f"Akun berhasil dibuat! Selamat datang, {user_data['nama']}!", "success")

        cur.close()
        db.close()
        return redirect(url_for("splash"))

    except Exception as e:
        print(f"Google Login Error: {e}")
        flash(f"Terjadi kesalahan saat login: {e}", "danger")
        return redirect(url_for("user_login"))

@app.route("/logout")
def user_logout():
    session.clear()
    flash("Anda telah keluar. Sampai jumpa!", "success")
    return redirect(url_for("landing"))

@app.route("/splash")
def splash():
    return render_template("splash.html")

# ============================================================
#  BERANDA USER (SETELAH LOGIN)
# ============================================================
@app.route("/beranda")
@user_required
def beranda():
    db   = get_db()
    cur  = db.cursor(dictionary=True)
    sort = request.args.get('sort', 'terbaru')

    cur.execute("SELECT id_gulma, nama_umum, nama_latin, deskripsi_umum, foto_referensi FROM tb_gulma ORDER BY id_gulma LIMIT 5")
    list_gulma = cur.fetchall()

    forum_posts = get_forum_posts(cur, sort=sort, limit=3)

    cur.close()
    db.close()
    return render_template('beranda.html', list_gulma=list_gulma, forum_posts=forum_posts, sort=sort)

# ============================================================
#  KAMERA & DETEKSI
# ============================================================
@app.route("/kamera")
@user_required
def kamera():
    return render_template("kamera.html")

@app.route("/deteksi", methods=["POST"])
@user_required
def deteksi():
    session_id = get_or_create_session_id()
    user_id    = session.get("user_id")

    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file"}), 400

    file = request.files["file"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Format file tidak didukung"}), 400

    # Ambil data lokasi dari form (dikirim dari frontend JS)
    lokasi_lat    = request.form.get("lokasi_lat", None)
    lokasi_lng    = request.form.get("lokasi_lng", None)
    lokasi_alamat = request.form.get("lokasi_alamat", None)
    lokasi_kota   = request.form.get("lokasi_kota", None)
    lokasi_provinsi = request.form.get("lokasi_provinsi", None)

    # Convert ke float jika ada
    if lokasi_lat:
        try: lokasi_lat = float(lokasi_lat)
        except: lokasi_lat = None
    if lokasi_lng:
        try: lokasi_lng = float(lokasi_lng)
        except: lokasi_lng = None

    ext      = file.filename.rsplit(".", 1)[1].lower()
    unique   = uuid.uuid4().hex
    ori_name = f"ori_{unique}.{ext}"
    ori_path = os.path.join(UPLOAD_FOLDER, ori_name)
    file.save(ori_path)

    results = model.predict(source=ori_path, conf=0.35, verbose=False)
    boxes   = results[0].boxes

    if len(boxes) == 0:
        return jsonify({"detected": False, "ori_file": ori_name,
                        "message": "Tidak ada gulma yang terdeteksi."})

    annotated = results[0].plot()
    ann_name  = f"ann_{unique}.jpg"
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, ann_name), annotated)

    sesi_foto = uuid.uuid4().hex[:12]
    db        = get_db()
    cur       = db.cursor(dictionary=True)

    detections = []
    for box in boxes:
        label      = model.names[int(box.cls[0])]
        confidence = float(box.conf[0])
        if confidence >= 0.35:
            cur.execute("""
                INSERT INTO tb_riwayat
                (session_id, user_id, id_sesi_foto, foto_upload, foto_annotasi,
                 label_ai, confidence_score,
                 lokasi_lat, lokasi_lng, lokasi_alamat, lokasi_kota, lokasi_provinsi)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (session_id, user_id, sesi_foto, ori_name, ann_name,
                  label, confidence,
                  lokasi_lat, lokasi_lng, lokasi_alamat, lokasi_kota, lokasi_provinsi))
            id_riwayat = cur.lastrowid
            detections.append({
                "id_riwayat": id_riwayat,
                "label": label,
                "confidence": round(confidence * 100, 1),
            })

    db.commit()
    cur.close()
    db.close()

    if not detections:
        return jsonify({"detected": False, "ori_file": ori_name,
                        "message": "Confidence terlalu rendah (<35%)."})

    return jsonify({
        "detected": True,
        "ori_file": ori_name,
        "ann_file": ann_name,
        "sesi_foto": sesi_foto,
        "detections": detections,
        "message": f"Ditemukan {len(detections)} objek gulma."
    })
# ============================================================
#  DETAIL SESI DETEKSI
# ============================================================
@app.route("/detail-sesi/<sesi_foto>")
@user_required
def detail_sesi(sesi_foto):
    db = get_db()
    cur = db.cursor(dictionary=True)

    cur.execute("""
        SELECT r.*, g.nama_umum, g.nama_latin, g.deskripsi_umum,
               g.bentuk_daun, g.warna_daun, g.karakter_batang, g.karakter_akar,
               g.cara_berkembangbiak, g.waktu_kendali_terbaik,
               g.metode_mekanik, g.metode_kimiawi, g.foto_referensi
        FROM tb_riwayat r
        LEFT JOIN tb_gulma g ON r.label_ai = g.label_model
        WHERE r.id_sesi_foto = %s AND r.user_id = %s
        ORDER BY r.confidence_score DESC
    """, (sesi_foto, session["user_id"]))
    
    deteksi_list = cur.fetchall()

    if not deteksi_list:
        cur.close()
        db.close()
        return "Data tidak ditemukan", 404

    cur.execute("""
        SELECT label_model, nama_umum, nama_latin, foto_referensi
        FROM tb_gulma
        ORDER BY nama_umum ASC
    """)
    semua_gulma = cur.fetchall()

    cur.close()
    db.close()

    foto_annotasi = deteksi_list[0].get('foto_annotasi')
    foto_upload   = deteksi_list[0].get('foto_upload')
    total         = len(deteksi_list)
    benar         = sum(1 for d in deteksi_list if d['status_akurasi'] == 'Benar')
    salah         = sum(1 for d in deteksi_list if d['status_akurasi'] == 'Salah')
    verified      = benar + salah
    akurasi_sesi  = round((benar / verified * 100), 1) if verified > 0 else 0

    return render_template("detail_sesi.html",
        deteksi_list=deteksi_list,
        semua_gulma=semua_gulma,
        foto_annotasi=foto_annotasi,
        foto_upload=foto_upload,
        total=total,
        benar=benar,
        salah=salah,
        verified=verified,
        akurasi_sesi=akurasi_sesi,
        sesi_foto=sesi_foto   # <- tambahkan ini agar tersedia di template
    )

# ============================================================
#  RIWAYAT USER
# ============================================================
@app.route("/riwayat")
@user_required
def riwayat():
    user_id = session["user_id"]
    page = request.args.get('page', 1, type=int)
    per_page = 9  # jumlah item per halaman
    offset = (page - 1) * per_page

    db = get_db()
    cur = db.cursor(dictionary=True)

    # Hitung total data untuk pagination
    cur.execute("SELECT COUNT(DISTINCT id_sesi_foto) AS total FROM tb_riwayat WHERE user_id = %s", (user_id,))
    total_data = cur.fetchone()["total"]
    total_pages = ceil(total_data / per_page) if total_data > 0 else 1

    # Ambil data dengan LIMIT dan OFFSET
    cur.execute("""
        SELECT r.id_riwayat, r.id_sesi_foto, r.foto_upload, r.foto_annotasi,
               r.label_ai, r.confidence_score, r.tgl_deteksi, r.status_akurasi,
               r.is_public, g.nama_umum
        FROM tb_riwayat r
        LEFT JOIN tb_gulma g ON r.label_ai = g.label_model
        WHERE r.user_id = %s
        ORDER BY r.tgl_deteksi DESC
        LIMIT %s OFFSET %s
    """, (user_id, per_page, offset))
    rows = cur.fetchall()
    cur.close()
    db.close()

    # Kelompokkan berdasarkan sesi_foto (sama seperti kode lama Anda)
    sesi_dict = OrderedDict()
    for r in rows:
        key = r["id_sesi_foto"]
        if key not in sesi_dict:
            sesi_dict[key] = {
                "id_sesi_foto": key,
                "foto_upload":  r["foto_upload"],
                "foto_annotasi": r["foto_annotasi"],
                "tgl_deteksi":  r["tgl_deteksi"],
                "deteksi":      []
            }
        sesi_dict[key]["deteksi"].append(r)

    riwayat_list = []
    for key, sesi in sesi_dict.items():
        first = sesi["deteksi"][0]
        riwayat_list.append({
            "id_sesi_foto":   key,
            "foto_upload":    first["foto_upload"],
            "foto_annotasi":  first["foto_annotasi"],
            "tgl_deteksi":    first["tgl_deteksi"],
            "total":          len(sesi["deteksi"]),
            "label_ai":       first["label_ai"],
            "nama_umum":      first.get("nama_umum"),
            "confidence_score": first["confidence_score"],
            "status_akurasi": first["status_akurasi"],
            "is_public":      first["is_public"],
        })

    return render_template("riwayat.html", riwayat=riwayat_list, page=page, total_pages=total_pages)

# ============================================================
#  hapus riwayat
# ============================================================
@app.route("/hapus-riwayat/<sesi_foto>", methods=["POST"])
@user_required
def hapus_riwayat(sesi_foto):
    db = get_db()
    cur = db.cursor()
    cur.execute("DELETE FROM tb_riwayat WHERE id_sesi_foto = %s AND user_id = %s", (sesi_foto, session["user_id"]))
    db.commit()
    affected = cur.rowcount
    cur.close()
    db.close()
    if affected:
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "message": "Data tidak ditemukan"}), 404
# ============================================================
#  BAGIKAN KE FORUM (DENGAN FORM EDIT)
# ============================================================
@app.route("/bagikan-ke-forum/<sesi_foto>", methods=["GET", "POST"])
@user_required
def bagikan_ke_forum(sesi_foto):
    db = get_db()
    cur = db.cursor(dictionary=True)
    
    cur.execute("""
        SELECT r.id_riwayat, r.label_ai, r.confidence_score, g.nama_umum
        FROM tb_riwayat r
        LEFT JOIN tb_gulma g ON r.label_ai = g.label_model
        WHERE r.id_sesi_foto = %s AND r.user_id = %s
        ORDER BY r.confidence_score DESC
    """, (sesi_foto, session["user_id"]))
    deteksi_list = cur.fetchall()
    
    if not deteksi_list:
        cur.close()
        db.close()
        flash("Data tidak ditemukan!", "danger")
        return redirect(url_for("riwayat"))
    
    first_id_riwayat = deteksi_list[0]['id_riwayat']
    
    if request.method == "POST":
        judul = request.form.get("judul", "").strip()
        deskripsi = request.form.get("deskripsi", "").strip()
        
        if not judul:
            flash("Judul tidak boleh kosong!", "danger")
            return redirect(url_for("bagikan_ke_forum", sesi_foto=sesi_foto))
        
        cur.execute("UPDATE tb_riwayat SET is_public = 1, tgl_dipublik = NOW() WHERE id_sesi_foto = %s", (sesi_foto,))
        
        cur.execute("SELECT id_forum FROM tb_forum WHERE id_sesi_foto = %s", (sesi_foto,))
        existing = cur.fetchone()
        if not existing:
            cur.execute("""
                INSERT INTO tb_forum (id_sesi_foto, id_riwayat, user_id, judul, deskripsi)
                VALUES (%s, %s, %s, %s, %s)
            """, (sesi_foto, first_id_riwayat, session["user_id"], judul, deskripsi))
        else:
            cur.execute("""
                UPDATE tb_forum SET judul = %s, deskripsi = %s, id_riwayat = %s
                WHERE id_sesi_foto = %s
            """, (judul, deskripsi, first_id_riwayat, sesi_foto))
        
        db.commit()
        cur.close()
        db.close()
        flash("Berhasil dibagikan ke forum!", "success")
        return redirect(url_for("forum"))
    
    cur.close()
    db.close()
    return render_template("bagikan_ke_forum.html", deteksi_list=deteksi_list, sesi_foto=sesi_foto)

# ============================================================
#  BUAT POSTINGAN BARU (dengan upload foto)
# ============================================================
@app.route("/buat-postingan", methods=["GET", "POST"])
@user_required
def buat_postingan():
    if request.method == "POST":
        judul = request.form.get("judul", "").strip()
        deskripsi = request.form.get("deskripsi", "").strip()
        if not judul:
            flash("Judul tidak boleh kosong!", "danger")
            return redirect(url_for("buat_postingan"))
        
        # Ambil data lokasi dari form
        lokasi_lat = request.form.get("lokasi_lat")
        lokasi_lng = request.form.get("lokasi_lng")
        lokasi_alamat = request.form.get("lokasi_alamat")
        lokasi_kota = request.form.get("lokasi_kota")
        lokasi_provinsi = request.form.get("lokasi_provinsi")
        
        # Konversi ke float jika ada
        if lokasi_lat:
            try:
                lokasi_lat = float(lokasi_lat)
            except:
                lokasi_lat = None
        if lokasi_lng:
            try:
                lokasi_lng = float(lokasi_lng)
            except:
                lokasi_lng = None
        
        foto_post = None
        if 'foto_post' in request.files:
            f = request.files['foto_post']
            if f and allowed_file(f.filename):
                ext = f.filename.rsplit('.', 1)[1].lower()
                fname = f"forum_{uuid.uuid4().hex}.{ext}"
                f.save(os.path.join(app.config['UPLOAD_FOLDER'], fname))
                foto_post = fname
        
        db = get_db()
        cur = db.cursor()
        cur.execute("""
            INSERT INTO tb_forum (user_id, judul, deskripsi, foto_post, id_riwayat,
                lokasi_lat, lokasi_lng, lokasi_alamat, lokasi_kota, lokasi_provinsi)
            VALUES (%s, %s, %s, %s, NULL, %s, %s, %s, %s, %s)
        """, (session["user_id"], judul, deskripsi, foto_post,
              lokasi_lat, lokasi_lng, lokasi_alamat, lokasi_kota, lokasi_provinsi))
        db.commit()
        cur.close()
        db.close()
        flash("Postingan berhasil dibuat!", "success")
        return redirect(url_for("forum"))
    
    return render_template("buat_postingan.html")
# ============================================================
#  FORUM (PILIH TEMPLATE BERDASARKAN LOGIN)
# ============================================================

@app.route("/forum")
def forum():
    # Ambil parameter filter & sort
    sort = request.args.get('sort', 'terbaru')
    page = request.args.get('page', 1, type=int)
    per_page = 6
    bulan = request.args.get('bulan', type=int)   # 1-12
    tahun = request.args.get('tahun', type=int)

    db = get_db()
    cur = db.cursor(dictionary=True)

    # --- Ambil daftar tahun unik dari semua postingan aktif (untuk dropdown filter) ---
    cur.execute("""
        SELECT DISTINCT YEAR(tgl_post) as tahun
        FROM tb_forum
        WHERE is_active = 1
        ORDER BY tahun DESC
    """)
    tahun_list = [row['tahun'] for row in cur.fetchall()]

    # --- Bangun query dengan filter bulan/tahun ---
    base_sql = """
        FROM tb_forum f
        JOIN tb_user u ON f.user_id = u.id_user
        LEFT JOIN tb_riwayat r ON f.id_riwayat = r.id_riwayat
        WHERE f.is_active = 1
    """
    params = []

    if bulan:
        base_sql += " AND MONTH(f.tgl_post) = %s"
        params.append(bulan)
    if tahun:
        base_sql += " AND YEAR(f.tgl_post) = %s"
        params.append(tahun)

    # --- Hitung total data dengan filter ---
    count_sql = "SELECT COUNT(*) as total " + base_sql
    cur.execute(count_sql, params)
    total_items = cur.fetchone()['total']
    total_pages = ceil(total_items / per_page) if total_items > 0 else 1

    # Pastikan page valid
    if page > total_pages and total_pages > 0:
        cur.close()
        db.close()
        return redirect(url_for('forum', page=total_pages, sort=sort, bulan=bulan, tahun=tahun))

    offset = (page - 1) * per_page

    # --- Query utama dengan sorting dan pagination ---
    if sort == 'teramai':
        order_by = "ORDER BY jumlah_komentar DESC, f.tgl_post DESC"
        # Karena jumlah_komentar adalah subquery, perlu ditulis ulang di SELECT utama.
        # Kita gunakan subquery di SELECT agar bisa di-ORDER.
        select_sql = """
            SELECT f.id_forum, f.judul, f.deskripsi, f.tgl_post, f.user_id,
                   u.nama as user_nama, u.foto_profile as user_foto,
                   (SELECT COUNT(*) FROM tb_komentar k WHERE k.id_forum = f.id_forum AND k.is_deleted = 0) as jumlah_komentar,
                   r.label_ai, r.confidence_score, r.foto_upload, r.foto_annotasi,
                   f.foto_post
        """ + base_sql + """
            ORDER BY jumlah_komentar DESC, f.tgl_post DESC
            LIMIT %s OFFSET %s
        """
    else:
        select_sql = """
            SELECT f.id_forum, f.judul, f.deskripsi, f.tgl_post, f.user_id,
                   u.nama as user_nama, u.foto_profile as user_foto,
                   (SELECT COUNT(*) FROM tb_komentar k WHERE k.id_forum = f.id_forum AND k.is_deleted = 0) as jumlah_komentar,
                   r.label_ai, r.confidence_score, r.foto_upload, r.foto_annotasi,
                   f.foto_post
        """ + base_sql + """
            ORDER BY f.tgl_post DESC
            LIMIT %s OFFSET %s
        """

    # Gabungkan parameter filter + limit & offset
    query_params = params + [per_page, offset]
    cur.execute(select_sql, query_params)
    posts = cur.fetchall()

    cur.close()
    db.close()

    # Data untuk template
    context = {
        'posts': posts,
        'page': page,
        'total_pages': total_pages,
        'sort': sort,
        'bulan_filter': bulan,
        'tahun_filter': tahun,
        'tahun_list': tahun_list
    }

    if session.get('user_logged_in'):
        return render_template('forum.html', **context)
    else:
        return render_template('forum_public.html', **context)
# ============================================================
#  FORUM PUBLIK (REDUNDAN, tapi biarkan)
# ============================================================
@app.route('/forum_public')
def forum_public():
    sort = request.args.get('sort', 'terbaru')
    db = get_db()
    cur = db.cursor(dictionary=True)
    
    if sort == 'teramai':
        cur.execute("""
            SELECT f.id_forum, f.judul, f.deskripsi, f.tgl_post,
                   u.nama as user_nama, u.foto_profile as user_foto,
                   (SELECT COUNT(*) FROM tb_komentar k WHERE k.id_forum = f.id_forum AND k.is_deleted = 0) as jumlah_komentar,
                   r.label_ai, r.confidence_score, r.foto_upload, r.foto_annotasi,
                   f.foto_post   -- TAMBAHKAN INI
            FROM tb_forum f
            JOIN tb_user u ON f.user_id = u.id_user
            LEFT JOIN tb_riwayat r ON f.id_riwayat = r.id_riwayat
            WHERE f.is_active = 1
            ORDER BY jumlah_komentar DESC, f.tgl_post DESC
        """)
    else:
        cur.execute("""
            SELECT f.id_forum, f.judul, f.deskripsi, f.tgl_post,
                   u.nama as user_nama, u.foto_profile as user_foto,
                   (SELECT COUNT(*) FROM tb_komentar k WHERE k.id_forum = f.id_forum AND k.is_deleted = 0) as jumlah_komentar,
                   r.label_ai, r.confidence_score, r.foto_upload, r.foto_annotasi,
                   f.foto_post   -- TAMBAHKAN INI
            FROM tb_forum f
            JOIN tb_user u ON f.user_id = u.id_user
            LEFT JOIN tb_riwayat r ON f.id_riwayat = r.id_riwayat
            WHERE f.is_active = 1
            ORDER BY f.tgl_post DESC
        """)
    posts = cur.fetchall()
    cur.close()
    db.close()
    return render_template('forum_public.html', posts=posts, sort=sort)

# ============================================================
#  DETAIL DISKUSI
# ============================================================
@app.route("/diskusi/<int:id_forum>")
def diskusi_detail(id_forum):
    db = get_db()
    cur = db.cursor(dictionary=True)
    
    # Ambil data forum dan user (tanpa riwayat dulu)
    cur.execute("""
        SELECT f.*, u.nama as user_nama, u.foto_profile as user_foto
        FROM tb_forum f
        JOIN tb_user u ON f.user_id = u.id_user
        WHERE f.id_forum = %s AND f.is_active = 1
    """, (id_forum,))
    post = cur.fetchone()
    if not post:
        cur.close()
        db.close()
        return "Diskusi tidak ditemukan", 404
    
    # Ambil data riwayat jika ada id_riwayat
    if post.get('id_riwayat'):
        cur.execute("""
            SELECT r.*, g.nama_umum, g.nama_latin, g.deskripsi_umum,
                   g.bentuk_daun, g.warna_daun, g.karakter_batang, g.karakter_akar,
                   g.cara_berkembangbiak, g.waktu_kendali_terbaik,
                   g.metode_mekanik, g.metode_kimiawi, g.foto_referensi
            FROM tb_riwayat r
            LEFT JOIN tb_gulma g ON r.label_ai = g.label_model
            WHERE r.id_riwayat = %s
        """, (post['id_riwayat'],))
        riwayat = cur.fetchone()
        if riwayat:
            for key, value in riwayat.items():
                if key not in post or post[key] is None:
                    post[key] = value
    
    # Ambil semua deteksi dalam sesi yang sama (jika ada id_sesi_foto dari riwayat)
    deteksi_sesi = []
    if post.get('id_sesi_foto'):
        cur.execute("""
            SELECT r.*, g.nama_umum, g.nama_latin, g.deskripsi_umum,
                   g.bentuk_daun, g.warna_daun, g.karakter_batang, g.karakter_akar,
                   g.cara_berkembangbiak, g.waktu_kendali_terbaik,
                   g.metode_mekanik, g.metode_kimiawi, g.foto_referensi
            FROM tb_riwayat r
            LEFT JOIN tb_gulma g ON r.label_ai = g.label_model
            WHERE r.id_sesi_foto = %s
            ORDER BY r.confidence_score DESC
        """, (post['id_sesi_foto'],))
        deteksi_sesi = cur.fetchall()
    
    # Ambil komentar
    cur.execute("""
        SELECT k.*, u.nama as user_nama, u.foto_profile as user_foto
        FROM tb_komentar k
        JOIN tb_user u ON k.user_id = u.id_user
        WHERE k.id_forum = %s AND k.is_deleted = 0
        ORDER BY k.tgl_komentar ASC
    """, (id_forum,))
    comments = cur.fetchall()
    cur.close()
    db.close()
    
    if session.get('user_logged_in'):
        return render_template("diskusi_detail.html", post=post, comments=comments, deteksi_sesi=deteksi_sesi)
    else:
        return render_template("diskusi_detail_public.html", post=post, comments=comments, deteksi_sesi=deteksi_sesi)
    
@app.route("/komentar/<int:id_forum>", methods=["POST"])
@user_required
def tambah_komentar(id_forum):
    isi = request.form.get("isi_komentar", "").strip()
    if not isi:
        flash("Komentar tidak boleh kosong!", "danger")
        return redirect(url_for("diskusi_detail", id_forum=id_forum))
    
    foto_komentar = None
    if 'foto_komentar' in request.files:
        f = request.files['foto_komentar']
        if f and allowed_file(f.filename):
            fname = f"kom_{uuid.uuid4().hex}.{f.filename.rsplit('.',1)[1].lower()}"
            f.save(os.path.join(UPLOAD_FOLDER, fname))
            foto_komentar = fname
    
    db = get_db()
    cur = db.cursor()
    cur.execute(
        "INSERT INTO tb_komentar (id_forum, user_id, isi_komentar, foto_komentar) VALUES (%s, %s, %s, %s)",
        (id_forum, session["user_id"], isi, foto_komentar)
    )
    db.commit()
    cur.close(); db.close()
    flash("Komentar berhasil ditambahkan.", "success")
    return redirect(url_for("diskusi_detail", id_forum=id_forum))


@app.route("/hapus_postingan/<int:id_forum>")
def hapus_postingan(id_forum):
    if not session.get('user_logged_in'):
        flash('Anda harus login untuk menghapus postingan.', 'danger')
        return redirect(url_for('login'))

    db = get_db()
    cur = db.cursor(dictionary=True)

    # Cek apakah postingan ada dan milik user yang login
    cur.execute("SELECT user_id FROM tb_forum WHERE id_forum = %s AND is_active = 1", (id_forum,))
    forum = cur.fetchone()
    if not forum:
        cur.close()
        db.close()
        flash('Postingan tidak ditemukan.', 'danger')
        return redirect(url_for('forum'))

    if forum['user_id'] != session.get('user_id'):
        cur.close()
        db.close()
        flash('Anda tidak memiliki izin untuk menghapus postingan ini.', 'danger')
        return redirect(url_for('forum'))

    # Hapus komentar terkait
    cur.execute("DELETE FROM tb_komentar WHERE id_forum = %s", (id_forum,))
    # Hapus forum (hard delete; jika ingin soft delete, UPDATE is_active = 0)
    cur.execute("DELETE FROM tb_forum WHERE id_forum = %s", (id_forum,))
    db.commit()
    cur.close()
    db.close()

    flash('Postingan berhasil dihapus.', 'success')
    return redirect(url_for('forum', page=1))
# ============================================================
#  HAPUS KOMENTAR (USER & ADMIN)
# ============================================================
@app.route('/hapus-komentar/<int:id_komentar>', methods=['POST'])
def hapus_komentar(id_komentar):
    if not session.get('user_logged_in') and not session.get('admin_logged_in'):
        return jsonify({'status': 'error', 'message': 'Harus login'}), 403
    
    db = get_db()
    cur = db.cursor(dictionary=True)
    cur.execute("SELECT user_id, id_forum FROM tb_komentar WHERE id_komentar = %s AND is_deleted = 0", (id_komentar,))
    kom = cur.fetchone()
    if not kom:
        cur.close(); db.close()
        return jsonify({'status': 'error', 'message': 'Komentar tidak ditemukan'}), 404
    
    user_id = session.get('user_id')
    is_admin = session.get('admin_logged_in')
    if kom['user_id'] == user_id or is_admin:
        cur.execute("UPDATE tb_komentar SET is_deleted = 1, tgl_dihapus = NOW() WHERE id_komentar = %s", (id_komentar,))
        db.commit()
        cur.close(); db.close()
        return jsonify({'status': 'success'})
    else:
        cur.close(); db.close()
        return jsonify({'status': 'error', 'message': 'Tidak memiliki izin'}), 403

# ============================================================
#  ADMIN AUTH
# ============================================================
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        db = get_db()
        cur = db.cursor(dictionary=True)
        cur.execute("SELECT * FROM tb_admin WHERE username = %s AND is_active = 1", (username,))
        admin = cur.fetchone()
        cur.close()
        db.close()

        if admin and admin.get("password") and check_password_hash(admin["password"], password):
            session["admin_logged_in"] = True
            session["admin_id"] = admin["id_admin"]
            session["admin_username"] = admin["username"]
            session["admin_email"] = admin["email"]
            session["admin_foto"] = admin.get("foto_profile")  # tambahkan foto
            flash(f"Selamat datang, {admin['username']}!", "success")
            return redirect(url_for("dashboard"))
        flash("Username atau password salah!", "danger")
    return render_template("admin/login.html")

@app.route("/admin/google/login")
def admin_google_login():
    redirect_uri = url_for('admin_google_callback', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route("/admin/google/callback")
def admin_google_callback():
    try:
        google.authorize_access_token()
        resp = google.get('userinfo')
        user_info = resp.json()
        email = user_info.get('email', '')
        google_id = user_info.get('id', '')
        fullname = user_info.get('name', email.split('@')[0])
        foto = user_info.get('picture', '')

        ALLOWED_ADMIN_EMAILS = ['fatmalestari396@gmail.com', 'akun9spensa@gmail.com']
        if email not in ALLOWED_ADMIN_EMAILS:
            flash('Akses ditolak! Email tidak terdaftar sebagai admin.', 'danger')
            return redirect(url_for('admin_login'))

        db = get_db()
        cur = db.cursor(dictionary=True)
        cur.execute("SELECT * FROM tb_admin WHERE google_id = %s OR email = %s", (google_id, email))
        existing = cur.fetchone()

        if existing:
            cur.execute(
                "UPDATE tb_admin SET last_login = %s, foto_profile = COALESCE(%s, foto_profile) WHERE id_admin = %s",
                (datetime.now(), foto, existing['id_admin'])
            )
            db.commit()
            session["admin_logged_in"] = True
            session["admin_id"] = existing["id_admin"]
            session["admin_username"] = existing["username"]
            session["admin_email"] = existing["email"]
            session["admin_foto"] = existing.get("foto_profile") or foto   # simpan foto
            flash(f"Selamat datang kembali, {existing['username']}!", "success")
        else:
            username = fullname.lower().replace(' ', '_')
            base = username
            i = 1
            while True:
                cur.execute("SELECT id_admin FROM tb_admin WHERE username = %s", (username,))
                if not cur.fetchone():
                    break
                username = f"{base}{i}"
                i += 1

            cur.execute("""
                INSERT INTO tb_admin (username, email, google_id, foto_profile, created_via, is_active, last_login)
                VALUES (%s, %s, %s, %s, 'google', 1, %s)
            """, (username, email, google_id, foto, datetime.now()))
            db.commit()
            new_id = cur.lastrowid
            session["admin_logged_in"] = True
            session["admin_id"] = new_id
            session["admin_username"] = username
            session["admin_email"] = email
            session["admin_foto"] = foto   # simpan foto
            flash(f"Selamat datang, {username}!", "success")

        cur.close()
        db.close()
        return redirect(url_for('dashboard'))

    except Exception as e:
        print(f"Admin Google Login Error: {e}")
        flash(f'Terjadi kesalahan: {e}', 'danger')
        return redirect(url_for('admin_login'))

@app.route("/admin/logout")
def admin_logout():
    # Hapus semua session admin
    session.pop("admin_logged_in", None)
    session.pop("admin_id", None)
    session.pop("admin_username", None)
    session.pop("admin_email", None)
    session.pop("admin_foto", None)
    flash("Anda telah logout!", "success")
    # Pastikan endpoint 'admin_login' benar-benar ada
    return redirect(url_for('admin_login'))

# ============================================================
#  ADMIN DASHBOARD
# ============================================================
@app.route("/admin")
@app.route("/admin/dashboard")
@admin_required
def dashboard():
    db  = get_db()
    cur = db.cursor(dictionary=True)

    cur.execute("SELECT COUNT(*) AS total FROM tb_riwayat")
    total = cur.fetchone()["total"]

    cur.execute("SELECT COUNT(*) AS benar FROM tb_riwayat WHERE status_akurasi = 'Benar'")
    benar = cur.fetchone()["benar"]

    cur.execute("SELECT COUNT(*) AS salah FROM tb_riwayat WHERE status_akurasi = 'Salah'")
    salah = cur.fetchone()["salah"]

    cur.execute("SELECT COUNT(*) AS belum FROM tb_riwayat WHERE status_akurasi = 'Belum Diverifikasi'")
    belum = cur.fetchone()["belum"]

    terverifikasi = benar + salah
    akurasi = round((benar / terverifikasi * 100), 1) if terverifikasi > 0 else 0

    # Statistik global (pengganti v_statistik_global)
    cur.execute("SELECT COUNT(DISTINCT user_id) AS total_user FROM tb_riwayat")
    total_user = cur.fetchone()["total_user"]

    cur.execute("SELECT COUNT(*) AS total_dipublik FROM tb_riwayat WHERE is_public = 1")
    total_dipublik = cur.fetchone()["total_dipublik"]

    cur.execute("SELECT label_ai, COUNT(*) AS jumlah FROM tb_riwayat GROUP BY label_ai ORDER BY jumlah DESC")
    dist_label = cur.fetchall()

    cur.execute("""
        SELECT r.*, g.nama_umum, u.nama as user_nama
        FROM tb_riwayat r
        LEFT JOIN tb_gulma g ON r.label_ai = g.label_model
        LEFT JOIN tb_user u ON r.user_id = u.id_user
        ORDER BY r.tgl_deteksi DESC LIMIT 10
    """)
    recent = cur.fetchall()

    # Akurasi per kelas (pengganti v_akurasi_per_kelas)
    cur.execute("""
        SELECT r.label_ai, g.nama_umum,
               COUNT(*) AS total_deteksi,
               SUM(CASE WHEN r.status_akurasi = 'Benar' THEN 1 ELSE 0 END) AS jumlah_benar,
               SUM(CASE WHEN r.status_akurasi = 'Salah' THEN 1 ELSE 0 END) AS jumlah_salah,
               SUM(CASE WHEN r.status_akurasi = 'Belum Diverifikasi' THEN 1 ELSE 0 END) AS belum_verif,
               ROUND(AVG(r.confidence_score) * 100, 1) AS rata_confidence
        FROM tb_riwayat r
        LEFT JOIN tb_gulma g ON r.label_ai = g.label_model
        GROUP BY r.label_ai, g.nama_umum
    """)
    akurasi_per_kelas = cur.fetchall()
    for k in akurasi_per_kelas:
        verif = (k["jumlah_benar"] or 0) + (k["jumlah_salah"] or 0)
        k["akurasi_persen"] = round(((k["jumlah_benar"] or 0) / verif * 100), 1) if verif > 0 else 0

    per_kelas = []
    for k in cur.fetchall() if False else akurasi_per_kelas:
        verif = (k["jumlah_benar"] or 0) + (k["jumlah_salah"] or 0)
        pct   = round(((k["jumlah_benar"] or 0) / verif * 100), 1) if verif > 0 else 0
        per_kelas.append({
            "label_ai":     k["label_ai"],
            "total_verif":  verif,
            "benar":        k["jumlah_benar"] or 0,
            "akurasi_kelas":pct
        })

    cur.close(); db.close()
    return render_template("admin/dashboard.html",
        total=total, benar=benar, salah=salah, belum=belum,
        akurasi=akurasi, dist_label=dist_label, recent=recent,
        per_kelas=per_kelas, total_user=total_user,
        total_dipublik=total_dipublik)
# ============================================================
#  ADMIN RIWAYAT
# ============================================================
@app.route("/admin/riwayat")
@admin_required
def admin_riwayat():
    db = get_db()
    cur = db.cursor(dictionary=True)
    cur.execute("""
        SELECT r.*, g.nama_umum, u.nama as user_nama
        FROM tb_riwayat r
        LEFT JOIN tb_gulma g ON r.label_ai = g.label_model
        LEFT JOIN tb_user u ON r.user_id = u.id_user
        ORDER BY r.tgl_deteksi DESC
    """)
    rows = cur.fetchall()
    cur.close()

    # Gunakan koneksi terpisah untuk ambil semua_gulma
    db2 = get_db()
    cur2 = db2.cursor(dictionary=True)
    cur2.execute("SELECT label_model, nama_umum FROM tb_gulma ORDER BY nama_umum")
    semua_gulma = cur2.fetchall()
    cur2.close()
    db2.close()

    # Proses sesi_dict, hitung statistik
    sesi_dict = {}
    for r in rows:
        key = r["id_sesi_foto"] or str(r["id_riwayat"])
        if key not in sesi_dict:
            sesi_dict[key] = {
                "id_sesi_foto": key,
                "foto_upload": r["foto_upload"],
                "foto_annotasi": r.get("foto_annotasi"),
                "tgl_deteksi": r["tgl_deteksi"],
                "user_nama": r.get("user_nama"),
                "deteksi": [],
                "total": 0,
                "benar": 0,
                "salah": 0,
                "belum": 0
            }
        # is_not_weed bisa dari label_asli atau status_akurasi
        is_not_weed = (r.get("label_asli") == "Bukan Gulma") or (r.get("status_akurasi") == "Bukan Gulma")
        sesi_dict[key]["deteksi"].append({
            "id_riwayat": r["id_riwayat"],
            "label_ai": r["label_ai"],
            "nama_umum": r.get("nama_umum"),
            "confidence_score": r["confidence_score"],
            "status_akurasi": r["status_akurasi"],
            "is_not_weed": is_not_weed,
            "label_asli": r.get("label_asli", "")
        })
        if not is_not_weed:
            sesi_dict[key]["total"] += 1
            if r["status_akurasi"] == "Benar":
                sesi_dict[key]["benar"] += 1
            elif r["status_akurasi"] == "Salah":
                sesi_dict[key]["salah"] += 1
            else:
                sesi_dict[key]["belum"] += 1

    data = []
    total_objek = 0
    benar_objek = 0
    salah_objek = 0
    belum_objek = 0
    bukan_gulma_objek = 0

    for key, sesi in sesi_dict.items():
        total_s = sesi["total"]
        benar_s = sesi["benar"]
        salah_s = sesi["salah"]
        belum_s = sesi["belum"]
        terverif = benar_s + salah_s
        akurasi_s = round((benar_s / terverif * 100), 1) if terverif > 0 else None
        sesi.update({"akurasi": akurasi_s})
        data.append(sesi)

        total_objek += total_s
        benar_objek += benar_s
        salah_objek += salah_s
        belum_objek += belum_s
        # Hitung bukan gulma dari deteksi
        for d in sesi["deteksi"]:
            if d["is_not_weed"]:
                bukan_gulma_objek += 1

    akurasi = round((benar_objek / (benar_objek + salah_objek) * 100), 1) if (benar_objek + salah_objek) > 0 else 0

    db.close()
    return render_template("admin/riwayat.html",
        data=data,
        total_objek=total_objek,
        benar_objek=benar_objek,
        salah_objek=salah_objek,
        belum_objek=belum_objek,
        bukan_gulma_objek=bukan_gulma_objek,
        akurasi=akurasi,
        semua_gulma=semua_gulma)
    
# ============================================================
#  ADMIN KONFIRMASI (UPDATE STATUS VERIFIKASI)
# ============================================================
@app.route("/admin/konfirmasi/<int:id_riwayat>", methods=["POST"])
@admin_required
def admin_konfirmasi(id_riwayat):
    """
    Endpoint untuk admin mengubah status verifikasi suatu riwayat deteksi.
    Menerima parameter:
        - label_asli: (string) nama gulma yang benar (jika salah), kosong jika benar.
        - is_not_weed: (string 'true'/'false') menandakan bahwa objek bukan gulma.
    """
    label_asli = request.form.get("label_asli", "").strip()
    is_not_weed = request.form.get("is_not_weed", "false") == "true"

    db = get_db()
    cur = db.cursor()

    try:
        if is_not_weed:
            # Kasus: Bukan Gulma -> status dianggap Salah, label_asli = 'Bukan Gulma'
            cur.execute("""
                UPDATE tb_riwayat
                SET status_akurasi = 'Salah', label_asli = 'Bukan Gulma', id_admin_verif = %s
                WHERE id_riwayat = %s
            """, (session.get("admin_id"), id_riwayat))
        elif label_asli:
            # Kasus: Salah -> update status dan label_asli
            cur.execute("""
                UPDATE tb_riwayat
                SET status_akurasi = 'Salah', label_asli = %s, id_admin_verif = %s
                WHERE id_riwayat = %s
            """, (label_asli, session.get("admin_id"), id_riwayat))
        else:
            # Kasus: Benar -> update status, kosongkan label_asli
            cur.execute("""
                UPDATE tb_riwayat
                SET status_akurasi = 'Benar', label_asli = NULL, id_admin_verif = %s
                WHERE id_riwayat = %s
            """, (session.get("admin_id"), id_riwayat))

        db.commit()
        cur.close()
        db.close()
        return jsonify({"status": "success", "message": "Data berhasil diperbarui."})

    except Exception as e:
        db.rollback()
        cur.close()
        db.close()
        print(f"Error in admin_konfirmasi: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/admin/komentar")
@admin_required
def admin_komentar():
    page = request.args.get('page', 1, type=int)
    per_page = 8
    offset = (page - 1) * per_page

    db = get_db()
    cur = db.cursor(dictionary=True)

    # Hitung total komentar
    cur.execute("SELECT COUNT(*) as total FROM tb_komentar WHERE is_deleted = 0")
    total = cur.fetchone()['total']
    total_pages = ceil(total / per_page) if total > 0 else 1

    # Ambil komentar dengan pagination dan JOIN
    cur.execute("""
        SELECT k.id_komentar, k.isi_komentar, k.tgl_komentar, k.foto_komentar,
               u.nama as user_nama, u.foto_profile as user_foto,
               f.judul as forum_judul, f.id_forum
        FROM tb_komentar k
        JOIN tb_user u ON k.user_id = u.id_user
        JOIN tb_forum f ON k.id_forum = f.id_forum
        WHERE k.is_deleted = 0
        ORDER BY k.tgl_komentar DESC
        LIMIT %s OFFSET %s
    """, (per_page, offset))
    komentar = cur.fetchall()
    cur.close()
    db.close()

    return render_template("admin/komentar.html", 
                           komentar=komentar, 
                           page=page, 
                           total_pages=total_pages)
# ============================================================
#  ADMIN HAPUS KOMENTAR
# ============================================================
@app.route("/admin/komentar/hapus/<int:id_komentar>", methods=["POST"])
@admin_required
def admin_hapus_komentar(id_komentar):
    db  = get_db()
    cur = db.cursor()
    cur.execute("""
        UPDATE tb_komentar
        SET is_deleted = 1, id_admin_del = %s, tgl_dihapus = NOW()
        WHERE id_komentar = %s
    """, (session.get("admin_id"), id_komentar))
    db.commit()
    cur.close(); db.close()
    flash("Komentar berhasil dihapus.", "success")
    return redirect(request.referrer or url_for("forum"))

# ============================================================
#  ADMIN CRUD GULMA
# ============================================================
@app.route("/admin/gulma")
@admin_required
def admin_gulma():
    db  = get_db()
    cur = db.cursor(dictionary=True)
    cur.execute("SELECT * FROM tb_gulma ORDER BY id_gulma")
    data = cur.fetchall()
    for g in data:
        cur.execute("SELECT nama_file FROM tb_foto_gulma WHERE id_gulma=%s ORDER BY urutan", (g["id_gulma"],))
        g["foto_list"] = [r["nama_file"] for r in cur.fetchall()]
        if not g["foto_list"] and g.get("foto_referensi"):
            g["foto_list"] = [g["foto_referensi"]]
    cur.close(); db.close()
    return render_template("admin/gulma.html", data=data)

def _simpan_foto_gulma(cur, id_gulma, request_files, mulai_urutan=1):
    foto_utama = None
    urutan     = mulai_urutan
    for i in range(5):
        key = f"foto_baru_{i}"
        if key in request_files:
            f = request_files[key]
            if f and f.filename and allowed_file(f.filename):
                fname = f"ref_{uuid.uuid4().hex}_{secure_filename(f.filename)}"
                f.save(os.path.join(UPLOAD_FOLDER, fname))
                cur.execute(
                    "INSERT INTO tb_foto_gulma (id_gulma, nama_file, urutan) VALUES (%s,%s,%s)",
                    (id_gulma, fname, urutan)
                )
                if foto_utama is None:
                    foto_utama = fname
                urutan += 1
    return foto_utama

@app.route("/admin/gulma/tambah", methods=["GET", "POST"])
@admin_required
def admin_gulma_tambah():
    if request.method == "POST":
        f   = request.form
        db  = get_db()
        cur = db.cursor()
        cur.execute("""
            INSERT INTO tb_gulma (label_model, nama_umum, nama_latin, deskripsi_umum,
            bentuk_daun, warna_daun, karakter_batang, karakter_akar,
            cara_berkembangbiak, waktu_kendali_terbaik, metode_mekanik, metode_kimiawi)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (f["label_model"], f.get("nama_umum",""), f.get("nama_latin",""),
              f.get("deskripsi_umum",""), f.get("bentuk_daun",""), f.get("warna_daun",""),
              f.get("karakter_batang",""), f.get("karakter_akar",""),
              f.get("cara_berkembangbiak",""), f.get("waktu_kendali_terbaik",""),
              f.get("metode_mekanik",""), f.get("metode_kimiawi","")))
        id_gulma   = cur.lastrowid
        foto_utama = _simpan_foto_gulma(cur, id_gulma, request.files)
        if foto_utama:
            cur.execute("UPDATE tb_gulma SET foto_referensi=%s WHERE id_gulma=%s", (foto_utama, id_gulma))
        db.commit(); cur.close(); db.close()
        flash("Data gulma berhasil ditambahkan!", "success")
        return redirect(url_for("admin_gulma"))
    return render_template("admin/gulma_form.html", action="tambah", data={}, foto_existing=[])

@app.route("/admin/gulma/edit/<int:id_gulma>", methods=["GET", "POST"])
@admin_required
def admin_gulma_edit(id_gulma):
    db  = get_db()
    cur = db.cursor(dictionary=True)
    if request.method == "POST":
        f        = request.form
        hapus_ids= request.form.getlist("hapus_foto")
        for hid in hapus_ids:
            cur.execute("SELECT nama_file FROM tb_foto_gulma WHERE id_foto=%s", (hid,))
            row = cur.fetchone()
            if row:
                try: os.remove(os.path.join(UPLOAD_FOLDER, row["nama_file"]))
                except: pass
            cur.execute("DELETE FROM tb_foto_gulma WHERE id_foto=%s", (hid,))
        cur.execute("SELECT COUNT(*) AS c FROM tb_foto_gulma WHERE id_gulma=%s", (id_gulma,))
        existing_count = cur.fetchone()["c"]
        foto_utama = _simpan_foto_gulma(cur, id_gulma, request.files, mulai_urutan=existing_count+1)
        cur.execute("""
            UPDATE tb_gulma SET label_model=%s, nama_umum=%s, nama_latin=%s, deskripsi_umum=%s,
            bentuk_daun=%s, warna_daun=%s, karakter_batang=%s, karakter_akar=%s,
            cara_berkembangbiak=%s, waktu_kendali_terbaik=%s, metode_mekanik=%s, metode_kimiawi=%s
            WHERE id_gulma=%s
        """, (f["label_model"], f.get("nama_umum",""), f.get("nama_latin",""),
              f.get("deskripsi_umum",""), f.get("bentuk_daun",""), f.get("warna_daun",""),
              f.get("karakter_batang",""), f.get("karakter_akar",""),
              f.get("cara_berkembangbiak",""), f.get("waktu_kendali_terbaik",""),
              f.get("metode_mekanik",""), f.get("metode_kimiawi",""), id_gulma))
        cur.execute("SELECT nama_file FROM tb_foto_gulma WHERE id_gulma=%s ORDER BY urutan LIMIT 1", (id_gulma,))
        first = cur.fetchone()
        if first:
            cur.execute("UPDATE tb_gulma SET foto_referensi=%s WHERE id_gulma=%s", (first["nama_file"], id_gulma))
        db.commit(); cur.close(); db.close()
        flash("Data gulma berhasil diperbarui!", "success")
        return redirect(url_for("admin_gulma"))
    cur.execute("SELECT * FROM tb_gulma WHERE id_gulma = %s", (id_gulma,))
    data = cur.fetchone()
    cur.execute("SELECT * FROM tb_foto_gulma WHERE id_gulma=%s ORDER BY urutan LIMIT 5", (id_gulma,))
    foto_existing = cur.fetchall()
    cur.close(); db.close()
    if not data:
        return "Data tidak ditemukan", 404
    return render_template("admin/gulma_form.html", action="edit", data=data, foto_existing=foto_existing)

@app.route("/admin/gulma/hapus/<int:id_gulma>", methods=["POST"])
@admin_required
def admin_gulma_hapus(id_gulma):
    db  = get_db()
    cur = db.cursor()
    cur.execute("DELETE FROM tb_gulma WHERE id_gulma = %s", (id_gulma,))
    db.commit(); cur.close(); db.close()
    flash("Data gulma berhasil dihapus!", "success")
    return redirect(url_for("admin_gulma"))

@app.route("/post_detail/<int:id_forum>")
def post_detail(id_forum):
    return redirect(url_for('diskusi_detail', id_forum=id_forum))

# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    print("="*55)
    print(f"  GULMAIFY — Running on port {port}")
    print("="*55)

    app.run(host="0.0.0.0", port=port)