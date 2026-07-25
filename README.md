# MP3 → MIDI (offline, CUDA)

Aplikasi lokal berbasis browser untuk mengubah MP3 menjadi file MIDI polifonik.
Inference berjalan di GPU NVIDIA lewat ONNX Runtime CUDA. Tidak ada satu byte pun
audio yang meninggalkan komputermu.

Ditargetkan untuk: **RTX 3060 12GB, driver 610.74, Windows/Linux, Python 3.10–3.12**.

---

## 1. Yang perlu disiapkan

| Kebutuhan | Keterangan |
|---|---|
| Python 3.10–3.12 | 3.13 kadang belum punya wheel `onnxruntime-gpu` |
| ffmpeg | sudah terpasang di PATH (`ffmpeg -version` harus jalan) |
| CUDA 12.x + cuDNN 9.x | driver 610.74 sudah lebih dari cukup |
| `models/nmp.onnx` | model Basic Pitch, ~20 MB, diunduh sekali |

## 2. Install

```bash
cd mp3-to-midi
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## 3. Ambil model (sekali saja, butuh internet)

Model yang dipakai adalah **Basic Pitch ICASSP-2022** dari Spotify (Apache-2.0),
file `nmp.onnx`. Letakkan di `models/nmp.onnx`.

Cara paling gampang — ambil dari paket resminya:

```bash
pip download basic-pitch --no-deps -d /tmp/bp
# ekstrak arsip yang terunduh, lalu copy:
#   basic_pitch/saved_models/icassp_2022/nmp.onnx  ->  models/nmp.onnx
```

Atau langsung dari repositori GitHub `spotify/basic-pitch`, di path
`basic_pitch/saved_models/icassp_2022/nmp.onnx`.

> Catatan jujur: nama folder di dalam paket pernah berubah antar versi. Kalau
> `nmp.onnx` tidak ada di path itu, cari saja dengan `find . -name "*.onnx"` di
> hasil ekstraksi. Yang dibutuhkan adalah file ONNX dengan input berbentuk
> `[batch, 43844, 1]`. `python run.py --check` akan memverifikasi bentuk ini dan
> menolak file yang salah, jadi kamu tidak akan diam-diam pakai model keliru.

Setelah ini aplikasi **100% offline**. Matikan WiFi kalau mau membuktikannya.

## 4. Cek dulu sebelum dipakai

```bash
python run.py --check
```

Contoh keluaran yang benar:

```
onnxruntime       1.20.1
providers         CUDAExecutionProvider, CPUExecutionProvider
gpu               NVIDIA GeForce RTX 3060  |  VRAM 12288 MB
driver            610.74
cuda provider     OK
ffmpeg            OK
model file        OK  models/nmp.onnx (19.8 MB)

--- live inference test ---
active provider   CUDAExecutionProvider
running on        GPU: NVIDIA GeForce RTX 3060  OK
```

Baris `active provider` itu intinya. Itu dibaca dari sesi ONNX yang sudah jadi,
bukan dari daftar provider yang "tersedia" — jadi tidak bisa bohong.

## 5. Jalankan

```bash
python run.py
```

Browser terbuka otomatis di `http://127.0.0.1:7860`. Kalau port dipakai, aplikasi
mencari port bebas berikutnya sendiri.

Opsi lain:

```bash
python run.py --cpu           # paksa CPU (buat perbandingan kecepatan)
python run.py --port 9000
python run.py --no-browser
python run.py --model D:\model\nmp.onnx
```

## 6. Cara pakai

1. Drag & drop file `.mp3` (atau klik area drop).
2. Opsional: buka **Pengaturan lanjutan** untuk mengatur sensitivitas.
3. Klik **Convert & Download**.
4. Progress bar bergerak 0 → 100%. Badge kanan atas menunjukkan GPU/CPU yang
   sedang dipakai.
5. File `.mid` terunduh otomatis begitu selesai.

### Arti pembagian progress

| Rentang | Yang terjadi | Di mana |
|---|---|---|
| 0–10% | dekode MP3 + windowing | CPU |
| 10–85% | inference neural net (progress nyata per batch) | **GPU** |
| 85–97% | pembentukan not dari posteriorgram | CPU |
| 97–100% | penulisan file MIDI | CPU |

Segmen 85–100% bisa terasa "menggantung" beberapa detik pada lagu panjang. Itu
normal: post-processing not adalah loop NumPy di CPU, bukan hang.

## 7. Pengaturan lanjutan

| Kontrol | Naikkan bila | Turunkan bila |
|---|---|---|
| Sensitivitas onset | banyak not palsu / dobel | banyak not terlewat |
| Ambang sustain | not kedengaran terlalu panjang | not terpotong-potong |
| Not terpendek | banyak not sangat pendek (noise) | trill/nada cepat hilang |
| Instrumen track MIDI | — | hanya label General MIDI untuk DAW |

Untuk stem gitar/vokal yang padat, biasanya onset 0.4–0.5 dan sustain 0.25–0.3
sudah bagus. Untuk bass, naikkan "Not terpendek" ke 200–300 ms.

## 8. Struktur project

```
mp3-to-midi/
├── run.py                entrypoint + diagnostik (--check)
├── requirements.txt
├── models/nmp.onnx       model Basic Pitch (kamu unduh sendiri)
├── app/
│   ├── config.py         konstanta model + default aplikasi
│   ├── device.py         deteksi GPU / provider
│   ├── audio.py          dekode ffmpeg + windowing
│   ├── transcribe.py     sesi ONNX, batching, unwrap output
│   ├── notes.py          posteriorgram -> note events
│   ├── midi_writer.py    penulis SMF tanpa dependensi
│   ├── jobs.py           job store + worker tunggal
│   └── server.py         FastAPI + SSE progress
├── static/               UI (HTML/CSS/JS, tanpa build step)
├── tools/selftest.py     tes offline tanpa GPU
└── tmp/                  upload & hasil, auto-bersih 1 jam
```

## 9. Tes

```bash
python tools/selftest.py
```

Menguji matematika windowing, koreksi drift frame→waktu, pembentukan not pada
posteriorgram sintetis, dan struktur byte file MIDI (dibaca ulang dengan parser
kecil). 56 assertion, tidak butuh GPU maupun file model.

## 10. Troubleshooting CUDA

**`cuda provider FAIL` atau `active provider CPUExecutionProvider`**

Urutan yang paling sering menyelesaikan masalah:

1. Pastikan yang terinstall `onnxruntime-gpu`, bukan `onnxruntime`. Kalau keduanya
   ada, keduanya bertabrakan:
   ```bash
   pip uninstall -y onnxruntime onnxruntime-gpu
   pip install onnxruntime-gpu==1.20.1
   ```
2. Install runtime CUDA/cuDNN lewat pip supaya tidak bergantung instalasi sistem:
   ```bash
   pip install nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cublas-cu12
   ```
3. Kalau masih gagal, coba turunkan satu-dua rilis (`1.19.2`, `1.18.1`). Pasangan
   cuDNN yang dituntut tiap build ORT bergeser antar versi.
4. Jalankan `python run.py --check` setelah setiap percobaan.

**Hasil MIDI kosong (0 not)** — turunkan sensitivitas onset ke 0.3, lalu ambang
sustain ke 0.2. Kalau audionya sangat pelan, normalisasi dulu di DAW.

**Timing MIDI melenceng makin jauh di akhir lagu** — laporkan; koreksi drift ada
di `notes.model_frames_to_time()` dan diuji di selftest, tapi kalau file
modelnya berbeda versi, geometri framenya bisa lain.

**Not hasilnya berantakan/acak sejak awal** — kemungkinan urutan output model
salah dikenali. Paksa manual:
```bash
# Linux/macOS
export BP_OUTPUT_ORDER=note,onset,contour
# Windows PowerShell
$env:BP_OUTPUT_ORDER="note,onset,contour"
```

## 11. Batasan yang perlu diketahui

* **Bukan pemisah instrumen.** Output adalah satu track MIDI. Konversi tiap stem
  satu per satu — itu memang alur yang direncanakan.
* **Drum/perkusi tidak cocok.** Model ini pendeteksi pitch; suara tanpa pitch
  akan jadi not acak. Jangan konversi stem drum.
* **Tanpa pitch bend.** Glissando dan bending gitar dibulatkan ke semitone
  terdekat.
* **Tanpa deteksi tempo/kunci.** Output selalu 120 BPM tanpa kuantisasi grid.
  Atur tempo di DAW setelah impor.
* **CPU tetap terpakai.** Dekode MP3, post-processing, dan penulisan MIDI adalah
  operasi CPU. Yang dijamin GPU-only adalah inference-nya, dan thread pool CPU
  ORT sengaja dibatasi 1 saat CUDA aktif.
* **Durasi maks 15 menit, ukuran maks 100 MB.** Bisa diubah lewat env
  `BP_MAX_DURATION_MIN` dan `BP_MAX_UPLOAD_MB`.

## 12. Variabel lingkungan

| Variabel | Default | Fungsi |
|---|---|---|
| `BP_MODEL_PATH` | `models/nmp.onnx` | lokasi model |
| `BP_BATCH_SIZE` | `32` | window per batch inference |
| `BP_FORCE_CPU` | – | `1` untuk memaksa CPU |
| `BP_OUTPUT_ORDER` | – | override urutan output model |
| `BP_MAX_UPLOAD_MB` | `100` | batas ukuran unggahan |
| `BP_MAX_DURATION_MIN` | `15` | batas durasi audio |
| `BP_TMP_DIR` | `tmp/` | folder kerja |
| `FFMPEG_BINARY` | – | path ffmpeg manual |

## 13. Lisensi & atribusi

Algoritma pembentukan not di `app/notes.py` adalah port dari
[Basic Pitch](https://github.com/spotify/basic-pitch) (Spotify, Apache-2.0).
Bobot model `nmp.onnx` juga berasal dari project tersebut dan mengikuti
lisensinya.

---

⚠️ **Catatan Edukasi:** Kode dan penjelasan di atas dibuat untuk tujuan
pembelajaran. Uji hanya di lingkungan/sistem milikmu sendiri, dan pastikan
penggunaannya sesuai hukum yang berlaku.
