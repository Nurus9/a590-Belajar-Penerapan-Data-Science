from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_URL = "https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/main/students_performance/data.csv"
ARTIFACT_PATH = Path(__file__).resolve().parent / "model" / "student_performance_pipeline.joblib"
FEATURE_COLUMNS = [
	"Marital_status",
	"Application_mode",
	"Course",
	"Daytime_evening_attendance",
	"Previous_qualification_grade",
	"Admission_grade",
	"Debtor",
	"Tuition_fees_up_to_date",
	"Gender",
	"Scholarship_holder",
	"Age_at_enrollment",
	"Curricular_units_1st_sem_approved",
	"Curricular_units_1st_sem_grade",
	"Curricular_units_2nd_sem_approved",
	"Curricular_units_2nd_sem_grade",
]
CATEGORICAL_COLUMNS = [
	"Marital_status",
	"Application_mode",
	"Course",
	"Daytime_evening_attendance",
	"Debtor",
	"Tuition_fees_up_to_date",
	"Gender",
	"Scholarship_holder",
]
NUMERICAL_COLUMNS = [
	"Previous_qualification_grade",
	"Admission_grade",
	"Age_at_enrollment",
	"Curricular_units_1st_sem_approved",
	"Curricular_units_1st_sem_grade",
	"Curricular_units_2nd_sem_approved",
	"Curricular_units_2nd_sem_grade",
]
TARGET_STATUSES = ["Dropout", "Graduate"]

# ── Human-readable label mappings ────────────────────────────────────────────
# Untuk memudahkan pengguna dalam memilih opsi, kita buat mapping dari nilai mentah ke label yang lebih deskriptif.
LABEL_MAPS = {
	"Marital_status": {
		1: "Lajang",
		2: "Menikah",
		3: "Duda/Janda",
		4: "Cerai",
		5: "Persatuan Faktual",
		6: "Pisah Secara Hukum",
	},
	"Daytime_evening_attendance": {
		0: "Malam",
		1: "Siang",
	},
	"Debtor": {
		0: "Bukan Debitur",
		1: "Debitur",
	},
	"Tuition_fees_up_to_date": {
		0: "Belum Lunas",
		1: "Lunas",
	},
	"Gender": {
		0: "Perempuan",
		1: "Laki-laki",
	},
	"Scholarship_holder": {
		0: "Tidak Ada Beasiswa",
		1: "Penerima Beasiswa",
	},
}

# fungsi untuk membuat opsi selectbox berdasarkan kolom dan mapping label
def make_options(df: pd.DataFrame, col: str) -> dict:
	"""Return {label: raw_value} dict for a selectbox."""
	mapping = LABEL_MAPS.get(col, {})
	raw_values = sorted(df[col].unique().tolist())
	return {mapping.get(v, str(v)): v for v in raw_values}

# fungsi untuk memuat data dengan caching agar tidak perlu reload setiap interaksi
@st.cache_data
def load_data() -> pd.DataFrame:
	return pd.read_csv(DATA_URL, sep=";")


def prepare_training_data(df: pd.DataFrame) -> pd.DataFrame:
	train_df = df[df["Status"].isin(TARGET_STATUSES)].copy()
	train_df["is_dropout"] = (train_df["Status"] == "Dropout").astype(int)
	return train_df

# fungsi untuk melatih model pipeline dan menghitung metrik evaluasi
def train_pipeline(train_df: pd.DataFrame) -> dict:
	x = train_df[FEATURE_COLUMNS]
	y = train_df["is_dropout"]

	x_train, x_test, y_train, y_test = train_test_split(
		x,
		y,
		test_size=0.2,
		random_state=42,
		stratify=y,
	)

	preprocess = ColumnTransformer(
		transformers=[
			(
				"cat",
				OneHotEncoder(handle_unknown="ignore"),
				CATEGORICAL_COLUMNS,
			)
		],
		remainder="passthrough",
	)

	pipeline = Pipeline(
		steps=[
			("preprocess", preprocess),
			("model", LogisticRegression(max_iter=3000, class_weight="balanced")),
		]
	)

	pipeline.fit(x_train, y_train)

	y_pred = pipeline.predict(x_test)
	y_prob = pipeline.predict_proba(x_test)[:, 1]

	metrics = {
		"accuracy": float(accuracy_score(y_test, y_pred)),
		"precision": float(precision_score(y_test, y_pred, zero_division=0)),
		"recall": float(recall_score(y_test, y_pred, zero_division=0)),
		"f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
		"roc_auc": float(roc_auc_score(y_test, y_prob)),
	}

	return {
		"pipeline": pipeline,
		"metrics": metrics,
		"sklearn_version": sklearn.__version__,
	}

# Fungsi untuk mendapatkan model pipeline dari cache atau melatih ulang jika tidak ada atau versi sklearn berubah
@st.cache_resource
def get_model_artifact(train_df: pd.DataFrame) -> dict:
	if ARTIFACT_PATH.exists():
		try:
			artifact = joblib.load(ARTIFACT_PATH)
			if (
				artifact.get("sklearn_version") == sklearn.__version__
				and artifact.get("target_statuses") == TARGET_STATUSES
			):
				return artifact
		except Exception:
			pass

	artifact = train_pipeline(train_df)
	artifact["target_statuses"] = TARGET_STATUSES
	ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(artifact, ARTIFACT_PATH)
	return artifact


# ── Rekomendasi bertingkat ────────────────────────────────────────────────────
# Fungsi ini memberikan rekomendasi tindakan berdasarkan tingkat risiko dropout yang diprediksi, serta faktor-faktor kritis seperti status debitur dan administrasi biaya kuliah.
def recommendation_text(probability: float, debtor: int, tuition_up_to_date: int) -> tuple[str, str]:
	"""
	Returns (level_label, recommendation_string).
	Level: rendah / sedang / tinggi / kritis
	"""
	recs = []

	if probability < 0.3:
		level = "🟢 Risiko Rendah"
		recs.append("Pertahankan monitoring rutin setiap semester.")
		recs.append("Dorong mahasiswa untuk aktif di kegiatan akademik dan organisasi kampus.")

	elif probability < 0.5:
		level = "🟡 Risiko Sedang"
		recs.append("Jadwalkan sesi konseling akademik minimal satu kali dalam semester berjalan.")
		recs.append("Pantau perkembangan nilai UTS dan UAS secara berkala.")
		if debtor == 1:
			recs.append("Koordinasikan dengan bagian keuangan mengenai opsi cicilan atau keringanan biaya.")
		if tuition_up_to_date == 0:
			recs.append("Ingatkan mahasiswa untuk segera menyelesaikan administrasi biaya kuliah.")

	elif probability < 0.75:
		level = "🟠 Risiko Tinggi"
		recs.append("Prioritaskan pendampingan akademik intensif — minimal dua kali pertemuan dengan dosen wali semester ini.")
		recs.append("Evaluasi beban SKS; pertimbangkan pengurangan jika mahasiswa kewalahan.")
		if debtor == 1:
			recs.append("Segera lakukan konseling finansial; status debitur secara signifikan meningkatkan risiko dropout.")
		if tuition_up_to_date == 0:
			recs.append("Lakukan follow-up administrasi biaya kuliah — hambatan finansial harus diselesaikan sebelum UAS.")

	else:
		level = "🔴 Risiko Kritis"
		recs.append("Intervensi segera diperlukan. Hubungi mahasiswa dan orang tua/wali dalam minggu ini.")
		recs.append("Susun rencana studi remedial bersama dosen wali dan bagian akademik.")
		recs.append("Tawarkan opsi cuti akademik sementara jika ada hambatan berat di luar akademik.")
		if debtor == 1:
			recs.append("Debitur dengan probabilitas sangat tinggi: koordinasikan dengan beasiswa darurat atau bantuan dana kampus.")
		if tuition_up_to_date == 0:
			recs.append("Biaya kuliah belum lunas pada level risiko ini dapat menjadi pemicu langsung dropout — selesaikan segera.")

	return level, " ".join(recs)


# ── Gauge chart ───────────────────────────────────────────────────────────────
# Fungsi ini membuat gauge chart menggunakan Plotly untuk memvisualisasikan probabilitas dropout dengan warna yang berubah sesuai tingkat risiko.
def dropout_gauge(probability: float) -> go.Figure:
	pct = probability * 100
	if probability < 0.3:
		bar_color = "#22c55e"   # hijau
	elif probability < 0.5:
		bar_color = "#eab308"   # kuning
	elif probability < 0.75:
		bar_color = "#f97316"   # oranye
	else:
		bar_color = "#ef4444"   # merah

	fig = go.Figure(go.Indicator(
		mode="gauge+number",
		value=pct,
		number={"suffix": "%", "font": {"size": 36}},
		gauge={
			"axis": {"range": [0, 100], "tickwidth": 1},
			"bar": {"color": bar_color, "thickness": 0.3},
			"steps": [
				{"range": [0, 30],  "color": "#dcfce7"},
				{"range": [30, 50], "color": "#fef9c3"},
				{"range": [50, 75], "color": "#ffedd5"},
				{"range": [75, 100],"color": "#fee2e2"},
			],
			"threshold": {
				"line": {"color": "black", "width": 2},
				"thickness": 0.75,
				"value": pct,
			},
		},
	))
	fig.update_layout(
		height=260,
		margin=dict(t=20, b=0, l=30, r=30),
		paper_bgcolor="rgba(0,0,0,0)",
		font_color="#444",
	)
	return fig

# ── Distribution chart (Plotly) ───────────────────────────────────────────────
# Fungsi ini membuat bar chart menggunakan Plotly untuk menunjukkan distribusi jumlah mahasiswa yang dropout vs tidak dropout, dengan persentase di atas setiap bar.
def dropout_distribution_chart(df: pd.DataFrame) -> go.Figure:
	counts = df["is_dropout"].value_counts().sort_index()
	labels = ["Tidak Dropout", "Dropout"]
	values = [counts.get(0, 0), counts.get(1, 0)]
	total = sum(values)
	colors = ["#22c55e", "#ef4444"]

	fig = go.Figure(go.Bar(
		x=labels,
		y=values,
		marker_color=colors,
		text=[f"{v:,}<br>({v/total*100:.1f}%)" for v in values],
		textposition="outside",
		width=0.4,
	))
	fig.update_layout(
		height=300,
		margin=dict(t=30, b=0, l=0, r=0),
		paper_bgcolor="rgba(0,0,0,0)",
		plot_bgcolor="rgba(0,0,0,0)",
		yaxis=dict(showgrid=True, gridcolor="#e5e7eb", title="Jumlah Mahasiswa"),
		xaxis=dict(title=""),
		showlegend=False,
	)
	return fig


# ── Main ──────────────────────────────────────────────────────────────────────
# Untuk membuat KPI
def render_overview(raw_df: pd.DataFrame, train_df: pd.DataFrame, metrics: dict) -> None:
	col1, col2, col3, col4, col5 = st.columns(5)
	col1.metric("Jumlah Mahasiswa", f"{len(raw_df):,}")
	col2.metric("Data Latih (Dropout+Graduate)", f"{len(train_df):,}")
	col3.metric("Akurasi Model", f"{metrics['accuracy']:.2f}")
	col4.metric("F1 Score", f"{metrics['f1_score']:.2f}")
	col5.metric("ROC-AUC", f"{metrics['roc_auc']:.2f}")

# Fungsi untuk merender distribusi dropout dengan chart Plotly
def render_distribution(train_df: pd.DataFrame) -> None:
	st.subheader("Distribusi Status Dropout")
	st.caption("Distribusi ini hanya menggunakan data berlabel final: Dropout vs Graduate.")
	st.plotly_chart(dropout_distribution_chart(train_df), use_container_width=True)

# Fungsi untuk merender form prediksi risiko dropout mahasiswa, mengambil input dari pengguna, memprediksi probabilitas dropout, dan memberikan rekomendasi tindakan berdasarkan hasil prediksi.
def render_prediction_form(df: pd.DataFrame, model) -> None:
	st.subheader("Prediksi Risiko Dropout Mahasiswa")

	with st.form("prediction_form"):
		col_a, col_b = st.columns(2)

		with col_a:
			ms_opts  = make_options(df, "Marital_status")
			marital_label = st.selectbox("Status Pernikahan", list(ms_opts))
			marital_status = ms_opts[marital_label]

			am_opts = make_options(df, "Application_mode")
			application_mode = st.selectbox("Mode Pendaftaran", list(am_opts))
			application_mode = am_opts[application_mode]

			course_opts = make_options(df, "Course")
			course = st.selectbox("Program Studi", list(course_opts))
			course = course_opts[course]

			att_opts = make_options(df, "Daytime_evening_attendance")
			att_label = st.selectbox("Waktu Kuliah", list(att_opts))
			attendance = att_opts[att_label]

			deb_opts = make_options(df, "Debtor")
			deb_label = st.selectbox("Status Keuangan", list(deb_opts))
			debtor = deb_opts[deb_label]

			tui_opts = make_options(df, "Tuition_fees_up_to_date")
			tui_label = st.selectbox("Biaya Kuliah", list(tui_opts))
			tuition = tui_opts[tui_label]

			gen_opts = make_options(df, "Gender")
			gen_label = st.selectbox("Jenis Kelamin", list(gen_opts))
			gender = gen_opts[gen_label]

			sch_opts = make_options(df, "Scholarship_holder")
			sch_label = st.selectbox("Beasiswa", list(sch_opts))
			scholarship = sch_opts[sch_label]

		with col_b:
			prev_qual_grade = st.number_input("Nilai Kualifikasi Sebelumnya", min_value=0.0, max_value=200.0, value=130.0)
			admission_grade = st.number_input("Nilai Penerimaan", min_value=0.0, max_value=200.0, value=130.0)
			age = st.number_input("Usia saat Mendaftar", min_value=15, max_value=70, value=20)
			sem1_approved = st.number_input("SKS Lulus Semester 1", min_value=0, max_value=30, value=5)
			sem1_grade = st.number_input("Nilai Rata-rata Semester 1 (0–20)", min_value=0.0, max_value=20.0, value=11.0)
			sem2_approved = st.number_input("SKS Lulus Semester 2", min_value=0, max_value=30, value=5)
			sem2_grade = st.number_input("Nilai Rata-rata Semester 2 (0–20)", min_value=0.0, max_value=20.0, value=11.0)

		submit = st.form_submit_button("🔍 Prediksi Risiko Dropout", use_container_width=True)

	if submit:
		input_df = pd.DataFrame([{
			"Marital_status": marital_status,
			"Application_mode": application_mode,
			"Course": course,
			"Daytime_evening_attendance": attendance,
			"Previous_qualification_grade": prev_qual_grade,
			"Admission_grade": admission_grade,
			"Debtor": debtor,
			"Tuition_fees_up_to_date": tuition,
			"Gender": gender,
			"Scholarship_holder": scholarship,
			"Age_at_enrollment": age,
			"Curricular_units_1st_sem_approved": sem1_approved,
			"Curricular_units_1st_sem_grade": sem1_grade,
			"Curricular_units_2nd_sem_approved": sem2_approved,
			"Curricular_units_2nd_sem_grade": sem2_grade,
		}])

		prob = float(model.predict_proba(input_df)[0, 1])
		level, rec = recommendation_text(prob, int(debtor), int(tuition))

		st.markdown("---")
		gauge_col, info_col = st.columns([1, 1])

		with gauge_col:
			st.markdown(f"#### Hasil: {level}")
			st.plotly_chart(dropout_gauge(prob), use_container_width=True)

		with info_col:
			st.markdown("#### Rekomendasi Tindakan")
			st.info(rec)

# Fungsi utama untuk merender halaman Streamlit, menampilkan metrik evaluasi, distribusi dropout, dan form prediksi risiko dropout mahasiswa.
def main() -> None:
	st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

	st.title("🎓 Student Dropout Monitoring & Prediction")
	st.caption("Prototype machine learning untuk memprediksi probabilitas mahasiswa dropout.")

	raw_df = load_data()
	train_df = prepare_training_data(raw_df)
	artifact = get_model_artifact(train_df)
	model = artifact["pipeline"]
	metrics = artifact["metrics"]

	render_overview(raw_df, train_df, metrics)
	render_distribution(train_df)
	render_prediction_form(raw_df, model)


if __name__ == "__main__":
	main()