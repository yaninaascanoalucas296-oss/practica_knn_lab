import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
HTML_CONTENT = (BASE_DIR / "index.html").read_text(encoding="utf-8")


def build_datasets():
    rng = np.random.default_rng(42)

    n_cli = 80
    sat_cli = np.array([1] * 40 + [0] * 40)
    rng.shuffle(sat_cli)

    valoracion = np.where(sat_cli == 1, rng.integers(4, 6, size=n_cli), rng.integers(1, 4, size=n_cli))
    tiempo_espera = np.where(sat_cli == 1, rng.integers(3, 18, size=n_cli), rng.integers(20, 55, size=n_cli))
    problemas = np.where(sat_cli == 1, rng.choice([0, 0, 0, 1], size=n_cli), rng.integers(1, 5, size=n_cli))

    df_cliente = pd.DataFrame({
        "valoracion": valoracion,
        "tiempo_espera": tiempo_espera,
        "problemas": problemas,
        "objetivo": sat_cli,
    })

    n_emp = 80
    inv_emp = np.array([1] * 40 + [0] * 40)
    rng.shuffle(inv_emp)

    empleados = np.where(inv_emp == 1, rng.integers(15, 65, size=n_emp), rng.integers(1, 16, size=n_emp))
    horas_manuales = np.where(inv_emp == 1, rng.integers(20, 55, size=n_emp), rng.integers(2, 20, size=n_emp))
    presupuesto = np.where(inv_emp == 1, rng.integers(1800, 6500, size=n_emp), rng.integers(150, 1600, size=n_emp))

    df_empresario = pd.DataFrame({
        "empleados": empleados,
        "horas_manuales": horas_manuales,
        "presupuesto": presupuesto,
        "objetivo": inv_emp,
    })

    return df_cliente, df_empresario


DF_CLIENTE, DF_EMPRESARIO = build_datasets()


def run_knn(mode: str, values: dict, k: int = 3):
    if mode == "empresario":
        df = DF_EMPRESARIO
        cols = ["empleados", "horas_manuales", "presupuesto"]
        pos_detail = "Empresario invierte en software"
        neg_detail = "Empresario no invierte"
    else:
        df = DF_CLIENTE
        cols = ["valoracion", "tiempo_espera", "problemas"]
        pos_detail = "Cliente satisfecho"
        neg_detail = "Cliente no satisfecho"

    X = df[cols]
    y = df["objetivo"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    modelo = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    modelo.fit(X_train_sc, y_train)

    y_pred = modelo.predict(X_test_sc)
    acc = float(accuracy_score(y_test, y_pred) * 100)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    new_df = pd.DataFrame([{c: float(values.get(c, 0)) for c in cols}])
    new_sc = scaler.transform(new_df)

    pred_val = int(modelo.predict(new_sc)[0])
    distances, indices = modelo.kneighbors(new_sc, n_neighbors=k)

    y_train_arr = np.asarray(y_train)
    neighbors = []
    votes_yes = 0
    votes_no = 0
    for dist, idx in zip(distances[0], indices[0]):
        lbl = int(y_train_arr[idx])
        if lbl == 1:
            votes_yes += 1
        else:
            votes_no += 1
        neighbors.append({
            "label_text": "Sí" if lbl == 1 else "No",
            "distance": round(float(dist), 3),
        })

    return {
        "accuracy": f"{acc:.1f}%",
        "records": len(df),
        "k": k,
        "prediction_text": "SÍ" if pred_val == 1 else "NO",
        "prediction_detail": pos_detail if pred_val == 1 else neg_detail,
        "is_positive": pred_val == 1,
        "confusion_matrix": {
            "no_no": int(cm[0][0]),
            "no_yes": int(cm[0][1]),
            "yes_no": int(cm[1][0]),
            "yes_yes": int(cm[1][1]),
        },
        "neighbors": neighbors,
        "votes_yes": votes_yes,
        "votes_no": votes_no,
    }


class KNNLabHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = HTML_CONTENT.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/predict":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8") or "{}")
            mode = payload.get("mode", "cliente")
            values = payload.get("values", {})
            k = max(1, min(15, int(payload.get("k", 3))))
            result = run_knn(mode, values, k=k)
            body = json.dumps(result).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def main():
    host = "127.0.0.1"
    port = 5000
    server = ThreadingHTTPServer((host, port), KNNLabHandler)
    url = f"http://{host}:{port}"
    print(f"SENATI / KNN Lab ejecutándose en {url}")
    print("Presiona Ctrl + C para detener el servidor.")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
