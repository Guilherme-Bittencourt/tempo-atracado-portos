"""Perfil operacional da atracação por K-Means (ajustado só no treino)."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def ajustar_perfis(treino, teste, perfil_cols, cfg, saida):
    """Escolhe k (silhueta, ou k fixo), ajusta o K-Means e grava a coluna 'perfil' em treino e teste."""
    scaler = StandardScaler().fit(treino[perfil_cols])
    Z_tr = scaler.transform(treino[perfil_cols])
    idx_s = np.random.default_rng(cfg.seed).choice(len(Z_tr), size=min(20_000, len(Z_tr)), replace=False)
    ks = list(cfg.k_range)
    inercias, silhuetas = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=cfg.seed).fit(Z_tr)
        inercias.append(km.inertia_)
        silhuetas.append(silhouette_score(Z_tr[idx_s], km.labels_[idx_s]))
        print(f"k={k:2d}  inércia={km.inertia_:14,.1f}  silhueta={silhuetas[-1]:.4f}", flush=True)
    k_silhueta = ks[int(np.argmax(silhuetas))]
    k_usado = cfg.k_fixo or k_silhueta
    print(f">>> k de maior silhueta: {k_silhueta} | k usado: {k_usado} <<<")
    print("    Veja o gráfico do cotovelo: só se pode dizer que os métodos 'convergiram' "
          "se o cotovelo também estiver nesse k.")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(ks, inercias, "o-"); ax[0].set_xlabel("k"); ax[0].set_ylabel("Inércia")
    ax[0].set_title("Método do cotovelo")
    ax[1].plot(ks, silhuetas, "o-"); ax[1].axvline(k_usado, ls="--", c="gray")
    ax[1].set_xlabel("k"); ax[1].set_ylabel("Silhueta média"); ax[1].set_title("Método da silhueta")
    saida.fig(fig, "fig_kmeans_k.png")

    kmeans = KMeans(n_clusters=k_usado, n_init=10, random_state=cfg.seed).fit(Z_tr)
    treino["perfil"] = "P" + kmeans.labels_.astype(str)
    teste["perfil"] = "P" + kmeans.predict(scaler.transform(teste[perfil_cols])).astype(str)
    silhueta_k = silhuetas[ks.index(k_usado)] if k_usado in ks else float("nan")

    caract = treino.groupby("perfil").agg(
        n=("ta", "size"), ta_mediano_h=("ta", "median"), peso_mediano_t=("peso_total", "median"),
        teu_mediano=("teu_total", "median"), registros_carga_mediano=("n_registros_carga", "median"),
        natureza_mais_comum=("natureza", lambda s: s.mode().iat[0]))
    saida.csv(caract, "perfis_kmeans.csv")
    return {"scaler": scaler, "kmeans": kmeans, "k_silhueta": k_silhueta, "k_usado": k_usado,
            "silhueta": silhueta_k, "caracterizacao": caract}
