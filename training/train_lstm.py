import os
import glob
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras as K

DATA_GLOB = "data/202*_nse-sec-bhavdata-full-*.csv"
SAVE_DIR = "models/lstm_v1"
SEQ_LEN = 90
BATCH = 256
EPOCHS = 50
LR = 1e-3
EMB_DIM = 16
VAL_START = "2025-01-01"
TEST_START = "2025-07-01"

COLS = [
    "SYMBOL","DATE1",
    "PREV_CLOSE","OPEN_PRICE","HIGH_PRICE","LOW_PRICE","CLOSE_PRICE",
    "TTL_TRD_QNTY","AVG_PRICE","NO_OF_TRADES","DELIV_PER"
]

RENAME = {
    "DATE1":"DATE",
    "TTL_TRD_QNTY":"VOLUME",
    "NO_OF_TRADES":"TRADES",
    "DELIV_PER":"DELIV_PCT",
    "PREV_CLOSE":"PREV",
    "OPEN_PRICE":"OPEN",
    "HIGH_PRICE":"HIGH",
    "LOW_PRICE":"LOW",
    "CLOSE_PRICE":"CLOSE",
    "AVG_PRICE":"AVG"
}

FEATURE_COLS = ["PREV", "OPEN", "HIGH", "LOW", "AVG" ,"VOLUME", "TRADES", "DELIV_PCT"]
TARGET_COL = "RET_FWD1"

np.random.seed(1327)
tf.random.set_seed(1327)

def load_all() -> pd.DataFrame:
    files = sorted(glob.glob(DATA_GLOB))
    if not files:
        raise FileNotFoundError(f"No files found matching {DATA_GLOB}")
    
    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        cols_present = [c for c in COLS if c in df.columns]
        df = df[cols_present].copy()
        for c in COLS:
            if c not in df.columns:
                df[c] = np.nan
        dfs.append(df)

    df = pd.concat(dfs, ignore_index = True)
    df = df.rename(columns = RENAME)

    df["DATE"] = pd.to_datetime(df["DATE"], errors = "coerce")
    df = df.dropna(subset = ["DATE", "SYMBOL", "CLOSE"])
    df = df.sort_values(["SYMBOL", "DATE"]).reset_index(drop=True)

    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors = "coerce")

    df[FEATURE_COLS + ["CLOSE"]] = df.groupby("SYMBOL")[FEATURE_COLS + ["CLOSE"]].transform("ffill")
    df = df.dropna(subset=FEATURE_COLS)

    df["RET_FWD1"] = df.groupby("SYMBOL")["CLOSE"].pct_change().shift(-1)

    df = df.dropna(subset = ["RET_FWD1"]).reset_index(drop=True)
    return df 



def date_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["DATE"] < VAL_START]
    val = df[(df["DATE"] >= VAL_START) & (df["DATE"] < TEST_START)]
    test = df[df["DATE"] >= TEST_START]
    return train, val, test

def build_windows(df: pd.DataFrame, sym2id: dict, seq_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs_num, xs_sym, ys = [], [], []

    for sym, g in df.groupby("SYMBOL", sort = False):
        g = g.sort_values("DATE")
        feat = g[FEATURE_COLS].values
        yret = g[TARGET_COL].values

        T = len(g)
        if T <= seq_len:
            continue

        for i in range(seq_len - 1 , T - 1):
            xs_num.append(feat[i - seq_len + 1: i+ 1])
            xs_sym.append(sym2id[sym])   
            ys.append(yret[i])

    X_num = np.asarray(xs_num, dtype = np.float32)
    X_sym = np.asarray(xs_sym, dtype = np.int32)
    y = np.asarray(ys, dtype = np.float32).reshape(-1, 1)
    return X_num, X_sym, y        

@dataclass
class ZScaler:
    mean_: np.ndarray
    std_: np.ndarray
    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / (self.std_ + 1e-8)




def fit_scaler_from_seq(X_num: np.ndarray) -> ZScaler:
    mean = X_num.reshape(-1, X_num.shape[-1]).mean(axis=0)
    std = X_num.reshape(-1, X_num.shape[-1]).std(axis=0)
    return ZScaler(mean, std)




def make_dataset(X_num, X_sym, y, batch, shuffle = True) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(((X_num, X_sym), y))
    if shuffle:
        ds = ds.shuffle(min(len(y), 100_000), seed = 1327)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds




def build_model(seq_len: int, n_feat: int, n_symbols: int, emb_dim = 16) -> K.Model:
    x_num = K.Input((seq_len, n_feat), name = "x_num")
    x_sym = K.Input(shape=(), dtype="int32", name="x_sym")   

    emb = K.layers.Embedding(input_dim = n_symbols,output_dim = emb_dim, name = "emb_sym")(x_sym)
    emb = K.layers.LayerNormalization(name = "ln_emb")(emb)

    h = K.layers.LSTM(128, return_sequences = False, name = "lstm")(x_num)
    h = K.layers.LayerNormalization()(h)

    z = K.layers.Concatenate()([h, emb])
    z = K.layers.Dense(128, activation = "relu")(z)
    z = K.layers.Dropout(0.2)(z)


    out = K.layers.Dense(1, name = "y")(z)
    model = K.Model(inputs=[x_num, x_sym], outputs=out)  

    model.compile(
        optimizer = K.optimizers.Adam(1e-3),
        loss = K.losses.Huber(delta = 0.01),
        metrics = [K.metrics.MeanAbsoluteError(name = "mae"), K.metrics.MeanSquaredError(name = "mse")]
    )

    return model

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Loading data…")
    df_all = load_all()

    # Symbol ids
    symbols = sorted(df_all["SYMBOL"].unique().tolist())
    sym2id = {s: i for i, s in enumerate(symbols)}
    n_symbols = len(symbols)
    print(f"Symbols: {n_symbols}")

    # Splits
    train_df, val_df, test_df = date_splits(df_all)
    print(f"Rows — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # Windows
    Xtr_raw, Str, ytr = build_windows(train_df, sym2id, SEQ_LEN)
    Xva_raw, Sva, yva = build_windows(val_df, sym2id, SEQ_LEN)
    Xte_raw, Ste, yte = build_windows(test_df, sym2id, SEQ_LEN)

    # Scale (fit on train only)
    scaler = fit_scaler_from_seq(Xtr_raw)
    Xtr = scaler.transform(Xtr_raw)
    Xva = scaler.transform(Xva_raw)
    Xte = scaler.transform(Xte_raw)

    # Datasets
    ds_tr = make_dataset(Xtr, Str, ytr, BATCH, shuffle=True)
    ds_va = make_dataset(Xva, Sva, yva, BATCH, shuffle=False)
    ds_te = make_dataset(Xte, Ste, yte, BATCH, shuffle=False)

    # Model
    model = build_model(SEQ_LEN, Xtr.shape[-1], n_symbols, EMB_DIM)
    model.summary()

    # Callbacks
    ckpt = K.callbacks.ModelCheckpoint(
        filepath=os.path.join(SAVE_DIR, "ckpt"),
        monitor="val_mae",
        mode="min",
        save_best_only=True,
        save_weights_only=True,
    )
    es = K.callbacks.EarlyStopping(monitor="val_mae", mode="min", patience=7, restore_best_weights=True)
    rlrop = K.callbacks.ReduceLROnPlateau(monitor="val_mae", mode="min", factor=0.5, patience=3, min_lr=1e-5)

    # Train
    history = model.fit(ds_tr, validation_data=ds_va, epochs=EPOCHS, callbacks=[ckpt, es, rlrop])

    # Evaluate
    print("Evaluating on test…")
    test_metrics = model.evaluate(ds_te, verbose=1)
    print("Test metrics:", dict(zip(model.metrics_names, test_metrics)))

    # Save model + scaler + mapping
    model.save(SAVE_DIR, include_optimizer=False)
    np.save(os.path.join(SAVE_DIR, "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(SAVE_DIR, "scaler_std.npy"), scaler.std_)
    pd.Series(sym2id).to_csv(os.path.join(SAVE_DIR, "sym2id.csv"))
    print(f"Saved model to {SAVE_DIR}")

if __name__ == "__main__":
    main()